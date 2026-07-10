"""
Melbourne Bus + Train + Tram Reform - adds real train and tram lines from
GTFS on top of the existing hand-drawn bus network, without altering bus
route lines or moving any train/tram stop. Trains and trams each run at
their own uniform speed and frequency (10 min headway for both).

Train and tram route lines come straight from the supplied GTFS shapes.txt
- nothing is resampled or redrawn for rail/tram. Each route's stop sequence
is pulled from a real trip's stop_times.txt entries for the chosen shape,
so it's exact, ordered, and free of duplicates (e.g. it won't pick up the
opposite direction's stop that happens to sit a few metres from this
direction's line). Proximity-based matching against stops.txt is only used
as a fallback if trip/stop_times data isn't available for a route.
"""

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge

import csv
import json
import math
from dataclasses import dataclass, field
from itertools import combinations

from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

BUS_GEOJSON = f"data/routes.geojson"
GTFS_ROUTES = f"gtfs/2/google_transit/routes.txt"
GTFS_SHAPES = f"gtfs/2/google_transit/shapes.txt"
GTFS_STOPS = f"gtfs/2/google_transit/stops.txt"
GTFS_TRIPS = f"gtfs/2/google_transit/trips.txt"
GTFS_STOP_TIMES = f"gtfs/2/google_transit/stop_times.txt"
GTFS_TRAM_ROUTES = f"gtfs/3/google_transit/routes.txt"
GTFS_TRAM_SHAPES = f"gtfs/3/google_transit/shapes.txt"
GTFS_TRAM_STOPS = f"gtfs/3/google_transit/stops.txt"
GTFS_TRAM_TRIPS = f"gtfs/3/google_transit/trips.txt"
GTFS_TRAM_STOP_TIMES = f"gtfs/3/google_transit/stop_times.txt"
SRL_GEOJSON = f"data/srl.geojson"

OUTPUT_GRAPH = f"data/graph.json"
OUTPUT_ROUTES_GEOJSON = f"data/routes.geojson"
OUTPUT_STOPS_DEBUG = f"data/routes_with_stops_debug.geojson"

BUS_SPEED_KMH = 25.0
TRAIN_SPEED_KMH = 40.0         # express-ish average incl. dwell; only affects ride time, not lines/stops
SRL_SPEED_KMH = 62.0           # Suburban Rail Loop: modern underground metro, wider stop spacing
                                # than the legacy network, so a higher average incl. dwell is reasonable
WALK_SPEED_M_PER_MIN = 80.0
STOP_SPACING_M = 400.0          # bus resampling only, unchanged from original
SNAP_TOLERANCE_M = 25.0         # geometric line-crossing cluster tolerance (bus<->bus, bus<->rail)
STATION_SNAP_TOLERANCE_M = 80.0 # how close a real GTFS station must be to a rail shape to "belong" to it
MIN_STOP_SEPARATION_M = 150.0
INTERCHANGE_PENALTY_MIN = 2.0
BOARD_PENALTY_MIN = 4.0        # routing-only overhead for catching any service at all (walking to
                                # the platform/stop, doors, settling in). Added to the "cost_min"
                                # used for pathfinding but NOT to the displayed "weight_min" wait
                                # time, so the router won't recommend hopping on a second service
                                # just to save one stop, without the itinerary showing a fake-long wait.
TRAIN_FREQUENCY = 10
B1_FREQUENCY = 5
B2_FREQUENCY = 10
SRL_FREQUENCY = 5
TRAM_SPEED_KMH = 20.0          # universal tram speed (avg incl. stops/dwell)
TRAM_FREQUENCY = 10            # universal tram frequency, same treatment as trains
TRAM_COLOR = "#91DE56"

WGS84 = "EPSG:4326"
METRIC = "EPSG:28355"  # GDA94 / MGA zone 55 - accurate for Melbourne

to_metric = Transformer.from_crs(WGS84, METRIC, always_xy=True).transform
to_wgs84 = Transformer.from_crs(METRIC, WGS84, always_xy=True).transform


# ---------------------------------------------------------------------------
# Load bus routes (unchanged lines, from the existing hand-drawn geojson)
# ---------------------------------------------------------------------------

def load_bus_routes(path):
    data = json.load(open(path))
    routes = {}
    for f in data["features"]:
        props = f["properties"]
        route_id = props["route"]
        corridor = props["corridor"]
        geom = f["geometry"]
        if geom["type"] == "MultiLineString":
            merged = linemerge(MultiLineString(geom["coordinates"]))
            if merged.geom_type != "LineString":
                raise ValueError(
                    f"Route {route_id} MultiLineString parts don't connect end-to-end "
                    f"(got {merged.geom_type}) — check the geometry for gaps."
                )
            line_wgs = merged
        else:
            line_wgs = LineString(geom["coordinates"])
        line_m = shapely_transform(to_metric, line_wgs)
        routes[route_id] = {
            "route_id": route_id,
            "corridor": corridor,
            "mode": "bus",
            "frequency_min": B1_FREQUENCY if corridor == 'B1' else B2_FREQUENCY,
            "speed_kmh": BUS_SPEED_KMH,
            "line_m": line_m,
            "line_wgs_coords": list(line_wgs.coords),
        }
    return routes


# ---------------------------------------------------------------------------
# Load train routes from GTFS: one real shape per route, real stations only
# ---------------------------------------------------------------------------

def gtfs_route_code(route_id):
    # "aus:vic:vic-02-ALM:" -> "ALM"
    core = route_id.strip(":").split(":")[-1]  # "vic-02-ALM"
    return core.split("-")[-1]


def load_gtfs_train_routes(routes_txt, shapes_txt, stops_txt, mode="rail",
                            corridor="RAIL", id_prefix="RAIL",
                            speed_kmh=TRAIN_SPEED_KMH, frequency_min=TRAIN_FREQUENCY,
                            color=None, stop_location_types=("1",),
                            trips_txt=None, stop_times_txt=None):
    # 1. which route codes count as real train/tram lines (skip rail-replacement buses)
    train_routes = {}
    with open(routes_txt, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["route_short_name"] == "Replacement Bus":
                continue
            code = gtfs_route_code(row["route_id"])
            train_routes[code] = {
                "route_id": row["route_id"],
                "short_name": row["route_long_name"],
                "route_number": row["route_short_name"],
            }

    # 2. group shape points by shape_id
    shape_pts = {}
    with open(shapes_txt, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            shape_pts.setdefault(row["shape_id"], []).append(
                (int(row["shape_pt_sequence"]), float(row["shape_pt_lat"]), float(row["shape_pt_lon"]))
            )

    def shape_code(shape_id):
        parts = shape_id.split("-")
        return parts[1] if len(parts) > 1 else None

    def haversine_len(coords_latlon):
        R = 6371000
        total = 0.0
        for (lat1, lon1), (lat2, lon2) in zip(coords_latlon, coords_latlon[1:]):
            p1, p2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlmb = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
            total += 2 * R * math.asin(math.sqrt(a))
        return total

    # 3. pick the longest shape for each route code -> canonical real line
    #    (keep the shape_id itself too, so we can look up the exact trip that used it)
    shapes_by_code = {}
    for shape_id, pts in shape_pts.items():
        code = shape_code(shape_id)
        if code not in train_routes:
            continue
        pts.sort(key=lambda t: t[0])
        coords_latlon = [(lat, lon) for _, lat, lon in pts]
        length = haversine_len(coords_latlon)
        best = shapes_by_code.get(code)
        if best is None or length > best[0]:
            shapes_by_code[code] = (length, coords_latlon, shape_id)

    # 4. load real stops; for each route, find stop sequence from trip/stop_times or by proximity
    real_stops = {}
    with open(stops_txt, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            stop_id = row["stop_id"]
            real_stops[stop_id] = {
                "name": row["stop_name"],
                "lat": float(row["stop_lat"]),
                "lon": float(row["stop_lon"]),
                "location_type": row.get("location_type", ""),
            }

    # 5. for each route shape, determine its stop sequence from trips/stop_times if available
    #    (this is *actual* stop order for a given shape), otherwise fall back to proximity matching
    real_stops_by_route = {}
    if trips_txt and stop_times_txt:
        # load trips -> shape_id mapping for this mode
        trips_by_shape = {}
        with open(trips_txt, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                shape_id = row.get("shape_id")
                if shape_id:
                    if shape_id not in trips_by_shape:
                        trips_by_shape[shape_id] = []
                    trips_by_shape[shape_id].append(row["trip_id"])

        # for each shape, pick one trip and read its stop_times to get exact sequence
        trip_stop_seqs = {}
        with open(stop_times_txt, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                trip_id = row["trip_id"]
                if trip_id not in trip_stop_seqs:
                    trip_stop_seqs[trip_id] = []
                trip_stop_seqs[trip_id].append((int(row["stop_sequence"]), row["stop_id"]))

        # for each route's canonical shape, lookup stop sequence from one trip
        for code, route_info in train_routes.items():
            _, coords_latlon, shape_id = shapes_by_code.get(code, (None, None, None))
            if not coords_latlon:
                continue
            trips_for_shape = trips_by_shape.get(shape_id, [])
            if trips_for_shape:
                # pick any trip on this shape, read its stop sequence
                trip_id = trips_for_shape[0]
                stops = sorted(trip_stop_seqs.get(trip_id, []))
                real_stops_by_route[code] = [
                    (seq, sid) for seq, sid in stops if sid in real_stops
                ]
                if real_stops_by_route[code]:
                    continue

            # fallback: proximity match stops to shape endpoints
            if coords_latlon:
                start_pt = coords_latlon[0]
                end_pt = coords_latlon[-1]
                best_start = min(real_stops.items(),
                                key=lambda x: (x[1]["lat"] - start_pt[0]) ** 2 + (x[1]["lon"] - start_pt[1]) ** 2)
                best_end = min(real_stops.items(),
                              key=lambda x: (x[1]["lat"] - end_pt[0]) ** 2 + (x[1]["lon"] - end_pt[1]) ** 2)
                real_stops_by_route[code] = [(0, best_start[0]), (1, best_end[0])]

    # 6. Build route dicts with display info
    routes = {}
    for code, route_info in train_routes.items():
        if code not in shapes_by_code:
            continue
        _, coords_latlon, _ = shapes_by_code[code]
        line_wgs = LineString([(lon, lat) for lat, lon in coords_latlon])
        line_m = shapely_transform(to_metric, line_wgs)
        route_id = f"{id_prefix}:{code}"
        routes[route_id] = {
            "route_id": route_id,
            "corridor": corridor,
            "mode": mode,
            "frequency_min": frequency_min,
            "speed_kmh": speed_kmh,
            "line_m": line_m,
            "line_wgs_coords": list(line_wgs.coords),
            "short_name": route_info["short_name"],
            "display_label": f'{route_info["route_number"]} — {route_info["short_name"]}',
        }
        if color:
            routes[route_id]["color"] = color

    return routes, real_stops_by_route, real_stops


def load_srl_route(path):
    """Load the single SRL route from hand-drawn GeoJSON (like bus routes)."""
    data = json.load(open(path))
    srl_routes = {}
    for f in data["features"]:
        props = f["properties"]
        # Match whichever key the geojson actually uses — bus routes.geojson
        # uses "route", but tolerate "route_id" too in case the SRL file
        # was authored with a different convention.
        route_id = props.get("route") or props.get("route_id")
        if route_id is None:
            raise KeyError(
                f"SRL feature has neither 'route' nor 'route_id' in properties: {props}"
            )
        geom = f["geometry"]
        if geom["type"] == "MultiLineString":
            merged = linemerge(MultiLineString(geom["coordinates"]))
            if merged.geom_type != "LineString":
                raise ValueError(f"SRL geometry broken")
            line_wgs = merged
        else:
            line_wgs = LineString(geom["coordinates"])
        line_m = shapely_transform(to_metric, line_wgs)
        srl_routes[route_id] = {
            "route_id": route_id,
            "corridor": "SRL",
            "mode": "rail",
            "frequency_min": SRL_FREQUENCY,
            "speed_kmh": SRL_SPEED_KMH,
            "line_m": line_m,
            "line_wgs_coords": list(line_wgs.coords),
            "short_name": "Suburban Rail Loop",
            "display_label": "Suburban Rail Loop",
            "color": "#008746",
        }

    # Seed 12 stops evenly spaced along the SRL line, each with a real
    # position (interpolated along the metric geometry) and a name, so
    # they behave exactly like GTFS-sourced stops downstream in build_stops.
    srl_stops_by_route = {}
    srl_stop_info = {}
    for route_id, route_data in srl_routes.items():
        line = route_data["line_m"]
        code = route_id.split(":")[-1]
        seq = []
        n_stops = 12
        for i in range(n_stops):
            frac = i / (n_stops - 1) if n_stops > 1 else 0
            pt_m = line.interpolate(frac, normalized=True)
            lon, lat = to_wgs84(pt_m.x, pt_m.y)
            stop_id = f"SRL_STOP_{i:02d}"
            srl_stop_info[stop_id] = {
                "name": f"Suburban Rail Loop Station {i + 1}",
                "lat": lat,
                "lon": lon,
                "location_type": "1",
            }
            seq.append((i, stop_id))
        srl_stops_by_route[code] = seq

    return srl_routes, srl_stops_by_route, srl_stop_info


# ---------------------------------------------------------------------------
# Core geometry: find route intersections, sample bus route line to stops
# ---------------------------------------------------------------------------

@dataclass
class Stop:
    x: float
    y: float
    lat: float
    lon: float
    name: str = ""
    routes: set = field(default_factory=set)
    is_interchange: bool = False


def resample_bus_line(line, spacing):
    """Sample a line at regular ~spacing intervals."""
    total = line.length
    samples = []
    for i in range(int(total / spacing) + 1):
        dist = i * spacing
        if dist <= total:
            pt = line.interpolate(dist)
            samples.append((pt.x, pt.y))
    return samples


def find_interchanges(routes):
    """Find clusters of route-crossing points (bus<->bus, bus<->rail, rail<->tram)."""
    crossing_pts = []
    route_ids = list(routes.keys())
    for i, rid_a in enumerate(route_ids):
        for rid_b in route_ids[i + 1:]:
            geom = routes[rid_a]["line_m"].intersection(routes[rid_b]["line_m"])
            if geom.is_empty:
                continue
            if geom.geom_type == "Point":
                crossing_pts.append((geom.x, geom.y, {rid_a, rid_b}))
            elif geom.geom_type == "MultiPoint":
                for pt in geom.geoms:
                    crossing_pts.append((pt.x, pt.y, {rid_a, rid_b}))
            elif geom.geom_type == "LineString":
                # Two routes overlap for a segment; sample the midpoint
                pt = geom.interpolate(0.5, normalized=True)
                crossing_pts.append((pt.x, pt.y, {rid_a, rid_b}))

    # Cluster nearby crossing points using single-linkage clustering.
    # Each cluster is a dict: {"points": [(x, y), ...], "routes": set(...)}
    clusters = []
    for x, y, route_set in crossing_pts:
        merged = False
        for cluster in clusters:
            cx = sum(p[0] for p in cluster["points"]) / len(cluster["points"])
            cy = sum(p[1] for p in cluster["points"]) / len(cluster["points"])
            if math.hypot(x - cx, y - cy) <= SNAP_TOLERANCE_M:
                cluster["points"].append((x, y))
                cluster["routes"] |= route_set
                merged = True
                break
        if not merged:
            clusters.append({"points": [(x, y)], "routes": set(route_set)})

    # Return cluster centroids (only clusters actually involving 2+ crossings
    # are meaningful interchanges; a single crossing point is still valid —
    # every entry here came from an intersection of exactly 2 routes at minimum)
    result = []
    for cluster in clusters:
        coords = cluster["points"]
        cx = sum(x for x, y in coords) / len(coords)
        cy = sum(y for x, y in coords) / len(coords)
        result.append({"x": cx, "y": cy, "routes": cluster["routes"]})
    return result


def build_stops(routes, interchanges, real_stops_by_route, real_stop_info):
    """
    Build stop dict and route stop sequences.
    For each route:
      - Real stops (train/tram/SRL) use their GTFS (or SRL-generated) names
      - Bus stops get sequential names: "Stop #1", "Stop #2", etc.
      - Stops on opposite sides of the road get the same name
    """
    stops = {}
    route_stop_sequences = {}

    # Map bare GTFS route codes (e.g. "ALM") back to their full prefixed
    # route_id (e.g. "RAIL:ALM") once, rather than re-scanning routes for
    # every stop sequence.
    code_to_full_id = {}
    for rid in routes.keys():
        if ":" in rid:
            code_to_full_id[rid.split(":", 1)[1]] = rid

    # 1. Process train/tram/SRL routes: use real (GTFS or SRL-generated) stop names
    for route_id, seq in real_stops_by_route.items():
        route_full_id = code_to_full_id.get(route_id, route_id)

        route_stop_sequences[route_full_id] = []
        for order, stop_id in seq:
            info = real_stop_info.get(stop_id)
            if info is None:
                # Shouldn't happen — every stop_id in real_stops_by_route
                # should have a matching entry in real_stop_info — but skip
                # gracefully rather than crash if GTFS data is inconsistent.
                continue
            if stop_id not in stops:
                x, y = to_metric(info["lon"], info["lat"])
                stops[stop_id] = Stop(
                    x=x,
                    y=y,
                    lat=info["lat"],
                    lon=info["lon"],
                    name=info["name"],
                )
            stops[stop_id].routes.add(route_full_id)
            route_stop_sequences[route_full_id].append((stop_id, 0))

    # 2. Process bus routes: resample line, create sequential stop names per route
    for route_id, route_info in routes.items():
        if route_info["mode"] != "bus":
            continue

        line = route_info["line_m"]
        pts = resample_bus_line(line, STOP_SPACING_M)
        
        # Remove duplicates and ensure minimum separation
        filtered = []
        for x, y in pts:
            if not filtered or math.hypot(x - filtered[-1][0], y - filtered[-1][1]) >= MIN_STOP_SEPARATION_M:
                filtered.append((x, y))
        
        route_stop_sequences[route_id] = []
        for stop_num, (x, y) in enumerate(filtered, 1):
            # Create a stop_id unique to this route and position
            stop_id = f"{route_id}__STOP_{stop_num:03d}"
            
            # Use sequential naming: "Stop #1", "Stop #2", etc.
            stop_name = f"Stop #{stop_num}"
            
            lon, lat = to_wgs84(x, y)
            if stop_id not in stops:
                stops[stop_id] = Stop(
                    x=x,
                    y=y,
                    lat=lat,
                    lon=lon,
                    name=stop_name,
                )
            else:
                # Stop already exists (possible at interchanges); reuse it
                stops[stop_id].name = stop_name
            
            stops[stop_id].routes.add(route_id)
            route_stop_sequences[route_id].append((stop_id, 0))

    # 3. Mark interchanges and ensure stops share names when on same corridor
    for ic in interchanges:
        candidates = []
        for sid, stop in stops.items():
            if math.hypot(stop.x - ic["x"], stop.y - ic["y"]) < STATION_SNAP_TOLERANCE_M:
                candidates.append((sid, math.hypot(stop.x - ic["x"], stop.y - ic["y"])))
        candidates.sort(key=lambda x: x[1])
        for sid, _ in candidates[:3]:  # Mark closest 3 as interchange
            stops[sid].is_interchange = True

    return stops, route_stop_sequences


# ---------------------------------------------------------------------------
# Graph construction: nodes, edges, Dijkstra setup
# ---------------------------------------------------------------------------

def build_graph(routes, stops, route_stop_sequences):
    nodes = {}
    edges = []

    def add_node(node_id, stop_id, node_type, route_id=None):
        if node_id in nodes:
            return
        if stop_id not in stops:
            return
        stop = stops[stop_id]
        nodes[node_id] = {
            "lat": stop.lat,
            "lon": stop.lon,
            "type": node_type,
            "stop_id": stop_id,
        }
        if route_id:
            nodes[node_id]["route"] = route_id

    def hub_in_id(stop_id):
        return f"{stop_id}__HUB_IN"

    def hub_out_id(stop_id):
        return f"{stop_id}__HUB_OUT"

    def route_node_id(stop_id, route_id):
        return f"{stop_id}__{route_id}"

    for stop_id in stops:
        add_node(hub_in_id(stop_id), stop_id, "hub_in")
        add_node(hub_out_id(stop_id), stop_id, "hub_out")
        edges.append({"from": hub_in_id(stop_id), "to": hub_out_id(stop_id), "type": "walk_through", "weight_min": 0})

    for stop_id, stop in stops.items():
        hin, hout = hub_in_id(stop_id), hub_out_id(stop_id)
        served = sorted(stop.routes)
        for r in served:
            rnid = route_node_id(stop_id, r)
            add_node(rnid, stop_id, "route", route_id=r)
            freq = routes[r]["frequency_min"]
            edges.append({"from": hin, "to": rnid, "type": "board", "route": r,
                          "weight_min": round(freq / 2, 3),
                          "cost_min": round(freq / 2 + BOARD_PENALTY_MIN, 3)})
            edges.append({"from": rnid, "to": hout, "type": "alight", "route": r, "weight_min": 0})

        for r_from in served:
            for r_to in served:
                if r_from == r_to:
                    continue
                freq_to = routes[r_to]["frequency_min"]
                edges.append({
                    "from": route_node_id(stop_id, r_from), "to": route_node_id(stop_id, r_to),
                    "type": "transfer", "route": r_to,
                    "weight_min": round(freq_to / 2 + INTERCHANGE_PENALTY_MIN, 3),
                })

    for route_id, seq in route_stop_sequences.items():
        speed_m_per_min = routes[route_id]["speed_kmh"] * 1000 / 60
        for (sid_a, dist_a), (sid_b, dist_b) in zip(seq, seq[1:]):
            ride_min = (dist_b - dist_a) / speed_m_per_min
            edges.append({"from": route_node_id(sid_a, route_id), "to": route_node_id(sid_b, route_id),
                           "type": "ride", "route": route_id, "weight_min": round(ride_min, 3)})
            edges.append({"from": route_node_id(sid_b, route_id), "to": route_node_id(sid_a, route_id),
                           "type": "ride", "route": route_id, "weight_min": round(ride_min, 3)})

    return nodes, edges


def add_walk_transfer_edges(nodes, edges, stops, max_walk_m=500):
    hub_stops = list(stops.items())
    added = 0
    for (sid_a, a), (sid_b, b) in combinations(hub_stops, 2):
        if a.routes & b.routes:
            continue
        d = math.hypot(a.x - b.x, a.y - b.y)
        if 0 < d <= max_walk_m:
            walk_min = d / WALK_SPEED_M_PER_MIN
            edges.append({"from": f"{sid_a}__HUB_OUT", "to": f"{sid_b}__HUB_IN", "type": "walk", "weight_min": round(walk_min, 3)})
            edges.append({"from": f"{sid_b}__HUB_OUT", "to": f"{sid_a}__HUB_IN", "type": "walk", "weight_min": round(walk_min, 3)})
            added += 2
    return added


def export_routes_geojson(routes, path):
    features = []
    for rid, r in routes.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [list(c) for c in r["line_wgs_coords"]]},
            "properties": {"route": rid, "corridor": r["corridor"], "mode": r["mode"],
                            "short_name": r.get("short_name", rid),
                            "display_label": r.get("display_label", r.get("short_name", rid))},
        })
    json.dump({"type": "FeatureCollection", "features": features}, open(path, "w"), indent=1)


def export_debug_geojson(stops, path):
    features = []
    for sid, s in stops.items():
        lon, lat = to_wgs84(s.x, s.y)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"stop_id": sid, "is_interchange": s.is_interchange, "routes": sorted(s.routes),
                            "name": s.name},
        })
    json.dump({"type": "FeatureCollection", "features": features}, open(path, "w"), indent=1)


def main():
    import os
    os.makedirs(f"data", exist_ok=True)

    bus_routes = load_bus_routes(BUS_GEOJSON)
    print(f"Loaded {len(bus_routes)} bus routes (lines unchanged)")

    train_routes, real_stops_by_route, real_stop_info = load_gtfs_train_routes(
        GTFS_ROUTES, GTFS_SHAPES, GTFS_STOPS,
        trips_txt=GTFS_TRIPS, stop_times_txt=GTFS_STOP_TIMES,
    )
    print(f"Loaded {len(train_routes)} train routes from GTFS")
    total_stations = len(set(sid for entries in real_stops_by_route.values() for _, sid, *_ in entries))
    print(f"Matched {total_stations} distinct real stations across those lines")

    srl_routes, srl_real_stops, srl_stop_info = load_srl_route(SRL_GEOJSON)
    print(f"Loaded {len(srl_routes)} SRL route ({len(next(iter(srl_real_stops.values())))} stops)")
    train_routes.update(srl_routes)
    real_stops_by_route.update(srl_real_stops)
    real_stop_info.update(srl_stop_info)

    tram_routes, tram_real_stops, tram_stop_info = load_gtfs_train_routes(
        GTFS_TRAM_ROUTES, GTFS_TRAM_SHAPES, GTFS_TRAM_STOPS,
        mode="tram", corridor="TRAM", id_prefix="TRAM",
        speed_kmh=TRAM_SPEED_KMH, frequency_min=TRAM_FREQUENCY,
        color=TRAM_COLOR, stop_location_types=("", "0"),
        trips_txt=GTFS_TRAM_TRIPS, stop_times_txt=GTFS_TRAM_STOP_TIMES,
    )
    print(f"Loaded {len(tram_routes)} tram routes from GTFS")
    total_tram_stops = len(set(sid for entries in tram_real_stops.values() for _, sid, *_ in entries))
    print(f"Matched {total_tram_stops} distinct real tram stops across those lines")
    train_routes.update(tram_routes)
    real_stops_by_route.update(tram_real_stops)
    real_stop_info.update(tram_stop_info)

    routes = {**bus_routes, **train_routes}

    interchanges = find_interchanges(routes)
    print(f"Found {len(interchanges)} clustered geometric interchange points")

    stops, route_stop_sequences = build_stops(routes, interchanges, real_stops_by_route, real_stop_info)
    n_ic = sum(1 for s in stops.values() if s.is_interchange)
    print(f"Total stops: {len(stops)} ({n_ic} interchange, {len(stops) - n_ic} regular)")

    nodes, edges = build_graph(routes, stops, route_stop_sequences)
    n_walk = add_walk_transfer_edges(nodes, edges, stops)
    print(f"Added {n_walk} walk-transfer edges")
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    n_named = sum(1 for s in stops.values() if s.name)
    print(f"{n_named} stops have a name (GTFS or sequential)")

    graph = {
        "meta": {
            "bus_speed_kmh": BUS_SPEED_KMH,
            "train_speed_kmh": TRAIN_SPEED_KMH,
            "srl_speed_kmh": SRL_SPEED_KMH,
            "tram_speed_kmh": TRAM_SPEED_KMH,
            "walk_speed_m_per_min": WALK_SPEED_M_PER_MIN,
            "interchange_penalty_min": INTERCHANGE_PENALTY_MIN,
            "board_penalty_min": BOARD_PENALTY_MIN,
            "stop_spacing_m": STOP_SPACING_M,
            "train_frequency": TRAIN_FREQUENCY,
            "B1_frequency": B1_FREQUENCY,
            "B2_frequency": B2_FREQUENCY,
            "tram_frequency": TRAM_FREQUENCY
        },
        "routes": {rid: {"frequency_min": r["frequency_min"], "corridor": r["corridor"], "mode": r["mode"],
                          **({"color": r["color"]} if "color" in r else {}),
                          **({"display_label": r["display_label"]} if "display_label" in r else {})}
                   for rid, r in routes.items()},
        "stop_names": {sid: s.name for sid, s in stops.items() if s.name},
        "nodes": nodes,
        "edges": edges,
    }
    json.dump(graph, open(OUTPUT_GRAPH, "w"))
    print(f"Wrote {OUTPUT_GRAPH}")

    export_routes_geojson(routes, OUTPUT_ROUTES_GEOJSON)
    print(f"Wrote {OUTPUT_ROUTES_GEOJSON}")

    export_debug_geojson(stops, OUTPUT_STOPS_DEBUG)
    print(f"Wrote {OUTPUT_STOPS_DEBUG}")


if __name__ == "__main__":
    main()