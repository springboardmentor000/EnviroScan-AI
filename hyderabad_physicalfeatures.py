import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# ── All 2 stations matching your datasets ──
STATIONS = [
    {"id": 344140, "name": "Somajiguda, Hyderabad - TSPCB", "lat": 17.417094, "lon": 78.457437},
    {"id": 407, "name": "Zoo Park, Hyderabad - TSPCB", "lat": 17.349694, "lon": 78.451437},
    {"id": 5647, "name": "Sanathnagar, Hyderabad - TSPCB", "lat": 17.4559458, "lon": 78.4332152},
    {"id": 5623, "name": "Central University, Hyderabad - TSPCB", "lat": 17.460103, "lon": 78.334361},
    {"id": 344104, "name": "Kompally Municipal Office, Hyderabad - TSPCB", "lat": 17.544899, "lon": 78.486949}
    ]

RADIUS = 7000  # 7 km radius around each station (meters)
CITY   = "Vijayawada"

# ── OSM tags to extract ──
OSM_TAGS = {
    "road":       {"highway": ["motorway", "trunk", "primary", "secondary",
                               "tertiary", "residential", "unclassified"]},
    "industrial": {"landuse": "industrial"},
    "waste":      {"landuse": ["landfill", "brownfield"],
                   "amenity": "waste_disposal",
                   "man_made": "wastewater_plant"},
    "farmland":   {"landuse": ["farmland", "farmyard", "orchard", "meadow"],
                        "natural": ["grassland"],
                        "landcover": ["farmland"]}
}


# ──────────────────────────────────────────
def count_features(lat, lon, tags, radius=RADIUS):
    """Count OSM features within radius meters of a point."""
    point = (lat, lon)
    counts = {}
    for feature_name, tag_dict in tags.items():
        total = 0
        # Handle dict of lists vs simple dict
        if isinstance(list(tag_dict.values())[0], list):
            key   = list(tag_dict.keys())[0]
            vals  = list(tag_dict.values())[0]
            query = {key: vals}
        else:
            query = tag_dict

        try:
            gdf = ox.features_from_point(point, tags=query, dist=radius)
            total = len(gdf)
        except Exception:
            total = 0  # No features found

        counts[f"{feature_name}_count"] = total
    return counts

# ──────────────────────────────────────────
# Loop all stations
# ──────────────────────────────────────────
records = []

for s in STATIONS:
    print(f"\n📍 Processing: {s['name']}")

    # Roads
    print("   🛣️  Counting roads...")
    try:
        G = ox.graph_from_point(
            (s["lat"], s["lon"]),
            dist=RADIUS,
            network_type="drive"
        )
        road_count = len(G.edges())
    except Exception as e:
        print(f"   ⚠️  Roads error: {e}")
        road_count = 0
    print(f"      → {road_count} road segments")

    # Industrial zones
    print("   🏭 Counting industrial zones...")
    try:
        gdf_ind = ox.features_from_point(
            (s["lat"], s["lon"]),
            tags={"landuse": "industrial"},
            dist=RADIUS
        )
        industrial_count = len(gdf_ind)
    except Exception:
        industrial_count = 0
    print(f"      → {industrial_count} industrial features")

    # Dump / waste sites
    print("   🗑️  Counting waste/dump sites...")
    try:
        gdf_waste = ox.features_from_point(
            (s["lat"], s["lon"]),
            tags={"landuse": ["landfill", "brownfield"]},
            dist=RADIUS
        )
        waste_count = len(gdf_waste)
    except Exception:
        waste_count = 0
    try:
        gdf_waste2 = ox.features_from_point(
            (s["lat"], s["lon"]),
            tags={"amenity": "waste_disposal"},
            dist=RADIUS
        )
        waste_count += len(gdf_waste2)
    except Exception:
        pass
    print(f"      → {waste_count} waste/dump features")

    # Agricultural / farmland
    print("   🌾 Counting farmland...")
    try:
        gdf_farm = ox.features_from_point(
            (s["lat"], s["lon"]),
            tags={"landuse": ["farmland", "farmyard", "orchard",
                              "meadow", "greenhouse_horticulture"]},
            dist=RADIUS
        )
        farmland_count = len(gdf_farm)
    except Exception:
        farmland_count = 0
    print(f"      → {farmland_count} farmland features")

    records.append({
        "city":             CITY,
        "location_id":      s["id"],
        "location":         s["name"],
        "lat":              s["lat"],
        "lon":              s["lon"],
        "radius_m":         RADIUS,
        "road_count":       road_count,
        "industrial_count": industrial_count,
        "waste_count":      waste_count,
        "farmland_count":   farmland_count,
    })

# ──────────────────────────────────────────
# Save
# ──────────────────────────────────────────
df = pd.DataFrame(records)
print(f"\n✅ OSM Features extracted for {len(df)} stations:")
print(df[["location", "road_count", "industrial_count",
          "waste_count", "farmland_count"]].to_string(index=False))

df.to_csv("Hyderabad_physical_features.csv", index=False)
print("\n💾 Saved → hyderabad_osm_features.csv")
