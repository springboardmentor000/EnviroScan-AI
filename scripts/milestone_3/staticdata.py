import osmnx as ox
import pandas as pd

cities = [

    # Delhi
    ("Delhi - Connaught Place", 28.63, 77.22, "urban"),
    ("Delhi - Rohini", 28.74, 77.10, "residential"),
    ("Delhi - Okhla Industrial", 28.52, 77.27, "industrial"),
    ("Delhi - Najafgarh", 28.61, 76.98, "rural"),

    # Mumbai
    ("Mumbai - Colaba", 18.91, 72.81, "urban"),
    ("Mumbai - Andheri", 19.12, 72.85, "residential"),
    ("Mumbai - Navi Mumbai", 19.03, 73.02, "industrial"),
    ("Mumbai - Vasai", 19.39, 72.83, "rural"),

    # Bangalore
    ("Bangalore - MG Road", 12.97, 77.61, "urban"),
    ("Bangalore - Whitefield", 12.98, 77.75, "residential"),
    ("Bangalore - Peenya", 13.02, 77.51, "industrial"),
    ("Bangalore - Devanahalli", 13.24, 77.71, "rural"),

    # Hyderabad
    ("Hyderabad - Hitech City", 17.45, 78.38, "urban"),
    ("Hyderabad - Banjara Hills", 17.41, 78.45, "residential"),
    ("Hyderabad - Patancheru", 17.53, 78.26, "industrial"),
    ("Hyderabad - Shadnagar", 17.07, 78.20, "rural"),

    # Chennai
    ("Chennai - T Nagar", 13.04, 80.23, "urban"),
    ("Chennai - Velachery", 12.98, 80.22, "residential"),
    ("Chennai - Ambattur", 13.11, 80.15, "industrial"),
    ("Chennai - Chengalpattu", 12.69, 79.98, "rural"),

    # Kolkata
    ("Kolkata - Park Street", 22.55, 88.35, "urban"),
    ("Kolkata - Salt Lake", 22.58, 88.42, "residential"),
    ("Kolkata - Howrah", 22.60, 88.31, "industrial"),
    ("Kolkata - Baruipur", 22.36, 88.44, "rural"),

    # Pune
    ("Pune - Shivaji Nagar", 18.53, 73.85, "urban"),
    ("Pune - Kharadi", 18.55, 73.93, "residential"),
    ("Pune - Pimpri", 18.62, 73.80, "industrial"),
    ("Pune - Talegaon", 18.73, 73.68, "rural"),

    # Ahmedabad
    ("Ahmedabad - CG Road", 23.03, 72.56, "urban"),
    ("Ahmedabad - Satellite", 23.02, 72.51, "residential"),
    ("Ahmedabad - Naroda", 23.07, 72.65, "industrial"),
    ("Ahmedabad - Sanand", 22.99, 72.38, "rural"),

    # Jaipur
    ("Jaipur - MI Road", 26.92, 75.79, "urban"),
    ("Jaipur - Vaishali Nagar", 26.91, 75.74, "residential"),
    ("Jaipur - Sitapura", 26.78, 75.84, "industrial"),
    ("Jaipur - Bagru", 26.82, 75.55, "rural"),

    # Lucknow
    ("Lucknow - Hazratganj", 26.85, 80.94, "urban"),
    ("Lucknow - Gomti Nagar", 26.87, 81.00, "residential"),
    ("Lucknow - Amausi", 26.76, 80.88, "industrial"),
    ("Lucknow - Malihabad", 26.92, 80.72, "rural"),

    # Chandigarh
    ("Chandigarh - Sector 17", 30.74, 76.78, "urban"),
    ("Chandigarh - Mohali", 30.70, 76.72, "residential"),
    ("Chandigarh - Industrial Area", 30.70, 76.80, "industrial"),
    ("Chandigarh - Zirakpur", 30.64, 76.82, "rural"),

    # Bhopal
    ("Bhopal - MP Nagar", 23.23, 77.43, "urban"),
    ("Bhopal - Arera Colony", 23.20, 77.43, "residential"),
    ("Bhopal - Govindpura", 23.25, 77.45, "industrial"),
    ("Bhopal - Sehore", 23.20, 77.08, "rural"),

    # Patna
    ("Patna - Fraser Road", 25.61, 85.14, "urban"),
    ("Patna - Kankarbagh", 25.60, 85.16, "residential"),
    ("Patna - Hajipur", 25.69, 85.21, "industrial"),
    ("Patna - Bihta", 25.55, 84.87, "rural"),

    # Kochi
    ("Kochi - Marine Drive", 9.97, 76.28, "urban"),
    ("Kochi - Kakkanad", 10.02, 76.33, "residential"),
    ("Kochi - Eloor", 10.07, 76.30, "industrial"),
    ("Kochi - Aluva", 10.10, 76.35, "rural"),
]

data = []

for name, lat, lon, area_type in cities:
    point = (lat, lon)

    roads = ox.geometries_from_point(point, tags={"highway": True}, dist=2000)
    road_count = len(roads)

    industries = ox.geometries_from_point(point, tags={"landuse": "industrial"}, dist=5000)
    industry_count = len(industries)

    agriculture = ox.geometries_from_point(point, tags={"landuse": "farmland"}, dist=5000)
    agriculture_count = len(agriculture)

    dumps = ox.geometries_from_point(point, tags={"landuse": "landfill"}, dist=5000)
    dump_count = len(dumps)

    data.append({
        "city": name,
        "latitude": lat,
        "longitude": lon,
        "road_count_2km": road_count,
        "industry_count_5km": industry_count,
        "agriculture_count_5km": agriculture_count,
        "dump_count_5km": dump_count
    })

df = pd.DataFrame(data)
df.to_csv("../data/static_features_osm.csv", index=False)

print("Static features saved from OSM")