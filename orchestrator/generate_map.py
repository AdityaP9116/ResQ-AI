import json
import argparse
import os
import folium
import base64

def get_base64_image(image_path):
    """Reads an image file and returns its Base64 encoding for HTML embedding."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None

def _world_xyz_to_latlon(x: float, y: float, z: float, base_lat: float = 37.0, base_lon: float = -122.0):
    """Convert sim world x,y (metres) to approximate lat/lon for map display."""
    import math
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(base_lat))
    lat = base_lat + y / m_per_deg_lat
    lon = base_lon + x / m_per_deg_lon
    return lat, lon


def build_map(report_path, use_world_coords: bool = True):
    print(f"Reading Flight Report from: {report_path}")

    if not os.path.exists(report_path):
        print(f"Error: Flight report not found at {report_path}")
        return

    with open(report_path, "r") as f:
        data = json.load(f)

    # Support both formats: flat list (legacy) or dict with "hazards" key (demo_flight)
    if isinstance(data, list):
        hazards = data
    elif isinstance(data, dict) and "hazards" in data:
        hazards = data["hazards"]
    else:
        hazards = []

    if not hazards:
        print("Flight report is empty. No hazards were detected.")
        return

    # Compute map center
    lats, lons = [], []
    for h in hazards:
        if "latitude" in h and "longitude" in h:
            lats.append(h["latitude"])
            lons.append(h["longitude"])
        elif "world_xyz" in h and h["world_xyz"]:
            x, y, z = h["world_xyz"][0], h["world_xyz"][1], h["world_xyz"][2]
            lat, lon = _world_xyz_to_latlon(x, y, z)
            lats.append(lat)
            lons.append(lon)
    if not lats:
        print("No valid hazard coordinates found.")
        return
    avg_lat = sum(lats) / len(lats)
    avg_lon = sum(lons) / len(lons)

    hazard_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=16, tiles="CartoDB dark_matter")

    color_map = {
        "fire": "red",
        "Fire": "red",
        "person": "blue",
        "vehicle": "orange",
        "building": "darkred",
        "Collapsed Building": "darkred",
        "Flood": "blue",
        "Traffic Incident": "orange",
    }

    print(f"Plotting {len(hazards)} hazards...")

    for hazard in hazards:
        if "latitude" in hazard and "longitude" in hazard:
            lat, lon = hazard["latitude"], hazard["longitude"]
        elif "world_xyz" in hazard and hazard["world_xyz"]:
            x, y, z = hazard["world_xyz"][0], hazard["world_xyz"][1], hazard["world_xyz"][2]
            lat, lon = _world_xyz_to_latlon(x, y, z)
        else:
            continue

        h_class = hazard.get("class_name", "unknown")
        h_id = hazard.get("hazard_id", 0)
        img_path = hazard.get("image_path")
        vlm_text = hazard.get("vlm_analysis", "")
        frame = hazard.get("frame_idx", 0)

        marker_color = color_map.get(h_class, color_map.get(h_class.lower(), "red"))

        b64_img = get_base64_image(img_path) if img_path else None
        img_html = ""
        if b64_img:
            img_html = f'<img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;" />'

        coord_str = f"World: [{hazard['world_xyz'][0]:.1f}, {hazard['world_xyz'][1]:.1f}, {hazard['world_xyz'][2]:.1f}]" if hazard.get("world_xyz") else f"GPS: {lat:.5f}, {lon:.5f}"

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 300px;">
            <h3 style="margin-top: 0; color: #d32f2f;">{h_class} (ID: {h_id})</h3>
            {img_html}
            <p><strong>Position:</strong> {coord_str}</p>
            <p><strong>Frame:</strong> {frame}</p>
            <hr>
            <p><strong>VLM Analysis:</strong><br><span style="color: #555;">{vlm_text}</span></p>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{h_class} Detected",
            icon=folium.Icon(color=marker_color, icon="info-sign"),
        ).add_to(hazard_map)

    # Save mapping output
    output_html = os.path.join(os.path.dirname(report_path), "Hazard_Map.html")
    hazard_map.save(output_html)
    
    print(f"\n=========================================")
    print(f"SUCCESS! Interactive Map saved to: {output_html}")
    print(f"=========================================\n")
    
    # Open the HTML file in the default browser automatically
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(output_html))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResQ-AI Map Generator")
    parser.add_argument("--report", type=str, required=True, help="Path to Flight_Report.json")
    args = parser.parse_args()
    build_map(args.report)
