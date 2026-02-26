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

def build_map(report_path):
    print(f"Reading Flight Report from: {report_path}")
    
    if not os.path.exists(report_path):
        print(f"Error: Flight report not found at {report_path}")
        return

    with open(report_path, 'r') as f:
        flight_report = json.load(f)

    if not flight_report:
        print("Flight report is empty. No hazards were detected.")
        return

    # Calculate map center (average of all hazard coordinates)
    avg_lat = sum(h["latitude"] for h in flight_report) / len(flight_report)
    avg_lon = sum(h["longitude"] for h in flight_report) / len(flight_report)

    # Initialize Folium Map (satellite/street view)
    hazard_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=18, tiles="CartoDB dark_matter")
    
    # Define color scheme based on severity/class
    color_map = {
        "Fire": "red",
        "Collapsed Building": "darkred",
        "Flood": "blue",
        "Traffic Incident": "orange"
    }

    print(f"Plotting {len(flight_report)} hazards...")

    for hazard in flight_report:
        lat = hazard["latitude"]
        lon = hazard["longitude"]
        h_class = hazard["class_name"]
        h_id = hazard["hazard_id"]
        img_path = hazard["image_path"]
        vlm_text = hazard["vlm_analysis"]
        frame = hazard["frame_idx"]
        
        # Default to red if class not in color map
        marker_color = color_map.get(h_class, "red")

        # Encode image for the popup
        b64_img = get_base64_image(img_path)
        img_html = ""
        if b64_img:
            img_html = f'<img src="data:image/jpeg;base64,{b64_img}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;" />'

        # Build Interactive HTML Popup Content
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 300px;">
            <h3 style="margin-top: 0; color: #d32f2f;">{h_class} (ID: {h_id})</h3>
            {img_html}
            <p><strong>GPS:</strong> {lat:.5f}, {lon:.5f}</p>
            <p><strong>Frame:</strong> {frame}</p>
            <hr>
            <p><strong>VLM Analysis:</strong><br><span style="color: #555;">{vlm_text}</span></p>
        </div>
        """
        
        # Add Marker to Map
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{h_class} Detected",
            icon=folium.Icon(color=marker_color, icon="info-sign")
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
