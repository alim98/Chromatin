import os
import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import trimesh
import base64
from io import BytesIO
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def capture_mesh_thumbnail(mesh_path, size=(200, 200)):
    """
    Generate a thumbnail image of a mesh
    
    Args:
        mesh_path (str): Path to the mesh file
        size (tuple): Size of the thumbnail (width, height)
        
    Returns:
        str: Base64-encoded PNG image data
    """
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        
        # Create a scene with the mesh
        scene = trimesh.Scene(mesh)
        
        # Set camera to a good default position
        scene.camera_transform = trimesh.transformations.rotation_matrix(
            angle=np.radians(30), 
            direction=[1, 0, 0]
        )
        
        # Render the mesh
        img = scene.save_image(resolution=size)
        
        # Convert to base64
        buffered = BytesIO()
        Image.open(BytesIO(img)).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating thumbnail for {mesh_path}: {e}")
        return ""

def generate_mesh_report(mesh_analysis_csv, mesh_validation_csv, mesh_dir, output_html):
    """
    Generate an HTML report of mesh analysis
    
    Args:
        mesh_analysis_csv (str): Path to mesh analysis CSV file
        mesh_validation_csv (str): Path to mesh validation CSV file
        mesh_dir (str): Directory containing mesh files
        output_html (str): Path to save the HTML report
    """
    # Load the data
    try:
        analysis_df = pd.read_csv(mesh_analysis_csv)
        print(f"Loaded analysis data: {len(analysis_df)} rows")
    except Exception as e:
        print(f"Error loading analysis CSV: {e}")
        analysis_df = pd.DataFrame()
    
    try:
        validation_df = pd.read_csv(mesh_validation_csv)
        print(f"Loaded validation data: {len(validation_df)} rows")
    except Exception as e:
        print(f"Error loading validation CSV: {e}")
        validation_df = pd.DataFrame()
    
    # Merge the dataframes if both exist
    if not analysis_df.empty and not validation_df.empty:
        # Extract sample ID from filename
        validation_df['sample_id'] = validation_df['filename'].apply(
            lambda x: x.split('_')[0] if '_' in x else x.split('.')[0]
        )
        
        # Merge on sample_id
        merged_df = pd.merge(analysis_df, validation_df, on='sample_id', how='outer')
        print(f"Merged data: {len(merged_df)} rows")
    elif not analysis_df.empty:
        merged_df = analysis_df
    elif not validation_df.empty:
        merged_df = validation_df
        # Extract sample ID from filename
        merged_df['sample_id'] = merged_df['filename'].apply(
            lambda x: x.split('_')[0] if '_' in x else x.split('.')[0]
        )
    else:
        print("No data available for report")
        return
    
    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chromatin Mesh Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #333;
            }
            .stats-box {
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .plot-container {
                margin-bottom: 30px;
            }
            .mesh-thumbnail {
                display: inline-block;
                margin: 5px;
                text-align: center;
            }
            .mesh-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 10px;
            }
            .mesh-card {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: white;
            }
            .mesh-card img {
                max-width: 100%;
                height: auto;
                border: 1px solid #eee;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chromatin Mesh Analysis Report</h1>
            <p>Generated on: """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    # Summary statistics
    html += """
            <h2>Summary Statistics</h2>
            <div class="stats-box">
    """
    
    if 'num_vertices' in merged_df.columns:
        html += f"""
                <p><strong>Total Meshes:</strong> {len(merged_df)}</p>
                <p><strong>Average Vertices:</strong> {merged_df['num_vertices'].mean():.1f}</p>
                <p><strong>Average Faces:</strong> {merged_df['num_faces'].mean():.1f}</p>
        """
    
    if 'is_watertight' in merged_df.columns:
        watertight_count = merged_df['is_watertight'].sum()
        html += f"""
                <p><strong>Watertight Meshes:</strong> {watertight_count} ({watertight_count/len(merged_df)*100:.1f}%)</p>
        """
    
    if 'loadable' in merged_df.columns:
        loadable_count = merged_df['loadable'].sum()
        html += f"""
                <p><strong>Loadable Meshes:</strong> {loadable_count} ({loadable_count/len(merged_df)*100:.1f}%)</p>
        """
    
    html += """
            </div>
    """
    
    # Create histograms
    hist_columns = ['num_vertices', 'num_faces', 'volume', 'surface_area', 
                   'compactness', 'sphericity', 'elongation', 'flatness']
    
    html += """
            <h2>Distribution of Mesh Properties</h2>
            <div class="plot-container">
    """
    
    for col in hist_columns:
        if col in merged_df.columns:
            # Create histogram
            fig = px.histogram(
                merged_df, 
                x=col, 
                title=f'Distribution of {col}',
                nbins=30,
                opacity=0.7
            )
            fig.update_layout(
                margin=dict(l=50, r=50, t=50, b=50),
                height=400
            )
            html += f"""
                <div id="{col}_hist" style="width:100%;height:400px;"></div>
                <script>
                    var fig = {fig.to_json()};
                    Plotly.newPlot('{col}_hist', fig.data, fig.layout);
                </script>
            """
    
    html += """
            </div>
    """
    
    # Create scatter plots
    scatter_pairs = [
        ('volume', 'surface_area'),
        ('compactness', 'sphericity'),
        ('elongation', 'flatness'),
        ('num_vertices', 'num_faces')
    ]
    
    html += """
            <h2>Relationship Between Mesh Properties</h2>
            <div class="plot-container">
    """
    
    for x_col, y_col in scatter_pairs:
        if x_col in merged_df.columns and y_col in merged_df.columns:
            # Create scatter plot
            fig = px.scatter(
                merged_df, 
                x=x_col, 
                y=y_col, 
                title=f'{y_col} vs {x_col}',
                hover_data=['sample_id'],
                opacity=0.7
            )
            fig.update_layout(
                margin=dict(l=50, r=50, t=50, b=50),
                height=400
            )
            html += f"""
                <div id="{y_col}_vs_{x_col}" style="width:100%;height:400px;"></div>
                <script>
                    var fig = {fig.to_json()};
                    Plotly.newPlot('{y_col}_vs_{x_col}', fig.data, fig.layout);
                </script>
            """
    
    html += """
            </div>
    """
    
    # Show mesh thumbnails with properties
    html += """
            <h2>Mesh Gallery</h2>
            <p>Sample of meshes with their properties:</p>
            <div class="mesh-grid">
    """
    
    # Get a sample of mesh files
    mesh_files = glob.glob(os.path.join(mesh_dir, "*.ply"))
    if len(mesh_files) > 20:
        # Take a random sample
        np.random.seed(42)
        mesh_files = np.random.choice(mesh_files, 20, replace=False)
    
    for mesh_file in mesh_files:
        sample_id = os.path.basename(mesh_file).split('_')[0]
        
        # Get mesh properties from dataframe
        mesh_data = merged_df[merged_df['sample_id'] == sample_id].iloc[0] if not merged_df[merged_df['sample_id'] == sample_id].empty else None
        
        # Generate thumbnail
        thumb = capture_mesh_thumbnail(mesh_file)
        
        html += f"""
            <div class="mesh-card">
                <h3>Sample {sample_id}</h3>
                <img src="{thumb}" alt="Mesh {sample_id}">
        """
        
        if mesh_data is not None:
            html += "<table>"
            if 'num_vertices' in mesh_data:
                html += f"<tr><td>Vertices</td><td>{mesh_data['num_vertices']}</td></tr>"
            if 'num_faces' in mesh_data:
                html += f"<tr><td>Faces</td><td>{mesh_data['num_faces']}</td></tr>"
            if 'volume' in mesh_data:
                html += f"<tr><td>Volume</td><td>{mesh_data['volume']:.2f}</td></tr>"
            if 'surface_area' in mesh_data:
                html += f"<tr><td>Surface Area</td><td>{mesh_data['surface_area']:.2f}</td></tr>"
            if 'sphericity' in mesh_data:
                html += f"<tr><td>Sphericity</td><td>{mesh_data['sphericity']:.3f}</td></tr>"
            html += "</table>"
        
        html += """
            </div>
        """
    
    html += """
            </div>
            
            <h2>Data Table</h2>
            <div style="max-height:500px;overflow:auto;">
                <table>
                    <thead>
                        <tr>
    """
    
    # Add table headers
    display_columns = ['sample_id', 'num_vertices', 'num_faces', 'volume', 'surface_area', 
                      'sphericity', 'elongation', 'is_watertight']
    
    for col in display_columns:
        if col in merged_df.columns:
            html += f"<th>{col}</th>"
    
    html += """
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add table rows
    for _, row in merged_df.iterrows():
        html += "<tr>"
        for col in display_columns:
            if col in merged_df.columns:
                value = row[col]
                if isinstance(value, (float, np.float64)):
                    formatted = f"{value:.3f}"
                elif isinstance(value, bool):
                    formatted = "Yes" if value else "No"
                else:
                    formatted = str(value)
                html += f"<td>{formatted}</td>"
        html += "</tr>"
    
    html += """
                    </tbody>
                </table>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(output_html, 'w') as f:
        f.write(html)
    
    print(f"Report saved to {output_html}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report of mesh analysis")
    parser.add_argument("--analysis_csv", type=str, 
                       default=os.path.join(config.ANALYSIS_OUTPUT_DIR, "mesh_analysis.csv"),
                       help="Path to mesh analysis CSV file")
    parser.add_argument("--validation_csv", type=str, 
                       default=os.path.join(config.ANALYSIS_OUTPUT_DIR, "mesh_validation_meshes.csv"),
                       help="Path to mesh validation CSV file")
    parser.add_argument("--mesh_dir", type=str, 
                       default=os.path.join(config.RESULTS_DIR, "meshes", "meshes"),
                       help="Directory containing mesh files")
    parser.add_argument("--output_html", type=str, 
                       default=os.path.join(config.RESULTS_DIR, "html", "mesh_report.html"),
                       help="Path to save the HTML report")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    
    # Generate the report
    generate_mesh_report(
        args.analysis_csv,
        args.validation_csv,
        args.mesh_dir,
        args.output_html
    )

if __name__ == "__main__":
    main() 