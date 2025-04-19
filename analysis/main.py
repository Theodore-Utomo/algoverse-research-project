import rasterio
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from rasterstats import zonal_stats
from sklearn.cluster import DBSCAN
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def graph_population_density():
    with rasterio.open("/Users/theodoreutomo/algoverse-research-project/data/raw/population/KEN_population_v1_0_gridded.tif") as src:
        print(src.profile)     # dtype, crs, transform, width/height, count (bands)
        print("Bounds:", src.bounds)
        print("NoData:", src.nodatavals)
        band1 = src.read(1)

    plt.imshow(band1, cmap="terrain")   
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM")
    plt.show()
    plt.savefig("population_density.png")


def graph_healthcare_facilities_locations(gdf):
    gdf.plot(figsize=(10, 10), marker='o', color='red', markersize=5)
    plt.title("Healthcare Facilities in Kenya")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()
    plt.savefig("healthcare_facilities_location.png")


def info_on_healthcare_amenity_types(gdf):
    unique_amenities = gdf['amenity'].dropna().unique()
    print("Unique amenities in the dataset:")
    print(unique_amenities)

    amenity_counts = gdf['amenity'].value_counts(dropna=True)
    print("\nCounts of each amenity type:")
    print(amenity_counts)
    
    amenity_counts.plot(kind='bar', figsize=(10,6), title='Distribution of Amenity Types')
    plt.xlabel("Amenity Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    plt.savefig("amenity_distribution.png")

def print_cities_info(gdf):
    rows_with_addr_city = gdf[gdf['addr_city'].notnull()]
    print("\nSample rows with non-null 'addr_city' values:")
    print(rows_with_addr_city[['name', 'addr_city', 'amenity']].head(10))

    print("\n")
    unique_cities = rows_with_addr_city['addr_city'].dropna().unique()
    print("Cities in the dataset:")
    for i in range(len(unique_cities)):
        print(f"{i+1}. {unique_cities[i]}")
        
def visualize_cluster_info(gdf):
    gdf_projected = gdf.to_crs(epsg=3857)
    # Extract x and y coordinates from the geometries into a NumPy array.
    coords = np.array(list(zip(gdf_projected.geometry.x, gdf_projected.geometry.y)))
    # These parameters may require tuning based on the spatial density of healthcare facilities.
    dbscan = DBSCAN(eps=5000, min_samples=5)  # eps=500 meters and min_samples=5 is a starting point
    dbscan.fit(coords)
    gdf_projected['cluster'] = dbscan.labels_
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the facilities colored by their cluster assignment.
    # The categorical=True and legend parameters help to differentiate cluster colors.
    gdf_projected.plot(column='cluster', categorical=True, legend=True,
                    markersize=10, cmap='tab20', ax=ax)
    ax.set_title("Healthcare Facilities Clustering (DBSCAN: eps=500m, min_samples=5)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    plt.show()
    plt.savefig("healthcare_facilities_clustered.png")
    
    cluster_counts = gdf_projected['cluster'].value_counts().sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cluster_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel("Cluster Label (-1 indicates noise)")
    ax.set_ylabel("Number of Facilities")
    ax.set_title("Number of Facilities per Cluster")
    plt.tight_layout()
    plt.show()
    plt.savefig("cluster_distribution.png")

def miscellaneous_inspection(gdf):
    missing_counts = gdf.isnull().sum()
    print("Missing values in each column:")
    print(missing_counts)

    duplicates = gdf.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    amenity_summary = gdf['amenity'].value_counts(dropna=True)
    print("Amenity counts:")
    print(amenity_summary)

    city_summary = gdf['addr_city'].value_counts(dropna=True)
    print("City counts:")
    print(city_summary)

    dental_facilities = gdf[gdf['amenity'].str.contains('dentist', case=False, na=False)]
    print("Number of dental facilities:", len(dental_facilities))

    fig, ax = plt.subplots(figsize=(10, 10))
    dental_facilities.plot(ax=ax, marker='o', color='magenta', markersize=30)
    ax.set_title("Dental Facilities in Kenya")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
    plt.savefig("dental_facilities_points.png")

def graph_healthcare_facilities():
    gdf = gpd.read_file('/Users/theodoreutomo/algoverse-research-project/data/raw/healthcarefac/hotosm_ken_health_facilities_points_shp/hotosm_ken_health_facilities_points_shp.shp')
    graph_healthcare_facilities_locations(gdf)
    print_cities_info(gdf)
    info_on_healthcare_amenity_types(gdf)
    visualize_cluster_info(gdf)
    miscellaneous_inspection(gdf)



def graph_healthcare_facilities_with_population_and_roads():
    facilities_gdf = gpd.read_file("/Users/theodoreutomo/algoverse-research-project/data/raw/healthcarefac/hotosm_ken_health_facilities_points_shp/hotosm_ken_health_facilities_points_shp.shp")
    raster_path = "/Users/theodoreutomo/algoverse-research-project/data/raw/population/KEN_population_v1_0_gridded.tif"
    with rasterio.open(raster_path) as src:
        pop_density = src.read(1)
        raster_crs = src.crs
        raster_bounds = src.bounds

    # Load the roads shapefile
    roads_gdf = gpd.read_file("/Users/theodoreutomo/algoverse-research-project/data/raw/roads/KEN_Roads/KEN_Roads.shp")
    
    if roads_gdf.crs is None:
    # Replace "EPSG:4326" with the actual CRS if different
        roads_gdf = roads_gdf.set_crs("EPSG:4326")

    facilities_gdf = facilities_gdf.to_crs(raster_crs)
    roads_gdf = roads_gdf.to_crs(raster_crs)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the population density raster with proper extent.
    extent = [raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top]
    img = ax.imshow(pop_density, extent=extent, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Population Density")

    # Overlay roads on the map.
    roads_gdf.plot(ax=ax, color='black', linewidth=1.5, label='Roads')

    # Overlay healthcare facilities on the map.
    facilities_gdf.plot(ax=ax, marker='o', color='red', markersize=30, label='Healthcare Facilities')

    # Add title and labels.
    ax.set_title("Population Density, Healthcare Facilities, and Roads in Kenya")
    ax.set_xlabel("Easting / Longitude")
    ax.set_ylabel("Northing / Latitude")
    ax.legend(loc='upper right')

    plt.show()
    plt.savefig("population_healthcare_roads_layered.png")
    
    facilities_gdf = facilities_gdf.to_crs(raster_crs)
    roads_gdf = roads_gdf.to_crs(raster_crs)

    # -------------------------------
    # Buffer Roads to Determine Accessibility
    # -------------------------------
    # Define the buffer distance in meters
    buffer_distance = 10000

    # Reproject roads to a projected CRS for correct buffering (EPSG:3857 uses meters)
    roads_projected = roads_gdf.to_crs(epsg=3857)
    roads_buffer = roads_projected.buffer(buffer_distance)
    # Combine the buffers into one geometry using union_all (replaces deprecated unary_union)
    roads_buffer_union = roads_buffer.union_all()
    # Reproject the unioned buffer back to the raster's CRS
    roads_buffer_union = gpd.GeoSeries(roads_buffer_union).set_crs(epsg=3857).to_crs(raster_crs)

    # -------------------------------
    # Identify Accessible Facilities
    # -------------------------------
    # Facilities that intersect with the buffered roads are considered accessible
    accessible_facilities = facilities_gdf[facilities_gdf.intersects(roads_buffer_union)]
    num_accessible = len(accessible_facilities)
    print("Number of accessible healthcare facilities:", num_accessible)

    # -------------------------------
    # Plotting the Data
    # -------------------------------
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the population density raster with correct extent
    extent = [raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top]
    img = ax.imshow(pop_density, extent=extent, cmap='viridis', alpha=0.6)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Population Density")

    # Plot roads as black lines
    roads_gdf.plot(ax=ax, color='black', linewidth=1.5)

    # Plot the buffered roads in blue with transparency
    gpd.GeoSeries(roads_buffer_union).plot(ax=ax, color='blue', alpha=0.3)

    # Plot all healthcare facilities as red dots
    facilities_gdf.plot(ax=ax, marker='o', color='red', markersize=30)

    # Create custom legend handles for clarity
    road_handle = Line2D([0], [0], color='black', lw=1.5, label='Roads')
    buffer_handle = Patch(facecolor='blue', edgecolor='blue', alpha=0.3, label='Buffered Roads')
    facility_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                            markersize=10, label='Healthcare Facilities')

    # Add the custom legend to the plot
    ax.legend(handles=[road_handle, buffer_handle, facility_handle], loc='upper right')

    ax.set_title("Population Density, Healthcare Facilities, and Roads in Kenya")
    ax.set_xlabel("Easting / Longitude")
    ax.set_ylabel("Northing / Latitude")

    plt.show()
    plt.savefig("population_healthcare_roads_layered_with_buffer")
    
    
def main():
    # 1) Population only
    graph_population_density()
    
    # 2) Healthcare facilities only (locations, amenity types, clusters, etc.)
    graph_healthcare_facilities()
    
    # 3) Combined population + facilities + roads, including accessibility buffer
    graph_healthcare_facilities_with_population_and_roads()


if __name__ == "__main__":
    main()
