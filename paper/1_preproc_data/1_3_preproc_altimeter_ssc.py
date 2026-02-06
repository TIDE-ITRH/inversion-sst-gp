import xarray as xr
import os

# Configuration
LON_LIMITS = (115, 118)
LAT_LIMITS = (-15.5, -12.5)
ALTIMETRY_DATA_DIR = "1_preproc_data/non_proc_data/altimetry"
PROCESSED_DIR = "1_preproc_data/proc_data"
OUTPUT_FILENAME = "altimeter_currents.nc"


# Main processing function
def process_altimetry_data(altimetry_dir, lon_limits, lat_limits):
    """
    Loads raw altimetry data, selects a geographical region,
    and returns a processed xarray Dataset.
    """
    # Get all NetCDF files in the directory
    geo_files = [f for f in os.listdir(altimetry_dir) if f.endswith(".nc")]
    geo_paths = [os.path.join(altimetry_dir, f) for f in geo_files]

    # Open multiple NetCDF files as a single dataset
    dsgeo_o = xr.open_mfdataset(geo_paths)

    # Select data within the specified geographical limits
    ds_geo_c = dsgeo_o.sel(
        longitude=slice(lon_limits[0], lon_limits[1]),
        latitude=slice(lat_limits[0], lat_limits[1]),
    )

    # Create a new Dataset with standardized coordinate names
    ds_geo = xr.Dataset(
        {
            "ugos": (["time", "lat", "lon"], ds_geo_c.ugos.values),
            "vgos": (["time", "lat", "lon"], ds_geo_c.vgos.values),
        },
        coords={
            "time": ds_geo_c.time.values,
            "lat": ds_geo_c.latitude.values,
            "lon": ds_geo_c.longitude.values,
        },
    )
    return ds_geo


# Main execution
if __name__ == "__main__":
    print("--- Starting preprocessing altimetry data ---")

    # Ensure the processed data directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading and processing altimetry data")
    # Process the altimetry data
    processed_altimetry_ds = process_altimetry_data(
        ALTIMETRY_DATA_DIR, LON_LIMITS, LAT_LIMITS
    )
    
    # Create process directory if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Save the processed dataset
    output_file_path = os.path.join(PROCESSED_DIR, OUTPUT_FILENAME)
    print(f"Saving processed altimetry data to {output_file_path}")
    processed_altimetry_ds.to_netcdf(output_file_path)

    print("Data processing complete")