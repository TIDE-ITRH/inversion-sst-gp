### How to obtain the processed data in the `./proc_data` folder

---
#### SUNTANS surface currents and surface temperature
1. Visit [https://doi.org/10.26182/7vmz-yz14](https://doi.org/10.26182/7vmz-yz14)
2. Download the NetCDF file `suntans_surface_prop_north_australian_basin.nc`
3. Place the dowloaded NetCDF file in `./non_proc_data/suntans`
6. Process the file by running `./1_1_preproc_suntans_data.py` this creates `./proc_data/suntans_1h.nc`, `./proc_data/suntans_24h.nc`, `./proc_data/suntans_dense_cloud.nc`, `./proc_data/suntans_measurement_error.nc`, and `./proc_data/suntans_sparse_cloud.nc`

---
#### Himawari-9 surface temperature
1. Visit [https://www.earthdata.nasa.gov/data/catalog](https://www.earthdata.nasa.gov/data/catalog) and search `H09-AHI-L3C-ACSPO-v2.90` and click on `GHRSST L3C NOAA/ACSPO Himawari-09 AHI Pacific Ocean Region Sea Surface Temperature v2.90 dataset`
2. Click on `Data Access`
3. Click on `Search Granules`
4. Download the following files, covering scenario of case 1 (2023-09-22 04:00:00) and case 2 (2023-12-18 01:00:00):
    - 2023-09-22 03:00:00
    - 2023-09-22 04:00:00
    - 2023-09-22 05:00:00
    - 2023-12-18 00:00:00
    - 2023-12-18 02:00:00
    - 2023-12-18 03:00:00
5. Place the downloaded NetCDF files in `./non_proc_data/himawari`
6. Process the files by running `./1_2_preproc_himawari_9_sst.py` this creates `./proc_data/himawari.nc`



---
#### Altimetry derived currents
1. Visit [https://data.marine.copernicus.eu/products](https://data.marine.copernicus.eu/products) and search for `SEALEVEL_GLO_PHY_L4_MY_008_047` and click on the `Global Ocean Gridded L 4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing` product
2. Click on `Data access`
3. Click on `Browse` of the Daily dataset
4. Download the following files, covering scenario of satellite application case 1 (2023-09-22 04:00:00) and case 2 (2023-12-18 01:00:00):
    - 2023-09-22 00:00:00
    - 2023-12-18 00:00:00
5. Place the downloaded NetCDF files in `./non_proc_data/altimetry`
6. Process the files by running `./1_3_preproc_altimeter_ssc.py` this creates `./proc_data/altimeter_currents.nc`