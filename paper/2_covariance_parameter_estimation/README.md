### How to obtain the covariance parameter estimates in the `./outputs`

---

1. Run `./2_1_optimize_osse_per_process.py` to estimate the covariance parameters of each process for the fully observed noiseless experiment at 2014-02-19 18:00:00. This creates `./outputs/num_model_estimated_t.csv`.  
2. Run `./2_2_run_multi_optimize_osse.py` to estimate parameters for all the observing system simulation experiments. Note that this script takes a long computation time due to the high number of parameter optimizations. Therefore, paralaization is advised (e.g., using an HPC system by submitting jobs using SLURM to parallelize the process). All parameter estimates are stored as JSON files in `./intermediate`.  
2. Run `./2_3_run_multi_optimize_satellite.py` to estimate parameters for all the satellite SST instances. All parameter estimates are stored as JSON files in `./intermediate`.
4. The individual parameter estimates in `./intermediate` are combined per application per model using `./2_4_merge_outputs.py`, creating corresponding CSV files in `./outputs`.  
