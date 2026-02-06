import json
import os

import pandas as pd

# Helper function
def load_json(path):
    """Load JSON file and return as Python dict."""
    with open(path, 'r') as f:
        return json.load(f)

def main():
    print('--- Merging outputs from covariance parameter estimation ---')

    # Folder containing JSON files
    main_folder = '2_covariance_parameter_estimation/intermediate'

    # Output directory for CSV files
    output_dir = '2_covariance_parameter_estimation/outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Mapping of data types to expected labels
    type_label_dict = {
        'time_24h': ["optimum", "gprm", "gprm_e", "gos"],
        'time_1h': ["optimum", "gprm", "gprm_e", "gos"],
        'noise': ["gprm", "gprm_e", "gos"],
        'cloud_dense': ["gprm", "gprm_e"],
        'cloud_sparse': ["gprm", "gprm_e"],
        'satellite': ["gprm_e"]
    }

    # Optional label renaming
    rename_dicts = {
        'time_1h': {
            'optimum': 'gp_num_t',
            'gprm': 'gp_num_t1',
            'gprm_e': 'gp_obs_t',
            'gos': 'gos_t',
        },
        'time_24h': {
            'optimum': 'gp_num_t',
            'gprm': 'gp_num_t1',
            'gprm_e': 'gp_obs_t',
            'gos': 'gos_t',
        },
        'default': {
            'gprm': 'gp_num_t',
            'gprm_e': 'gp_obs_t',
            'gos': 'gos_t',
            'GPRM_e': 'gp_obs_t'
        }
    }

    # Mapping of type to column name for "step"
    step_key_map = {
        'noise': 'noise_sd',
        'cloud_dense': 'coverage_dense',
        'cloud_sparse': 'coverage_sparse',
        'time_1h': 'time_sec',
        'time_24h': 'time_sec',
        'satellite': 'time'
    }

    print("Starting to process files")
    for data_type, labels in type_label_dict.items():
        print(f"Processing {data_type}")

        for label in labels:
            # Find relevant files
            if data_type == 'satellite':
                file_names = [
                    f for f in os.listdir(main_folder)
                    if f.endswith('.json') and data_type in f
                ]
            else:
                file_names = [
                    f for f in os.listdir(main_folder)
                    if f.endswith('.json') and label in f
                    and not (label == 'gprm' and 'gprm_e' in f) and data_type in f
                ]
                

            if not file_names:
                print(f"  No files found for {label} in {data_type}.")
                continue
            else:
                print(f"  Found {len(file_names)} files for {label} in {data_type}.")
            
            step_column = step_key_map[data_type]
            records = []
            for file_name in file_names:
                data = load_json(os.path.join(main_folder, file_name))

                # Flatten if needed
                if label in {'gprm', 'gprm_e', 'optimum', 'GPRM_e'}:
                    data.update(data.pop('est_params', {}))

                # Rename step key
                data[step_column] = data.pop("step")

                records.append(data)

            # Determine new label name
            rename_map = rename_dicts.get(data_type, rename_dicts['default'])
            label_output = rename_map.get(label, label)

            # Convert to DataFrame and export
            df = pd.DataFrame(records).set_index(step_column).sort_index()
            output_path = f'{output_dir}/{data_type}_{label_output}.csv'
            df.to_csv(output_path)
            
    print("Finished processing")
    
if __name__ == "__main__":
    main()