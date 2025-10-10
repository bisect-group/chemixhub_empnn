### Running the CheMixHub benchmark 
This folder contains the scripts to run the mixture model benchmark.

For details any of the CLI arguments, run `python name_of_script.py --help`

#### Creating dataset splits

If you have implemented a new dataset and want create splits for it you can use:

- `make_splits.py`
  Create splits for a CheMixHub Dataset.

  Example of CLI call: `python make_splits.py --dataset_name dataset_name --split_type kfold`

#### Running experiments

- `run_cmp.py`
  Run number of component-based split experiment for a model based on the specified YAML config file. For a detailed explanation of the YAML config file, please refer to the `README.md` file in the `../config` folder.
  
  Example of CLI call: `python run_cmp.py --config ../config/example.yaml --k_values 5 10 20`

- `run_cv.py`
  Run cross validation experiment for a model based on the specified YAML config file. For a detailed explanation of the YAML config file, please refer to the `README.md` file in the `../config` folder.
  
  Example of CLI call: `python run_cv.py --config ../config/example.yaml`

- `run_lmo.py`
  Run molecule exclusion-based split experiment for a model based on the specified YAML config file. For a detailed explanation of the YAML config file, please refer to the `README.md` file in the `../config` folder.

  Example of CLI call: `python run_cmp.py --config ../config/example.yaml`

- `run_model.py`
   Runs model training and evaluation on a random split based on the specified YAML config file. For a detailed explanation of the YAML config file, please refer to the `README.md` file in the `../config` folder.
   
   Example of CLI call: `python run_model.py --config ../config/example.yaml --experiment_name test_run`

- `run_temp.py`
  Run temperature range-based split experiment for a model based on the specified YAML config file. For a detailed explanation of the YAML config file, please refer to the `README.md` file in the `../config` folder.

  Example of CLI call: `python run_temp.py --config ../config/example.yaml --physics_based True`

#### Running dataset statistics scripts

- `run_group_features.py`

- `run_rot_bond.py`