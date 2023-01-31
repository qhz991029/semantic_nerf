## Data Preparation
Use `generate_surface_normal.py` to generate surface normal, and groung truth is saved as `.pt` file (it will take about 50GB for Replica and derived surface normal)
## Run Semantic_NeRF
Hyper-parameters for Replica are set in `SSR_Replica_config.yaml`. Run command
```
python3 train_SSR_main.py --config_file ./SSR/configs/SSR_Replica_config.yaml
```
to start training and evaluation. 