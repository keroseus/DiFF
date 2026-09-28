# HumanMotionFlowEstimate

Point-cloud scene-flow estimation for human radar data. The implementation
combines a PointKAN feature encoder, a flow-matching velocity field, and an
optional Doppler motion prior.

## Contents

```text
config.yaml                 # MMBody configuration template
config_milliflow.yaml       # MilliFlow configuration template
train.py                    # MMBody/general training entry point
train_milliflow.py          # MilliFlow training entry point
evaluate.py                 # Checkpoint evaluation entry point
evaluate_milliflow.py       # MilliFlow 6/4-KNN evaluation entry point
datasets.py                 # Dataset loaders
flow_matching.py            # Flow matching and ODE integration
flow_models.py              # Velocity-field networks
pointkan_encoder.py         # PointKAN encoder
ckpt/                       # Optional reference checkpoints
models/                     # PointKAN/KAN modules
kat_rational/               # Rational activation implementation
pointnet2_ops_lib/          # PointNet++ extension source
rational_kat_cu/            # Optional rational CUDA extension
```

## Installation

Use a Conda environment with a CUDA-compatible PyTorch installation:

```bash
pip install -r requirements.txt
pip install ./pointnet2_ops_lib
```

The optional rational CUDA extension can be installed with:

```bash
pip install ./rational_kat_cu
```

## Dataset layout

The dataset is not included. Set the dataset path in the selected YAML
configuration file.

MMBody data should follow this layout:

```text
data/mmbody/
├── train/sequence_name/*.json
└── test/sequence_name/*.json
```

MilliFlow data should follow this layout:

```text
data/milliflow/data/
├── train/sequence_name/*.json
├── val/sequence_name/*.json
└── test/sequence_name/*.json
```

Each sample contains `pc1`, `pc2`, and `gt_flow`. MMBody samples store the
measured Doppler value in the fifth column of `pc1`.

## Training

Train on MMBody:

```bash
python train.py --config config.yaml
```

Train on MilliFlow:

```bash
python train_milliflow.py --config config_milliflow.yaml
```


## Evaluation

Evaluate an MMBody checkpoint with:

```bash
python evaluate.py --config config.yaml --ckpt /path/to/model.pth
```

Evaluate a MilliFlow 6/4-KNN checkpoint with:

```bash
python evaluate_milliflow.py --config config_milliflow.yaml --ckpt /path/to/model.pth
```

Reference checkpoints, when included, are stored under `ckpt/`.

## License

See [LICENSE](LICENSE).
