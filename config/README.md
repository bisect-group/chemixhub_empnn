# Model Configuration File

We detail the structure and meaning of parameters in the `config.yaml` file used to train and benchmark deep learning-based mixture property prediction models in CheMixHub. 

## Mixture model parameters

The `mixture_model` defines how molecular and mixture-level information are encoded, aggregated, and used for prediction. The example values matches the one found in `config/example.yaml`. Values starting with `$` should not be overwritten.

### General model parameters

|Parameter|Description|Example value|
|-|-|-|
|`dropout_rate`|Dropout rate applied throughout the model|`0.0`|
|`attn_num_heads`|Number of attention heads for self-attention layers|`8`|
|`mol_aggregation`|How molecular embeddings are aggregated within a mixture. Options: `["mean", "max", "pna", "scaled_pna", "set2set"]`|`"mean"`|

### Molecule Encoder (`mol_encoder`)

This part of the model encodes each molecule in the mixture.

|Parameter|Description|Example value|
|-|-|-|
|`type`|Type of molecular encoder. Options: `["linear", "gnn"]`|`"linear"`|
|`output_dim`| Output dimension of molecular embeddings.|`128`|

If using a GNN-based encoder, further specify:

|Parameter|Description|Example value|
|-|-|-|
|`global_dim`|Matches the molecular encoder output dimension, leave as `${mixture_model.mol_encoder.output_dim}`||
|`hidden_dim`|Hidden layer size in the GNN|50|
|`depth`|Number of message-passing layers|3|

### Fraction Aggregation (`fraction_aggregation`)

Handles the composition-dependent weighting of molecules in a mixture.

|Parameter|Description|Example value|
|-|-|-|
|`type`|Aggregation strategy. Options: `["concat", "multiply", "film"]`|`"concat"`|

If using FiLM layer as aggregation, further specify 

|Parameter|Description|Example value|
|-|-|-|
|`activation`|Activation function options: `["relu", "sigmoid"]`|`"sigmoid"`|
|`output_dim`|Matches the molecular encoder output dimension, leave as `${mixture_model.mol_encoder.output_dim}`||

### Context Aggregation (`context_aggregation`)

Integrates contextual variables (e.g., temperature, pressure).

|Parameter|Description|Example value|
|-|-|-|
|`type`|Aggregation type. Options: `["concat", "film"]`|`"concat"`|

If using FiLM layer as aggregation, further specify:

|Parameter|Description|Example value|
|-|-|-|
|`activation`|Activation function options: `["relu", "sigmoid"]`|`"sigmoid"`|
|`output_dim`|Matches the molecular encoder output dimension, leave as `${mixture_model.mol_encoder.output_dim}`||

### Mixture Encoder (`mix_encoder`)

Encodes the molecular mixture from a set of molecular embeddings.

|Parameter|Description|Example value|
|-|-|-|
| `type`       | Encoder type. Options: `["self_attn", "deepset"]` | `"deepset"` |
| `embed_dim`  | Embedding dimension for mixture representation.   | `128`       |
| `num_layers` | Number of layers in the encoder.                  | `2`         |

If using self-attention, further specify:

|Parameter|Description|Example value|
|-|-|-|
|`add_mlp`|If the mixture embedding is run through a MLP|`False`|

### Aggregation Blocks parameters (based on value of `mol_aggregation`)

If you selected `"set2set"` in `set2set_aggregation`, further specify:

|Parameter|Description|Example value|
|-|-|-|
|`processing_steps`|Number of processing steps|`3`|

### Regressor Head (`regressor`)

Predicts the target property from the encoded mixture representation.

|Parameter|Description|Example|
|-|-|-|
| `type`       | Type of prediction head. Options: `["mlp", "physics_based"]` | `"mlp"` |
| `hidden_dim` | Hidden layer size.                                           | `100`   |
| `num_layers` | Number of layers.                                            | `2`     |

If you selected `"mlp"`, further specify:

|Parameter|Description|Example|
|-|-|-|
|`output_dim`|Size of the output dimension|`1`|

If you selected `"physics-based"`, further specify:

|Parameter|Description|Example|
|-|-|-|
|`law`|law used to make prediction, only Arrhenius is available at the moment.|`arrhenius`|

## Dataset Configuration (`dataset`)

The `dataset` defines on which dataset and task the model is trained on, and using which molecular featurization. Note that if using `"custom_molecular_graphs"` then a GNN-based encoder must be specified under `mol_encoder` in `mixture_model`

|Parameter|Description|Example|
|-|-|-|
|`name`|Dataset name|`"miscible-solvent"`|
|`property`|Target property to predict (task)|`"Density"`|
|`featurization`| Type of molecular features. Options: `["custom_molecular_graphs", "molt5_embeddings", "rdkit2d_normalized_features"]`|`"rdkit2d_normalized_features"`|

## Optimization and Training

### Scheduler & Optimizer

|Parameter|Description|Example value|
|-|-|-|
| `loss_type`      | Loss function.                       | `"mse"`  |
| `optimizer_type` | Optimizer.                           | `"adam"` |
| `lr_mol_encoder` | Learning rate for molecular encoder. | `1e-4`   |
| `lr_other`       | Learning rate for other parameters.  | `5e-4`   |
| `weight_decay`   | L2 regularization coefficient.       | `0`      |

### Trainer Settings

|Parameter|Description|Example value|
|-|-|-|
| `seed`           | Random seed for reproducibility.           | `42`     |
| `root_dir`       | Path to output directory (to fill).        | `""`     |
| `num_workers`    | Number of data loader workers.             | `8`      |
| `max_epochs`     | Maximum training epochs.                   | `100`    |
| `batch_size`     | Batch size.                                | `1024`   |
| `device`         | Training device.                           | `"cuda"` |
| `early_stopping` | Whether to stop early based on validation. | `True`   |
| `patience`       | Number of epochs to wait before stopping.  | `100`    |

## Example Usage

To train the model:

```bash
python train.py --config config.yaml
```

To override parameters directly:

```bash
python train.py --config config.yaml mixture_model.mol_encoder.type=gnn dataset.property="Viscosity"
```