import os
import torch
import pandas as pd
import numpy as np
import wandb

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchtune.training.metric_logging import WandBLogger
from omegaconf import OmegaConf

from mixhub.data.dataset import MixtureTask
from mixhub.data.data import DATA_CATALOG
from mixhub.data.featurization import FEATURIZATION_TYPE
from mixhub.data.collate import custom_collate
from mixhub.data.splits import SplitLoader

from mixhub.model.train import train
from mixhub.model.predict import predict
from mixhub.model.model_builder import build_mixture_model


def main():
    # Initialize wandb
    wandb.init()
    config_dict = wandb.config

    # Load base config and update with sweep parameters
    base_config = OmegaConf.load("../config/example.yaml")

    # Update config with wandb sweep parameters
    for key, value in config_dict.items():
        OmegaConf.update(base_config, key, value, merge=True)

    config = base_config

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    print(f"Running on: {device}")

    root_dir = config.root_dir
    os.makedirs(root_dir, exist_ok=True)

    featurization = config.dataset.featurization

    if FEATURIZATION_TYPE[featurization] == "graphs" and (
        config.mixture_model.mol_encoder.type != "gnn"
        and config.mixture_model.mol_encoder.type != "equivariant_gnn"
        and config.mixture_model.mol_encoder.type != "mpnn"
    ):
        raise ValueError(
            f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mixture_model.mol_encoder.type}"
        )

    if FEATURIZATION_TYPE[featurization] == "tensors" and (
        config.mixture_model.mol_encoder.type == "gnn"
        or config.mixture_model.mol_encoder.type == "equivariant_gnn"
        or config.mixture_model.mol_encoder.type == "mpnn"
    ):
        raise ValueError(
            f"featurization is:{FEATURIZATION_TYPE[featurization]} but molecule encoder is: {config.mixture_model.mol_encoder.type}"
        )

    # Dataset
    dataset = DATA_CATALOG[config.dataset.name]()
    property = config.dataset.property

    mixture_task = MixtureTask(
        property=property,
        dataset=dataset,
        featurization=featurization,
    )

    # Split Loader - Only use first split (split_num=0)
    split_loader = SplitLoader(split_type="kfold")

    split_num = 0
    run_name = f"sweep_split_{split_num}"

    print(f"Training/validating on split {split_num}")
    train_indices, val_indices, test_indices = split_loader(
        property=mixture_task.property,
        cache_dir=mixture_task.dataset.data_dir,
        split_num=split_num,
    )

    # Data Loader
    train_dataset = Subset(mixture_task, train_indices.tolist())
    val_dataset = Subset(mixture_task, val_indices.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate,
        num_workers=config.num_workers,
    )

    model = build_mixture_model(config=config.mixture_model)
    model = model.to(device)

    # Save hyper parameters
    experiment_name = (
        f"{config.dataset.featurization}_{config.mixture_model.mix_encoder.type}"
    )
    OmegaConf.save(config, f"{root_dir}/hparams_{experiment_name}.yaml")

    # Training with WandB logger
    wandb_logger = WandBLogger(project=wandb.run.project, log_dir=root_dir)

    train(
        root_dir=root_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_type=config.loss_type,
        lr_mol_encoder=config.lr_mol_encoder,
        lr_other=config.lr_other,
        device=device,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        experiment_name=run_name,
        wandb_logger=wandb_logger,
    )

    print(f"Testing on split {split_num}")

    # Data Loader (one big batch)
    test_dataset = Subset(mixture_task, test_indices.tolist())
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_dataset.__len__(),
        collate_fn=custom_collate,
        num_workers=config.num_workers,
    )

    metric_dict, y_pred, y_test = predict(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    print(metric_dict)

    # Log final test metrics to wandb
    wandb.log(
        {
            "test_mae": metric_dict["mae"],
            "test_rmse": metric_dict["rmse"],
            "test_r2": metric_dict["r2"],
        }
    )

    test_metrics = pd.DataFrame(metric_dict, index=["metrics"]).transpose()
    test_metrics.to_csv(os.path.join(config.root_dir, f"{run_name}_test_metrics.csv"))

    y_pred = y_pred.detach().cpu().numpy().flatten()
    y_test = y_test.detach().cpu().numpy().flatten()
    test_predictions = pd.DataFrame(
        {
            "Predicted_Experimental_Values": y_pred,
            "Ground_Truth": y_test,
            "MAE": np.abs(y_pred - y_test),
        },
        index=range(len(y_pred)),
    )
    test_predictions.to_csv(
        os.path.join(config.root_dir, f"{run_name}_test_predictions.csv"),
        index=False,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
