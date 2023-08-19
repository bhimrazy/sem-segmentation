import argparse
from os.path import join

import mlflow
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from src.dataset import RudrakshaDataModule
from src.io import load_config, load_data
from src.model import RudrakshaSegModel
from src.utils import split_data

# argument parser
parser = argparse.ArgumentParser(description="Main script pipeline")
parser.add_argument(
    "-c", "--config", type=str, required=True, help="path to config file"
)


def main():
    # parse arguments
    args = parser.parse_args()
    # load config
    cfg = load_config(args.config)

    dataset_path = join(cfg["data"]["data_dir"], cfg["data"]["dataset_folder"])

    # load data
    images, masks = load_data(dataset_path)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        images,
        masks,
        cfg["experiment"]["split_ratio"],
        cfg["experiment"]["split_ratio"],
        cfg["experiment"]["random_seed"],
    )

    # data module
    data_module = RudrakshaDataModule(
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        cfg["experiment"]["batch_size"],
        cfg["experiment"]["num_workers"],
    )

    # model
    model = RudrakshaSegModel(cfg["model"]["num_classes"])

    # mlflow
    mlflow.set_experiment(cfg["experiment"]["name"])
    mlflow.pytorch.autolog()

    # trainer
    trainer = Trainer(
        max_epochs=cfg["experiment"]["num_epochs"],
        accelerator=cfg["experiment"]["accelerator"],
        devices=cfg["experiment"]["devices"],
        logger=MLFlowLogger(),
        log_every_n_steps=2,
        check_val_every_n_epoch=1,
        callbacks=[],
    )

    # train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
