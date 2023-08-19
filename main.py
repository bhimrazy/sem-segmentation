import argparse
from os.path import join

import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import wandb
from src.dataset import RudrakshaDataModule
from src.io import load_config, load_data
from src.model import RudrakshaSegModel
from src.utils import split_data
from src.viz import plot_predictions

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

    # seed
    seed_everything(cfg["experiment"]["random_seed"])

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
        cfg["experiment"]["image_size"],
    )

    # model
    model = RudrakshaSegModel(
        num_classes=cfg["model"]["num_classes"], lr=cfg["experiment"]["lr"]
    )

    # mlflow
    mlflow.set_experiment(cfg["experiment"]["name"])
    mlflow.set_tag("model", cfg["model"]["name"])
    mlflow.pytorch.autolog()

    # Create WandB and MLflow loggers
    wandb_logger = WandbLogger(
        project=cfg["experiment"]["name"], log_model=True, tags=[cfg["model"]["name"]]
    )
    mlflow_logger = MLFlowLogger(experiment_name=cfg["experiment"]["name"])

    # learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # trainer
    trainer = Trainer(
        max_epochs=cfg["experiment"]["num_epochs"],
        accelerator=cfg["experiment"]["accelerator"],
        devices=cfg["experiment"]["devices"],
        logger=[wandb_logger, mlflow_logger],
        log_every_n_steps=2,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor],
    )

    # train
    trainer.fit(model, data_module)

    # test
    trainer.test(model, data_module)

    # model path
    model_path = "artifacts/model.pth"
    # save model
    torch.save(model, model_path)

    # # load model
    model = torch.load(model_path)

    data_module.setup()
    images, masks = next(iter(data_module.train_dataloader()))

    # predict
    with torch.no_grad():
        model.eval()
        pred = model(images)

    # plot
    fig = plot_predictions(
        images, masks, pred, cfg["experiment"]["batch_size"], title="Train"
    )

    # Save the plot to a file
    predictions = "artifacts/train-predictions.png"
    fig.savefig(predictions)
    wandb_logger.experiment.log({"predictions": wandb.Image(fig)})
    mlflow.log_artifact(predictions)
    fig.close()

    images, masks = next(iter(data_module.val_dataloader()))

    # predict
    with torch.no_grad():
        model.eval()
        pred = model(images)

    # plot
    fig = plot_predictions(
        images, masks, pred, cfg["experiment"]["batch_size"], title="Val"
    )

    # Save the plot to a file
    predictions = "artifacts/valid-predictions.png"
    fig.savefig(predictions)
    wandb_logger.experiment.log({"predictions": wandb.Image(fig)})
    mlflow.log_artifact(predictions)
    fig.close()

    images, masks = next(iter(data_module.test_dataloader()))

    # predict
    with torch.no_grad():
        model.eval()
        pred = model(images)

    # plot
    fig = plot_predictions(
        images, masks, pred, cfg["experiment"]["batch_size"], title="Test"
    )

    # Save the plot to a file
    predictions = "artifacts/test-predictions.png"
    fig.savefig(predictions)
    wandb_logger.experiment.log({"predictions": wandb.Image(fig)})
    mlflow.log_artifact(predictions)
    fig.close()


if __name__ == "__main__":
    main()
