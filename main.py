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

    # Training predictions
    predict_and_log(model, data_module.train_dataloader(), "Train", wandb_logger, cfg)

    # Validation predictions
    predict_and_log(model, data_module.val_dataloader(), "Valid", wandb_logger, cfg)

    # Test predictions
    predict_and_log(model, data_module.test_dataloader(), "Test", wandb_logger, cfg)


def predict_and_log(model, dataloader, title, logger, cfg):
    images, masks = next(iter(dataloader))

    # Predict
    with torch.no_grad():
        model.eval()
        pred = model(images)

    # Plot
    fig = plot_predictions(
        images, masks, pred, cfg["experiment"]["batch_size"], title=title
    )

    # Save the plot to a file
    predictions = f"artifacts/{title.lower()}-predictions.png"
    fig.savefig(predictions)

    # Log to WandB
    logger.experiment.log({"predictions": wandb.Image(fig)})

    # Log to MLflow
    mlflow.log_artifact(predictions)

    # Close the figure
    fig.close()


if __name__ == "__main__":
    main()
