import os
import wandb
import torch
import argparse
from lightning import seed_everything
from src.model import RudrakshaSegModel
from src.dataset import RudrakshaDataModule
from src.io import load_data
from src.utils import split_data
from os.path import join
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Rudraksha Segmentation Evaluation')
parser.add_argument('-m','--model', default='UNet', type=str, help='model name')
parser.add_argument('-r','--modelreg', default='rudraksha-segmentation/Semantic Segmentation UNet lr_fixed/model-4jgahlq4:v0', type=str, help='WandB model registry')

args = parser.parse_args()
run = wandb.init(project="evaluation")


cfg = {
    "data":{
        "data_dir": "data",
        "dataset_folder": "RudrakshaDataset"
    },
    "model":{
        "name": args.model,
        "num_classes": 1,
        "smp_encoder": "resnet18"
    },
    "loss":{
        "name": "GeneralizedDiceLoss"
    },
    "experiment":{
        "name": "Rudraksha Segmentation",
        "num_epochs": 100,
        "patience": 20,
        "image_size": 256,
        "batch_size": 8,
        "learning_rate": 0.001,
        "split_ratio": 0.2,
        "num_workers": 1,
        "accelerator": "cpu", # cpu, cuda, or mps
        "devices": "auto",
        "random_seed": 42
    }
}

seed_everything(cfg["experiment"]["random_seed"])

# Load model from WandB
model_registry = args.modelreg
artifact = run.use_artifact(model_registry, type='model')
artifact_dir = artifact.download()

checkpoint_path = os.path.join(artifact_dir, 'model.ckpt')

# Load the model
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

model = RudrakshaSegModel(model_name=cfg["model"]["name"], num_classes=1, smp_encoder="resnet18", loss_fn="GeneralizedDiceLoss")
model.load_state_dict(checkpoint["state_dict"])

# Evaluate the model




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


y_true = []
y_pred = []

data_module.setup()
with torch.no_grad():
    model.eval()
    for batch in data_module.test_dataloader():
        x, y = batch
        y_hat = model(x)
        
        y_true.extend(y)
        y_pred.extend(y_hat)



def plot_masks_and_histograms(y_true, y_pred):
    # Calculate the number of rows needed (5 images per row)
    num_rows = 4

    # Create subplots with 4 rows and 5 columns initially
    fig, axs = plt.subplots(4, 5, figsize=(20, 15))

    for i in range(num_rows):
        for j in range(5):
            idx = i * 5 + j  # Calculate the index of the image in the arrays
            if idx >= len(y_true):
                break  # Break if we have displayed all images
            # Plot the first row of the true mask
            axs[0, j].imshow(y_true[idx].permute(1,2,0), cmap='gray')
            axs[0, j].set_title('True Mask')

            # Plot the histogram of the true mask
            axs[1, j].hist(y_true[idx].flatten(), bins=256, color='blue')
            axs[1, j].set_title('Histogram (True)')

            # Plot the predicted mask
            axs[2, j].imshow(y_pred[idx].sigmoid().permute(1,2,0), cmap='gray')
            axs[2, j].set_title('Predicted Mask')

            # Plot the histogram of the predicted mask
            axs[3, j].hist(y_pred[idx].sigmoid().flatten(), bins=256, color='red')
            axs[3, j].set_title('Histogram (Predicted)')

    # Set common y-axis labels
    for j in range(5):
        axs[0, j].set_ylabel('True Mask')
        axs[1, j].set_ylabel('Histogram (True)')
        axs[2, j].set_ylabel('Predicted Mask')
        axs[3, j].set_ylabel('Histogram (Predicted)')

    plt.tight_layout()
    plt.savefig('evaluation.png')
    plt.close()

plot_masks_and_histograms(y_true, y_pred)



# white pixels ratio
mask_white_counts = []
pred_mask_white_counts = []
images = []

for i, (mask, pred_mask) in enumerate(zip(y_true[:8], y_pred[:8])):  

    no_of_white_pixel_in_mask = torch.sum(mask.flatten())
    no_of_white_pixel_in_pred_mask = torch.sum(pred_mask.sigmoid().flatten())
    mask_white_counts.append(no_of_white_pixel_in_mask)
    pred_mask_white_counts.append(no_of_white_pixel_in_pred_mask)
    images.append(f"Mask{i+1}")


# Create the line plot
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

plt.plot(images, mask_white_counts, marker="o", label="Truth")
plt.plot(
    images,
    pred_mask_white_counts,
    marker="o",
    label="Predicted",
)

# Add labels to the axes
plt.xlabel("Batch")
plt.ylabel("White Count")

# Add title to the plot
plt.title("Number of White Counts")

pad = 100
# Display data values as points on the plot
for i, (x, y1, y2) in enumerate(zip(images, mask_white_counts, pred_mask_white_counts)):
    plt.text(x, y1 + pad, f"{int(y1)}", ha="center", va="baseline")
    plt.text(x, y2 + pad, f"{int(y2)}", ha="center", va="baseline")

# Add a legend in the best location
plt.legend(loc="best")

# Show the plot
plt.savefig('white_count.png')
plt.close()