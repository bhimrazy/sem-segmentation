import torch
import matplotlib.pyplot as plt


def plot_predictions(images, masks, pred, BATCH, title="Test", figsize=(20, 10)):
    # Create a figure with a grid of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=BATCH, figsize=figsize)

    fig.suptitle(f"{title} Data Predictions", fontsize=16)
    fig.tight_layout()

    # Plot the images in the subplots
    for i, ax in enumerate(ax1):
        # Get the image and label for the current subplot
        image = images[i].cpu()
        image = image.numpy().transpose((1, 2, 0))
        ax.imshow(image, cmap="gray")
        ax.title.set_text("Images")
        ax.axis("off")

    # Plot the images in the subplots
    for i, ax in enumerate(ax2):
        # Get the image and label for the current subplot
        image = masks[i].cpu()
        image = image.numpy().transpose((1, 2, 0))
        ax.imshow(image, cmap="gray")
        ax.title.set_text("Masks")
        ax.axis("off")

    # Plot the images in the subplots
    for i, ax in enumerate(ax3):
        # Get the image and label for the current subplot
        image = torch.sigmoid(pred[i]).cpu()
        # image = torch.where(image > 0.5, torch.tensor(1), torch.tensor(0))
        image = image.numpy().transpose((1, 2, 0))
        ax.imshow(image, cmap="gray")
        ax.title.set_text("Predicted Masks")
        ax.axis("off")

    return plt
