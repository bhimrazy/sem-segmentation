# Rudraksha Segmentation Project

## Introduction

The Rudraksha Segmentation Project focuses on accurate segmentation of individual Rudraksha dark spot in images. Leveraging advanced deep learning models, the project contributes to the field of image analysis and cultural studies.

## Getting Started

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bhimrazy/rudraksha-segmentation.git
   cd rudraksha-segmentation
   ```

2. Set up a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Add environment variables:

   ```bash
   cp .env.example .env
   ```

   Then add the value for `RUDRAKSHA_DATASET_ID` in `.env` file.

### Dataset Preparation

To prepare the dataset, run the following command:\
It will download the dataset from google drive and extract it to `data/` directory.

```bash
make dataset
```

### Training

Run the training script using one of the following methods:

```bash
python main.py

# switch model
python main.py model.name="MODEL_KEY_NAME"
```

Available models with their keys in the factroy are:

- `UNet`: MONAI U-Net model
- `DeepLabV3`: smp DeepLabV3 model
- `DeepLabV3Plus`: smp DeepLabV3Plus model
- `FCN8s`: Custom FCN8s model
- `CustomUNet`: Custom UNet model
- `CustomResUNet`: Custom ResUNet model
- `AttResUNet`: Custom Attention ResUNet model
- `AttentionUNet`: Custom Attention UNet model
- `FPN`: Smp FPN model
- `UNETR`: MONAI UNETR model
- `SwinUNETR`: MONAI SwinUNETR model
- `SmpResUNet`: Smp ResUNet model
- `SmpResUNetPlusPlus`: Smp ResUNetPlusPlus model

### Project Structure

- `data/`: Contains the dataset used for training and testing.
- `src/`: Contains the source code of the project.
- `scripts/`: Contains utility scripts, such as downloading datasets.
- `configs/`: Contains configuration files for model training.
- `artifacts/`: Contains trained model checkpoints and prediction images.
- `main.py`: Main training script.
- `requirements.txt`: List of required packages.

## Contributing

If you'd like to contribute to this project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
