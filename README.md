# A Comparative Study of State-of-the-Art Deep Learning Models for Semantic Segmentation of Pores in Scanning Electron Microscope Images of Activated Carbon

## ABSTRACT

Accurate measurement of the microspores, mesopores, and macropores on the surface of the activated carbon is essential due to its direct influence on the material’s adsorption capacity, surface area, and overall performance in various applications like water purification, air filtration, and gas separation. Traditionally, Scanning Electron Microscopy (SEM) images of activated carbons are collected and manually annotated by a human expert to differentiate and measure different pores on the surface. However, manual analysis of such surfaces is costly, time-consuming, and resource-intensive, as it requires expert supervision. In this paper, we propose an automatic deep-learning-based solution to address this challenge of activated carbon surface segmentation. Our deep-learning approach optimizes pore analysis by reducing time and resources, eliminating human subjectivity, and effectively adapting to diverse pore structures and imaging conditions. We introduce a novel SEM image segmentation dataset for activated carbon, comprising 128 images that capture the variability in pore sizes, structures, and imaging artifacts. Challenges encountered during dataset creation, irregularities in pore structures, and the presence of impurities were addressed to ensure robust model performance. We then evaluate the state-of-the-art deep learning models on the novel semantic segmentation task that shows promising results. Notably, DeepLabV3Plus, DeepLabV3, and FPN emerge as the most promising models based on semantic segmentation test results, with DeepLabV3Plus achieving the highest test Dice coefficient of 68.678%. Finally, we explore the optimization of learning rates for each model, outline the key research challenges, and discuss potential research directions to address these challenges.

<div align="center">
    <img src="https://github.com/user-attachments/assets/4dac2ab9-67b6-407e-b3e5-c19e4b6758d9" alt="Issue screenshot" />
</div>


```
@ARTICLE{10478488,
  author={Pokharel, Bishwas and Pandey, Deep Shankar and Sapkota, Anjuli and Yadav, Bhimraj and Gurung, Vasanta and Adhikari, Mandira Pradhananga and Regmi, Lok Nath and Adhikari, Nanda Bikram},
  journal={IEEE Access}, 
  title={A Comparative Study of State-of-the-Art Deep Learning Models for Semantic Segmentation of Pores in Scanning Electron Microscope Images of Activated Carbon}, 
  year={2024},
  volume={12},
  number={},
  pages={50217-50243},
  keywords={Scanning electron microscopy;Carbon;Imaging;Analytical models;Deep learning;Data models;Transmission electron microscopy;Semantic segmentation;SEM images;activated carbon;ground truth;Adam optimizer;intersection over union;dice coefficient},
  doi={10.1109/ACCESS.2024.3381523}}
```

## Getting Started

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/bhimrazy/sem-segmentation.git
   cd sem-segmentation
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

### Dataset Preparation
Dataset: [Google Drive Link](https://drive.google.com/file/d/1arcACo6jnXPurgLeVfFkm-jsyvkHhhZK) 

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
- `TransUNet`: TransUnet from `mkaa44`: default image size to 256
- `SwinUnet`: SwinUnet from `HuCaoFighting`: default image size to 224

### Project Structure

- `data/`: Contains the dataset used for training and testing.
- `src/`: Contains the source code of the project.
- `scripts/`: Contains utility scripts, such as downloading datasets.
- `configs/`: Contains configuration files for model training.
- `artifacts/`: Contains trained model checkpoints and prediction images.
- `main.py`: Main training script.
- `requirements.txt`: List of required packages.

## License

This project is licensed under the [MIT License](LICENSE).
