# Project Overview:

## Tools and Technologies Utilized

1. [PyTorch](https://pytorch.org/)
   This Project is based on PyTorch. Pytorch is really flexible to implement any kind of neural network. It is also very fast and easy to use. It is also very popular among the research community.
2. [PyTorch Lightning](https://lightning.ai)
   PyTorch Lightning is a wrapper around PyTorch that handles a lot of the boilerplate code and makes it easier to train models. It also makes it easier to train models on multiple GPUs and TPUs. It also has a lot of other features like logging, checkpointing, etc. which has been used in this project.
3. [Segmentation Models Pytorch](https://smp.readthedocs.io)
   Segmentation Models Pytorch is a library that provides a lot of pre-trained models for segmentation. It also provides a lot of loss functions for segmentation.
4. [MONAI](https://monai.io/)
   MONAI is a PyTorch-based, open-source framework for deep learning It provides a lot of tools for medical imaging. It also provides a lot of pre-trained models for medical imaging.
5. [Hydra](https://hydra.cc/)
   Hydra is a framework for elegantly configuring complex applications. It is used to configure the hyperparameters of the model. It also makes it easier to run the same model with different hyperparameters. In this project it has been used to control several parameters like dataset, model , optimizer, loss, etc. based on the configuration file.
6. [Matplotlib](https://matplotlib.org/)
   Matplotlib is a plotting library for Python. It is used to plot the images and masks. It is also used to plot the loss and accuracy curves.
7. [Wandb](https://wandb.ai/)
   Wandb is a tool for visualizing and logging the training process. In this project it has been used to log the training process and visualize the results, like the loss, accuracy, etc. ALong with that it also logs the hyperparameters and the model architecture, with several checkpoints. It also provides a beautiful dashboard to visualize the results and also compare the results of different experiments.
8. [MLFlow](https://mlflow.org/)
   MLFlow is a tool for tracking the experiments. It is used to track the experiments and compare the results of different experiments. In this project it has been used to track the experiments and compare the results of different experiments.

## Project Configuration

### Seed Everything

    In this project, a seed 42 has been used. This is to ensure that the results are reproducible. We can also use different seeds for different experiments to see how the results vary with different seeds.

### Configuring the Dataset

#### Data Distribution

    dataset_path = data/RudrakshaDataset
    Dataset Info:
    Number of images: 128
    Number of masks: 128

    The dataset has been divided into 3 parts:
    1. Training Set: 60%
    2. Validation Set: 20%
    3. Test Set: 20%


    Number of training images: 78
    Number of validation images: 25
    Number of test images: 25

    Ratio of training images: 0.609375 ~ 60%
    Ratio of validation images: 0.1953125 ~ 20%
    Ratio of test images: 0.1953125 ~ 20%

    Train set is used for training the model and the validation set is used for validating the model during the training process.
    The test set is used to test the model after the training process is complete.

#### DataModule Configuration

    PyTorch Lightning DataModule is used to make the data loading process. It uses Pytorch Dtataset and DataLoader to load the data.

    In this project, it uses the following parameters:

    1. batch_size: 8. The batch size of the data loader. It mainly depends upon the GPU memory. If the GPU memory is large, we can use a larger batch size. But if the GPU memory is small, we need to use a smaller batch size.
    2. num_workers: 4. The number of workers to use for loading the data. It depends upon the number of CPU cores. If the number of CPU cores is large, we can use a larger number of workers. But if the number of CPU cores is small, we need to use a smaller number of workers. It helps in loading the data faster.
    3. image_size: 256. The size of the image. The image is resized to this size before feeding it to the model. We have kept it 256 because the model is trained on 256x256 images. If we want to train the model on larger images, we can increase this value.

    In our case, for the image transformation we have only used the resizing and conversion to tensor.

### Configuring the Model, Loss Function, Optimizer and Learning Rate Scheduler

    In this project, model is made based on lighting module. We have used Lightning modlue instead of bare pyttorch modlue because lighting provides all the built in functions to run training, validation , testing and logging. It also provides a lot of other features like checkpointing, logging, etc. which makes it easier to train the model.

    In side of the Lightining modlue we have imported our opytorch model class which comes based on the model factory. It is a factory design pattern which helps us to create different models based on the configuration. We can create different models by changing the
    name of the model that we want to load from the availabe models.

    Our `RudrakshaSegModel` ` class is a subclass of `LightningModule` class. It takes following parameters:

    1. model_name: The name of the model to use.
    We have our custom model defned from scratch and other models loaded from smp library and monai library. We have also used few of the opensouce models from github.
    2. smp_encoder: The name of the encoder to use. It is used only when the model is being used from smp library. This smp library has a lot of pre-trained models. We can use any of them by changing the name of the encoder. In our case we have used `resnet50` encoder. We can also use other encoders like `resnet34`, `resnet101`, etc.
    3. num_classes: The number of classes in the dataset. In our case it is 1. We have only one class in our dataset, which is the mask of the image.
    4. lr: The learning rate of the model. It is used by the optimizer to update the weights of the model. The initial learning rate is set to 0.001. But it is changed during the training process using the learning rate scheduler.


    Loss function: Even incase of loss function we have a factory setup to use loss functions based on the configuration. But Mainly we have used `GeneralizedDiceLoss` as our loss function.We mainly used this loass function as it is very good in terms of handling class imbalance.

    Optimizer: We have used `Adam` optimizer. It is a very popular optimizer and it is very good for training deep neural networks. It is also very fast and easy to use. It is also very popular among the research community.

    Learning Rate Scheduler: We have used `ReduceLROnPlateau` learning rate scheduler. It is a very popular learning rate scheduler. It reduces the learning rate when the validation loss stops improving. It helps in improving the performance of the model.
    We have following parameters choosen for the learning rate scheduler:
    1. mode: min. It means that the learning rate will be reduced when the validation loss stops improving.
    2. factor: 0.5. It means that the learning rate will be reduced by a factor of 0.5.
    3. patience: 5. It means that the learning rate will be reduced when the validation loss stops improving for 5 epochs.
    4. verbose: True. It means that the learning rate scheduler will print a message when the learning rate is reduced.
    5. threshold: 0.001. It means that the learning rate will be reduced only when the validation loss stops improving by more than 0.0001.

### Configuring the Training Process

    We have also used `EarlyStopping` callback. It is used to stop the training process when the validation loss stops improving. It helps in preventing overfitting.

    We have also used `ModelCheckpoint` callback. It is used to save the best model during the training process. It helps in preventing overfitting. In our case it saves top 3 models based on the validation loss.

    We have also used `LearningRateMonitor` callback. It is used to log the learning rate during the training process. It helps in visualizing the learning rate during the training process.

    We have also used `WandbLogger` callback. It is used to log the training process. It helps in visualizing the training process.
    We hae also used `MLFlowLogger` callback. It is used to log the training process. It helps in visualizing the training process.

    We could have just use any one of the logging tool but we have used both of them to show how we can use multiple logging tools at the same time. We can also use other logging tools like Tensorboard, Neptune, etc.
    We mainly used MLFlow to visualize experiments locally and then also used wandb at the same time to share the results with the team and also to log artifacts like images, checkpoints of model. We can also use MLFlow to log artifacts but we have used wandb for that.

    In this project,We have used `Trainer` class from PyTorch Lightning. It is used to train the model. It uses the following parameters:

    1. max_epochs: 100. The maximum number of epochs to train the model. It is set to 100. But the training process will stop if the validation loss stops improving. It is updated based on the experimentation needs.
    2. accelerator: cuda. It means that the model will be trained on GPU. If we want to train the model on CPU, we can change it to cpu.
    3. devices: auto. It means that the model will be trained on all the available GPUs. If we want to train the model on a specific GPU, we can change it to the index of the GPU.
    4. logger = [wandb_logger, mlflow_logger]. It means that the training process will be logged using both wandb and mlflow.
    5. callbacks = [early_stopping, checkpoint_callback, lr_monitor]. It means that the training process will use early stopping, model checkpoint and learning rate monitor callbacks.
    6. log_every_n_steps: 2. It means that the training process will log the results after every step.
    7. check_val_every_n_epoch = 1. It means that the validation process will be run after every epoch.

    Trainer is used to train the model and also run validation and testing process.

### Log Artifacts

    In this project after the taining process is complete we log the artifacts like prediction images, checkpoints of model, etc. We have used wandb to log the artifacts.
