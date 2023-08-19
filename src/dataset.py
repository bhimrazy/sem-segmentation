from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RudrakshaDataset:
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


class RudrakshaDataModule(LightningDataModule):
    def __init__(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        batch_size=4,
        num_workers=4,
    ):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

    def setup(self, stage=None):
        self.train_dataset = RudrakshaDataset(
            self.X_train, self.y_train, self.transform
        )
        self.valid_dataset = RudrakshaDataset(
            self.X_valid, self.y_valid, self.transform
        )
        self.test_dataset = RudrakshaDataset(self.X_test, self.y_test, self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
