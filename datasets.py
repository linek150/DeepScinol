import torchvision.datasets
import torchvision.transforms
from torch.utils.data import Dataset
from sklearn import datasets
import torch
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter
from config import DATASET_DIR, WRITER_PREFIX
from matplotlib import pyplot as plt
from config_enums import DatasetEnum, LossEnum
from typing import Tuple
from torchvision.transforms import ToTensor, Normalize

class CustomDataset(Dataset):
    def __init__(self, name: DatasetEnum, n_dim: int = 100, validation=False):
        self.name = name
        self.input_size = None
        self.output_size = None
        self.no_classes = None
        self.data = None
        self.default_loss = None
        self.validation_data_loader = None
        if validation:
            self.make_validation_dataset(name)
        else:
            self.make_train_dataset(name)
            self._set_in_out_size()
            self._set_loss()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    def _set_in_out_size(self):
        if self.name in [DatasetEnum.FLATTEN_MNIST]:
            self.input_size = self.data[0][0].shape[0]
            self.output_size = self.no_classes
        elif self.name in [DatasetEnum.ND_REG, DatasetEnum.ONED_REG, DatasetEnum.DUM_ABS]:
            self.input_size = int(self.data[0][0].shape[0])
            self.output_size = int(self.data[0][1].shape[0])
        elif self.name in [DatasetEnum.MNIST, DatasetEnum.CIFAR10]:
            self.input_size = None
            self.output_size = None
        else:
            raise NotImplementedError(f"Specify how to define in_out_size for {self.name}.")

    def _set_loss(self):
        if self.name in [DatasetEnum.FLATTEN_MNIST, DatasetEnum.MNIST, DatasetEnum.CIFAR10]:
            self.default_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        elif self.name in [DatasetEnum.ND_REG, DatasetEnum.ONED_REG]:
            self.default_loss = torch.nn.MSELoss(reduction='mean')
        elif self.name in [DatasetEnum.DUM_ABS]:
            self.default_loss = LossEnum.MAE
        else:
            raise NotImplementedError(f"Specify default loss for {self.name}.")

    def _get_formated_data(self, x, y):
        return [(x[i], y[i]) for i in range(len(x))]

    def make_train_dataset(self, name:DatasetEnum):
        if name == DatasetEnum.ND_REG:
            x, y = datasets.make_regression(n_features=n_dim, n_samples=100, random_state=0, noise=1)
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(1)
            self.data = self._get_formated_data(x,y)

        if name == DatasetEnum.FLATTEN_MNIST:
            # only vector as input
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten])
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True, transform=transform)
            self.no_classes = len(self.data.classes)

        if name == DatasetEnum.MNIST:
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True,
                                                   transform=torchvision.transforms.ToTensor())
            self.no_classes = len(self.data.classes)
        if name == DatasetEnum.CIFAR10:
            transform = torchvision.transforms.Compose([ ToTensor(),
                                                        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.data = torchvision.datasets.CIFAR10(DATASET_DIR, download=True,
                                                     transform=transform)
            self.no_classes = len(self.data.classes)
        if name == DatasetEnum.DUM_ABS:
            x = torch.ones(1).view(-1, 1)
            y = torch.ones_like(x)*(-10)
            self.data = self._get_formated_data(x, y)
        if name == DatasetEnum.ONED_REG:
            x = torch.arange(-5, 5, 0.1).view(-1, 1)
            y = (-5 * x + 0.1 * torch.randn(x.size()))
            self.data = self._get_formated_data(x, y)

    def make_validation_dataset(self, name: DatasetEnum):
        if name == DatasetEnum.FLATTEN_MNIST:
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten])
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True, transform=transform, train=False)
            self.no_classes = len(self.data.classes)
        elif name == DatasetEnum.CIFAR10:
            transform = torchvision.transforms.Compose([ToTensor(),
                                                        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.data = torchvision.datasets.CIFAR10(DATASET_DIR, download=True,
                                                     transform=transform, train=False)
            self.no_classes = len(self.data.classes)
        elif name == DatasetEnum.MNIST:
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True,
                                                   transform=torchvision.transforms.ToTensor(), train=False)
            self.no_classes = len(self.data.classes)
        else:
            raise NotImplementedError(f"Create validation set for {name.value}")

