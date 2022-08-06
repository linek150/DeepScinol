import os.path
import numpy as np
import pandas as pd
import torchvision.datasets
import torchvision.transforms
from torch.utils.data import Dataset
from sklearn import datasets
import torch
import requests
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
        self.make_dataset(name, validation=validation)
        if not validation:
            self._set_in_out_size()
            self._set_loss()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]

    def _set_in_out_size(self):
        if self.name in [DatasetEnum.FLATTEN_MNIST, DatasetEnum.ADULT, DatasetEnum.COVER_TYPE]:
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
        if self.name in [DatasetEnum.FLATTEN_MNIST, DatasetEnum.MNIST, DatasetEnum.CIFAR10, DatasetEnum.ADULT,
                         DatasetEnum.COVER_TYPE]:
            self.default_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        elif self.name in [DatasetEnum.ND_REG, DatasetEnum.ONED_REG]:
            self.default_loss = torch.nn.MSELoss(reduction='mean')
        elif self.name in [DatasetEnum.DUM_ABS]:
            self.default_loss = LossEnum.MAE
        else:
            raise NotImplementedError(f"Specify default loss for {self.name}.")

    def _get_formated_data(self, x, y):
        return [(x[i], y[i]) for i in range(len(x))]

    def make_dataset(self, name:DatasetEnum, validation:bool):
        train = not validation
        if name == DatasetEnum.ND_REG:
            x, y = datasets.make_regression(n_features=n_dim, n_samples=100, random_state=0, noise=1)
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(1)
            self.data = self._get_formated_data(x,y)

        if name == DatasetEnum.FLATTEN_MNIST:
            # only vector as input
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten])
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True, transform=transform, train=train)
            self.no_classes = len(self.data.classes)

        if name == DatasetEnum.MNIST:
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True,
                                                   transform=torchvision.transforms.ToTensor(), train=train)
            self.no_classes = len(self.data.classes)

        if name == DatasetEnum.CIFAR10:
            transform = torchvision.transforms.Compose([ ToTensor(),
                                                        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            self.data = torchvision.datasets.CIFAR10(DATASET_DIR, download=True,
                                                     transform=transform, train=train)
            self.no_classes = len(self.data.classes)

        if name == DatasetEnum.ADULT:
            x, y = self.get_adult_dataset(train=train)
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False)
            self.no_classes = len(y.unique())
            self.data = self._get_formated_data(x, y)
        if name == DatasetEnum.COVER_TYPE:
            x,y = self.get_covertype_dataset(train=train)
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y = torch.tensor(y, dtype=torch.long, requires_grad=False)
            self.no_classes = len(y.unique())
            self.data = self._get_formated_data(x, y)

        if name == DatasetEnum.DUM_ABS:
            x = torch.ones(1).view(-1, 1)
            y = torch.ones_like(x)*(-10)
            self.data = self._get_formated_data(x, y)
        if name == DatasetEnum.ONED_REG:
            x = torch.arange(-5, 5, 0.1).view(-1, 1)
            y = (-5 * x + 0.1 * torch.randn(x.size()))
            self.data = self._get_formated_data(x, y)
        if self.data is None:
            raise NotImplementedError(f"Create dataset for {name.value}, train: {train}")

    def get_covertype_dataset(self, train):
        data= None
        file_name="covtype.data"
        file_path=os.path.join(DATASET_DIR, file_name)
        datasets_dir=os.path.basename(os.path.normpath(DATASET_DIR))
        if not os.path.exists(file_path):
            print("Downloading conver type dataset...")
            self.download_covertype_dataset_file(datasets_dir, file_name)
        data=pd.read_csv(file_path, header=None)
        data_np=data.to_numpy(dtype=np.int64)
        train_size=int(4e4) #full size ~56000
        if train:
            data_np=data_np[:train_size]
        else:
            data_np=data_np[train_size:]
        return data_np[:, :-1], data_np[:, -1]-1 # -1 becouse classes labels are from 1 to 7

    def download_covertype_dataset_file(self, datasets_dir, file_name):
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        response=requests.get(url)
        if datasets_dir not in os.listdir():
            os.mkdir(datasets_dir)
        compresed_file_name=file_name+".gz"
        full_relative_file_path=os.path.join(datasets_dir, file_name)
        open(compresed_file_name, "wb").write(response.content)
        import gzip
        with gzip.open(compresed_file_name, "rb") as compressed:
            decompressed_bytes=compressed.read()
            with open(full_relative_file_path, "wb") as decompressed:
                decompressed.write(decompressed_bytes)
        os.remove(compresed_file_name)




    def get_adult_dataset(self, train):
        data = None
        file_name="adult.data" if train else "adult.test"
        file_path=os.path.join(DATASET_DIR, file_name)
        datasets_dir=os.path.basename(os.path.normpath(DATASET_DIR))

        if not os.path.exists(file_path):
            print("Downloading adult dataset...")
            self.download_adult_dataset_file(datasets_dir, file_name)
        skip_N_rows=0 if train else 1 #there is line of description in test file
        data=pd.read_csv(file_path, header=None, skiprows=skip_N_rows)
        if train:
            #such country doesnt exist in test set so it is removed here
            data.drop(data.index[data[13] == " Holand-Netherlands"], inplace=True)
        data_np = self.preprocess_adult_dataset(data)
        return data_np[:, :-1], data_np[:, -1]

    def preprocess_adult_dataset(self, data):
        for column_name in data:
            if data[column_name].dtype == object:
                if len(data[column_name].value_counts()) == 2:
                    data[column_name], _ = pd.factorize(data[column_name])
                else:
                    data = pd.get_dummies(data, columns=[column_name], prefix='', prefix_sep='')
        data_np = data.to_numpy(dtype=np.int32)
        return data_np

    def download_adult_dataset_file(self, datasets_dir, file_name):
        file_url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"+file_name
        response=requests.get(file_url)
        if datasets_dir not in os.listdir():
            os.mkdir(datasets_dir)
        open(os.path.join(datasets_dir, file_name),"wb").write(response.content)