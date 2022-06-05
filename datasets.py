import torchvision.datasets
import torchvision.transforms
from torch.utils.data import Dataset
from sklearn import datasets
import torch
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter
from config import DATASET_DIR, WRITER_PREFIX
from matplotlib import pyplot as plt
from config_enums import DatasetEnum


class CustomDataset(Dataset):
    def __init__(self, name: DatasetEnum, n_dim: int = 100):
        self.name = name
        self.input_size=None
        self.output_size=None
        self.no_classes = None
        self.data = None
        self.default_loss= None
        if name == DatasetEnum.ONED_REG:
            x = torch.arange(-5, 5, 0.1).view(-1, 1)
            y = (-5 * x + 0.1 * torch.randn(x.size()))
            self.data = [(x[i], y[i]) for i in range(len(x))]
            #fig = plt.figure()
            #plt.scatter(y,x)
            #writer = SummaryWriter(WRITER_PREFIX + "\dataset")
            #writer.add_figure("data_plot", fig)


        if name == DatasetEnum.ND_REG:
            x, y = datasets.make_regression(n_features=n_dim, n_samples=100, random_state=0, noise=1)
            x = torch.tensor(x, dtype=torch.float, requires_grad=False)
            y = torch.tensor(y, dtype=torch.float, requires_grad=False).unsqueeze(1)
            self.data = [(x[i], y[i]) for i in range(len(x))]

        if name == DatasetEnum.MNIST:
            # only vector as input
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten])
            self.data = torchvision.datasets.MNIST(DATASET_DIR, download=True, transform=transform)
            self.no_classes = len(self.data.classes)

        self._set_in_out_size()
        self._set_loss()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _set_in_out_size(self):
        if self.name in [DatasetEnum.MNIST]:
            self.input_size = self.data[0][0].shape[0]
            self.output_size = self.no_classes
        if self.name in [DatasetEnum.ND_REG, DatasetEnum.ONED_REG]:
            self.input_size = int(self.data[0][0].shape[0])
            self.output_size = int(self.data[0][1].shape[0])
        assert self.input_size is not None
        assert self.output_size is not None

    def _set_loss(self):
        if self.name in [DatasetEnum.MNIST]:
            self.default_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        if self.name in [DatasetEnum.ND_REG, DatasetEnum.ONED_REG]:
            self.default_loss = torch.nn.functional.mse_loss
        assert self.default_loss is not None

