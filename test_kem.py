#!/usr/bin/env python3
import torch
import argparse
import torchvision
from torchvision import datasets, transforms
from models import *
from tqdm import tqdm
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import strftime
import sys


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-bs', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', '-tbs', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', '-e', type=int, default=15,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', '-s', type=int, default=None, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--optimizer', '-o', default="adam",
                        choices=["adam", "sgd", "scinol2dl", "sinh", "nsinh"])
    parser.add_argument('--model', '-m', default="lr",
                        choices=["cnn", "lr", "fcn4", "fcn2", "resnet18", "vgg16", "alexnet", "inception_v3"])
    parser.add_argument('--dataset', '-d', default="mnist",
                        choices=["mnist", "fmnist", 'cifar10', "cifar100", "synth"])
    parser.add_argument('--loss', '-l', default="ce", choices=["ce", "mse"])
    parser.add_argument('--tag', '-t', default=None)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                       transform=transform)
        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transform)
    elif args.dataset == "fmnist":
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True,
                                              transform=transform)
        test_dataset = datasets.FashionMNIST('../data', train=False,
                                             transform=transform)
    elif args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                         transform=transform)
        test_dataset = datasets.CIFAR10('../data', train=False,
                                        transform=transform)
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                                          transform=transform)
        test_dataset = datasets.CIFAR100('../data', train=False,
                                         transform=transform)
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    if args.loss == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    elif args.loss == "mse":
        loss_fn = torch.nn.MSELoss(reduction='sum')

    num_outputs = len(train_loader.dataset.classes)
    input_shape = train_loader.dataset.data.shape[1:]
    if args.model == "lr":
        model = LR(input_shape, num_outputs)
    elif args.model == "cnn":
        model = CNN(input_shape, num_outputs)
    elif args.model == "fcn4":
        model = FCN4(input_shape, num_outputs)
    elif args.model == "fcn2":
        model = FCN2(input_shape, num_outputs)
    elif args.model == "resnet18":
        model = torchvision.models.resnet18()
        model.fc = nn.Linear(512, num_outputs)
    elif args.model == "inception_v3":
        raise NotImplementedError()
        # model = torchvision.models.inception_v3()
        # model.fc = nn.Linear(512, num_outputs)
    elif args.model == "vgg16":
        # out of cuda memory :(
        model = torchvision.models.vgg16()
        model.classifier[-1] = nn.Linear(4096, num_outputs)
    else:
        raise ValueError("Unsupported model: {}".format(args.model))

    model = model.to(device)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'scinol2dl':
        optimizer = scinol.Scinol2Dl(model.parameters())
    elif args.optimizer == 'sinh':
        optimizer = sinh.Sinh(model.parameters())
    elif args.optimizer == 'nsinh':
        optimizer = sinh.NormSinh(model.parameters())
    else:
        raise ValueError("Unsupported optimizer: {}".format(args.optimizer))

    full_run_tag = strftime("%m.%d_%H:%M:%S") + "_" + args.optimizer
    if args.tag is not None:
        full_run_tag += "_" + args.tag
    tb_train_writer = tf.summary.create_file_writer("./tb_logs/{}_train".format(full_run_tag))
    tb_train_avg_writer = tf.summary.create_file_writer("./tb_logs/{}_train_avg".format(full_run_tag))
    tb_test_writer = tf.summary.create_file_writer("./tb_logs/{}_test".format(full_run_tag))

    train_data_size = len(train_loader.dataset)
    num_train_batches = int(np.ceil(train_data_size / train_loader.batch_size))

    summary_prefix = "{}/{}".format(args.dataset, args.model)


    def train(epoch):
        model.train()
        average_loss = 0
        all_correct = 0

        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc="Epoch {}".format(epoch),
                                              total=num_train_batches):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            all_correct += correct
            with tb_train_writer.as_default():
                step = (epoch - 1) * num_train_batches + batch_idx + 1
                tf.summary.scalar("{}/accuracy".format(summary_prefix), correct / len(data), step=step)
                tf.summary.scalar("{}/loss".format(summary_prefix), loss.item() / len(data), step=step)

        average_loss /= train_data_size
        average_acc = all_correct / train_data_size

        with tb_train_avg_writer.as_default():
            step = epoch * num_train_batches
            tf.summary.scalar("{}/accuracy".format(summary_prefix), average_acc, step=step)
            tf.summary.scalar("{}/loss".format(summary_prefix), average_loss, step=step)
        print('Train: loss: {:.3f}, acc: {}/{} ({:.3f})'.format(
            average_loss, all_correct, train_data_size, average_acc,
        ), file=sys.stderr)

        return average_loss, average_acc


    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        test_data_size = len(test_loader.dataset)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= test_data_size
        test_acc = correct / test_data_size
        with tb_test_writer.as_default():
            step = epoch * num_train_batches
            tf.summary.scalar("{}/accuracy".format(summary_prefix), test_acc, step=step)
            tf.summary.scalar("{}/loss".format(summary_prefix), test_loss, step=step)

        print('Test: loss: {:.3f}, acc: {}/{} ({:.3f})'.format(
            test_loss, correct, test_data_size,
            test_acc), file=sys.stderr)
        return test_loss, test_acc


    test(epoch=0)
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    # prefix = "scinol_"+"eta01"
    # print(prefix+",trainl,"+",".join(["{:0.4f}".format(x) for x in train_losses]))
    # print(prefix+",traina,"+",".join(["{:0.4f}".format(x) for x in train_accs]))
    # print(prefix+",testl,"+",".join(["{:0.4f}".format(x) for x in test_losses]))
    # print(prefix+",testa,"+",".join(["{:0.4f}".format(x) for x in test_accs]))
