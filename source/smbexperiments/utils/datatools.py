import torchvision
import numpy as np
from typing import Optional
from torch.utils.data import RandomSampler, DataLoader, Subset
from smbexperiments.experimentlogger import get_logger

logger = get_logger(__name__)


def get_train_loader(n_samples: Optional[int], train_set, batch_size: int):
    if n_samples is not None:
        logger.info(f" Dataset filtering is active: training with {n_samples} samples and batch size {batch_size}")
        num_train_samples = n_samples
        sample_ds = Subset(train_set, np.arange(num_train_samples))
        sample_sampler = RandomSampler(sample_ds)
        return DataLoader(sample_ds,
                          sampler=sample_sampler,
                          batch_size=batch_size)

    else:
        return DataLoader(train_set,
                          drop_last=True,
                          shuffle=True,
                          batch_size=batch_size)


def get_dataset(dataset_name, batch_size, n_samples: Optional[int]):
    if dataset_name == "mnist":
        train_set = torchvision.datasets.MNIST("Datasets", train=True,
                                               download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.5,), (0.5,))
                                               ]))

        test_set = torchvision.datasets.MNIST("Datasets", train=False,
                                              download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.5,), (0.5,))
                                              ]))

        train_loader = get_train_loader(n_samples=n_samples,
                                        train_set=train_set,
                                        batch_size=batch_size,
                                        )

        return train_set, test_set, train_loader

    if dataset_name == "cifar10":
        transform_function = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                             torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                              (0.2023, 0.1994, 0.2010)),
                                                             ])

        train_set = torchvision.datasets.CIFAR10(root='Datasets',
                                                 train=True,
                                                 download=True,
                                                 transform=transform_function
                                                 )

        test_set = torchvision.datasets.CIFAR10(root="Datasets",
                                                train=False,
                                                download=True,
                                                transform=transform_function
                                                )

        train_loader = get_train_loader(n_samples=n_samples,
                                        train_set=train_set,
                                        batch_size=batch_size,
                                        )

        return train_set, test_set, train_loader

    if dataset_name == "cifar100":
        transform_function = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                                             torchvision.transforms.RandomHorizontalFlip(),
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                              (0.2023, 0.1994, 0.2010)),
                                                             ])

        train_set = torchvision.datasets.CIFAR100(
            root='Datasets',
            train=True,
            download=True,
            transform=transform_function)

        test_set = torchvision.datasets.CIFAR100(
            root='Datasets',
            train=False,
            download=True,
            transform=transform_function)

        train_loader = get_train_loader(n_samples=n_samples,
                                        train_set=train_set,
                                        batch_size=batch_size,
                                        )

        return train_set, test_set, train_loader
