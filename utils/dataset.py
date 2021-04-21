import os

import torch
from torch.utils import data

from utils import util


def input_fn(file_names, batch_size, palette):
    dataset = Dataset(file_names, palette)

    batch_size = min(batch_size, len(dataset))
    num_worker = min(os.cpu_count() - 2, batch_size)
    loader = data.DataLoader(dataset, batch_size, True, num_workers=num_worker,
                             collate_fn=Dataset.collate_fn, drop_last=True)
    return loader, dataset


class Dataset(data.Dataset):

    def __init__(self, file_names, palette):
        self.palette = palette
        self.file_names = file_names

    def __getitem__(self, index):
        file_name = self.file_names[index]

        image = util.load_image(file_name)
        label = util.load_label(file_name)

        image, label = util.random_crop(image, label)
        image, label = util.random_augmentation(image, label)

        image = torch.from_numpy(image.transpose(2, 0, 1))

        one_hot = util.one_hot_it(label, self.palette)
        one_hot = torch.from_numpy(one_hot)

        return image, one_hot

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def collate_fn(batch):
        image, label = zip(*batch)
        return torch.stack(image, 0), torch.stack(label, 0)
