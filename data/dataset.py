import torch
import torch.utils.data
import numpy as np
from datasets import interleave_datasets


class CMDC_AUDIO_DATASET(torch.utils.data.Dataset):
    def __init__(self, DATASET_PROCESSED, x_label: list):
        self.dataset = DATASET_PROCESSED
        self.x_label = x_label

    def __getitem__(self, index):
        return [np.asarray(self.dataset[index][tag], dtype=np.float32) for tag in self.x_label], \
            self.dataset[index]['label']

    def __len__(self):
        return len(self.dataset)


class Dataset_EATD(torch.utils.data.Dataset):
    def __init__(self, EATD_PROCESSED, set_type='train', num=3, indices=None):
        if indices is None:
            indices = [0, 1, 2]
        self.dataset = EATD_PROCESSED
        datasets = [self.dataset.shard(num, i) for i in range(num)]
        if set_type == 'train':
            self.dataset = interleave_datasets([datasets[i] for i in indices])
        else:
            self.dataset = datasets[indices[0]]

    def __getitem__(self, index):
        return self.dataset[index]['audio'], self.dataset[index]['label_eatd']

    def __len__(self):
        return len(self.dataset)


class EATD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample


class DAIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'data': self.data[idx], 'label': self.labels[idx]}
        return sample
