import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Shapes(object):

    def __init__(self, dataset_zip=None):
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            self.dataset_zip = np.load(loc, encoding='latin1')
        else:
            self.dataset_zip = dataset_zip
        self.imgs = torch.from_numpy(self.dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64), None
        return x


class Dataset(object):
    def __init__(self, loc):
        self.dataset = torch.load(loc).float().div(255)

    def __len__(self):
        return self.dataset.size(0), None

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        return self.dataset[index], None


class Faces(Dataset):
    LOC = 'data/basel_face_renders.pth'

    def __init__(self):
        return super(Faces, self).__init__(self.LOC)


class CelebA(torchvision.datasets.ImageFolder):
    celeba_dirname  = '/scratch/gobi1/datasets/celeb-a'
    eval_partition_filename = \
            '/scratch/gobi1/datasets/celeb-a/list_eval_partition.csv'
    attr_filename = '/scratch/gobi1/datasets/celeb-a/list_attr_celeba.csv'
    def __init__(self, train=True, mode='train', **kwargs):
        from datetime import datetime
        import pandas as pd
        from collections import OrderedDict
        from torchvision.transforms.functional import crop
        t = datetime.now()
        print('loading CelebA data')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda p: crop(p, 40, 15, 148, 148)),
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor()])
        super(CelebA, self).__init__(self.celeba_dirname, transform=transform,
                **kwargs)
        print('done; it took {} secs'.format(datetime.now() - t))
        self.eval_partition = pd.read_csv(self.eval_partition_filename)
        attr_csv = pd.read_csv(self.attr_filename)
        attr_csv = attr_csv * (attr_csv > 0)  # {-1, 1} -> {0, 1}
        vals = attr_csv.values[:, 1:].astype(np.int_)  # attr values only
        self.attr = [torch.tensor(val, dtype=torch.long) for val in vals]
        # this might come in handy for adding axis labels to plots
        self.idx_to_attr = \
                OrderedDict({i: attr_csv.columns[i+1] for i in range(40)})

        self.train = train or mode == 'test_train'

        self.train_range = self._get_range(self.eval_partition, 0)
        self.valid_range = self._get_range(self.eval_partition, 1)
        self.test_range = self._get_range(self.eval_partition, 2)
        self.test_train_range = self._get_range(self.eval_partition, 3)
        self.test_valid_range = self._get_range(self.eval_partition, 4)
        self.test_test_range = self._get_range(self.eval_partition, 5)

        if mode == 'train':
            self.samples = self.samples[self.train_range[0]:self.train_range[1]]
            self.attr = self.attr[self.train_range[0]:self.train_range[1]]
        elif mode == 'validation':
            self.samples = self.samples[self.valid_range[0]:self.valid_range[1]]
            self.attr = self.attr[self.valid_range[0]:self.valid_range[1]]
        elif mode == 'test':
            self.samples = self.samples[self.test_range[0]:self.test_range[1]]
            self.attr = self.attr[self.test_range[0]:self.test_range[1]]
        elif mode == 'test_train':
            self.samples = self.samples[self.test_train_range[0]:self.test_train_range[1]]
            self.attr = self.attr[self.test_train_range[0]:self.test_train_range[1]]
        elif mode == 'test_validation':
            self.samples = self.samples[self.test_valid_range[0]:self.test_valid_range[1]]
            self.attr = self.attr[self.test_valid_range[0]:self.test_valid_range[1]]
        elif mode == 'test_test':
            self.samples = self.samples[self.test_test_range[0]:self.test_test_range[1]]
            self.attr = self.attr[self.test_test_range[0]:self.test_test_range[1]]
        else:
            raise ValueError('bad mode')
        self.mode = mode

    def __getitem__(self, index):
        img, fake_label = super(CelebA, self).__getitem__(index)
        attr = self.attr.__getitem__(index)
        return img, attr

    @staticmethod
    def _get_range(df, partition_val):  
        """
        partition val in {0, 1, 2} represents {train, valid, test}
        partition val in {3, 4, 5} split the test set into a 80/10/10 train/valid/test set
        
        """
        if partition_val in [0, 1, 2]:
            min_idx = df['partition'][df['partition'] == partition_val].index.min()
            max_idx = df['partition'][df['partition'] == partition_val].index.max()
            return min_idx, max_idx
        elif partition_val == 3:
            return [182639, 198607]
        elif partition_val == 4:
            return [198608, 200603]
        elif partition_val == 5:
            return [200604, 202600]
        else: 
            raise ValueError('bad partition val')

    def _attrs_to_str(self, list_of_int_attrs):
        s = ''
        for i in range(40):
            s += '\n{} = {}'.format(self.idx_to_attr[i], list_of_int_attrs[i])
        return s
    
    @property
    def ndim(self):
        return 64*64
