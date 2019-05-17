import torch
import numpy as np
import os
import sys

from tqdm import tqdm_notebook
from PIL import Image
from urllib import request
from torch.utils.data import Dataset
sys.path.append("../semi-supervised")
n_labels = 10
cuda = torch.cuda.is_available()


class SpriteDataset(Dataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    * color
    * shape
    * scale
    * orientation
    * x-position
    * y-position
    """
    def __init__(self, transform=None):
        self.transform = transform
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        
        try:
            self.dset = np.load("./dsprites.npz", encoding="bytes")["imgs"]
        except FileNotFoundError:
            request.urlretrieve(url, "./dsprites.npz")
            self.dset = np.load("./dsprites.npz", encoding="bytes")["imgs"]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        sample = self.dset[idx]
                
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def get_mnist(location="./", batch_size=64, labels_per_class=100, preprocess=True):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    from utils import onehot
    

    
    mnist_train = MNIST(location, train=True, download=True)
    numpy_mnist = mnist_train.data.numpy().reshape(-1, 784) / 255
    mnist_mean = numpy_mnist.mean(0)
    mnist_std = numpy_mnist.std(0)

    mnist_indices = np.where(mnist_std > 0.1)


    if preprocess:
        flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1)[mnist_indices].bernoulli()
    else:
        flatten_bernoulli = lambda x: transforms.ToTensor()(x).view(-1)

    mnist_train = MNIST(location, train=True, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True,
                        transform=flatten_bernoulli, target_transform=onehot(n_labels))

    
    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(mnist_train.targets.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=2, pin_memory=cuda)
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda)

    return labelled, unlabelled, validation, mnist_mean, mnist_std

def get_svhn(location="./", batch_size=64, labels_per_class=1000, extra=True):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import SVHN
    import torchvision.transforms as transforms
    from utils import onehot

    std = np.array([0.19653187, 0.19832356, 0.19942404]) # precalulated
    std = std.reshape(3, 1, 1)
    std = torch.tensor(std).float()
    
    def flatten(x):
        x = transforms.ToTensor()(x)
        # x += torch.rand(3, 32, 32) / 255.
        # x /= std
        return x.view(-1)
    
    
    svhn_train = SVHN(location, split="train", download=True,
                        transform=flatten, target_transform=onehot(n_labels))
    
    if extra:
        svhn_extra = SVHN(location, split="extra", download=True,
                            transform=flatten, target_transform=onehot(n_labels))
        svhn_train = torch.utils.data.ConcatDataset([svhn_train, svhn_extra])
        
    print("Len of svhn train", len(svhn_train))
    svhn_valid = SVHN(location, split="test", download=True,
                        transform=flatten, target_transform=onehot(n_labels))

    
    def uniform_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler
    
    def get_sampler(dataset_len, n=None):
        # Only choose digits in n_labels
        # (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        # np.random.shuffle(indices)
        # indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        if n is None:
            indices = np.arange(dataset_len)
        else:
            indices = np.random.choice(dataset_len, size=n * 10, replace=False)
        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(len(svhn_train), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(svhn_train, batch_size=batch_size, num_workers=2, pin_memory=cuda)
    validation = torch.utils.data.DataLoader(svhn_valid, batch_size=batch_size, num_workers=2, pin_memory=cuda)
    return labelled, unlabelled, validation, std

def get_celeba(location="./", batch_size=64, labels_per_class=100, examples_num=None):
    from torch.utils.data.sampler import SubsetRandomSampler
    n_labels = 4

    celeba_train, celeba_test, label_names, _ = get_celeba_singletag(examples_num)

    celeba_train = torch.utils.data.TensorDataset(*celeba_train)
    celeba_test = torch.utils.data.TensorDataset(*celeba_test)

    def get_sampler(dataset_len, n=None):
        # Only choose digits in n_labels
        # (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        # np.random.shuffle(indices)
        # indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        if n is None:
            indices = np.arange(dataset_len)
        else:
            indices = np.random.choice(dataset_len, size=n * n_labels, replace=False)
        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(celeba_train, batch_size=batch_size, num_workers=2, pin_memory=cuda,
                                           sampler=get_sampler(len(celeba_train), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(celeba_train, batch_size=batch_size, num_workers=2, pin_memory=cuda)
    validation = torch.utils.data.DataLoader(celeba_test, batch_size=batch_size, num_workers=2, pin_memory=cuda)


    return labelled, unlabelled, validation, label_names
    



def get_celeba_images(examples_num):

    dataset_dir = "/mnt/users/mwolczyk/local/Repos/networks-do-networks/dataset/img_align_celeba/"
    orig_size = [178, 218]
    crop_size = [140, 140]
    target_size = [64, 64]

    start_y = (orig_size[1] - crop_size[0]) // 2
    start_x = (orig_size[0] - crop_size[1]) // 2

    loaded_images = []

    train = []
    valid = []
    test = []

    images_list = sorted(os.listdir(dataset_dir))
    for idx, img_name in enumerate(tqdm_notebook(images_list)):
        if examples_num is not None and idx >= examples_num:
            break
        img = Image.open(dataset_dir + img_name).convert("RGB")
        img = img.crop((
            start_x,
            start_y,
            start_x + crop_size[0],
            start_y + crop_size[1]
        ))
        img = np.array(img.resize(target_size, Image.BILINEAR)) / 255
        img = img.transpose(2, 0, 1)
        if idx < 162770:
            train += [img]
        elif idx < 182637:
            valid += [img]
        else:
            test += [img]

    return (torch.tensor(train, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32),
            torch.tensor(test, dtype=torch.float32))


def get_celeba_singletag(examples_num):
    if examples_num is None:
        examples_num = 200000
    attr_labels = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive",
        "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose",
        "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows",
        "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
        "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open",
        "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin",
        "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns",
        "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
        "Young"
    ]
    dataset_dir = "/mnt/users/mwolczyk/local/Repos/networks-do-networks/dataset/"

    # chosen_attributes = ["Male", "Smiling"]
    # TODO: uwazaj na kolejnosc
    chosen_attributes = ["Male", "Smiling"]
    chosen_indices = [idx for idx, label in enumerate(attr_labels)
                      if label in chosen_attributes]

    # TODO: to trzeba madrzej
    classes_num = 4
    labels_names = ["F/NS", "F/S", "M/NS", "M/S"]
    # labels_names = ["Not smiling", "Smiling"]


    train_y = []
    valid_y = []
    test_y = []
    with open(dataset_dir + "/list_attr_celeba.txt") as f:
        f.readline() # Omitting header
        f.readline() # Omitting label list
        for line_idx, line in enumerate(f):
            if examples_num is not None and line_idx >= examples_num:
                break

            labels = line.split()[1:]  # skip filename in the first column

            label_val = 0
            for idx, attr_idx in enumerate(chosen_indices):
                label_val *= 2
                val = int(labels[attr_idx])
                if val == 1:
                    label_val += 1
                elif val == -1:
                    pass
                else:
                    raise ValueError("Ani jeden ani minus jeden: {}".format(label))

            one_hot_label = [0] * classes_num
            one_hot_label[label_val] = 1

            if line_idx < 162770:
                train_y += [one_hot_label]
            elif line_idx < 182637:
                valid_y += [one_hot_label]
            else:
                test_y += [one_hot_label]

    train_y = torch.tensor(train_y, dtype=torch.float32)
    valid_y = torch.tensor(valid_y, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32)

    train_x, valid_x, test_x = get_celeba_images(examples_num)

    # If the example has no representation, pick
    # Y[Y.sum(1) == 0, -1] = 1
    # print("Ratio of labels:", Y.sum(axis=0) / Y.shape[0])
    # print("Nonzero count", np.count_nonzero(Y.sum(1)))

    return (train_x, train_y), (test_x, test_y), labels_names, "celeba_singletag"
