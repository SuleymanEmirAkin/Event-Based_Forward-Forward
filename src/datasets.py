import numpy as np
import torch
import cv2 as cv
import random
from src import utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        if "nmnist" == opt.input.dataset:
            self.dataset = utils.get_NMNIST_partition(opt, partition) 
        elif "mnist" in opt.input.dataset: 
            self.dataset = utils.get_MNIST_partition(opt, partition) 
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.dataset)

    def _generate_sample(self, index):
        sample, class_label = self.dataset[index]

        if self.opt.training.backpropagation:
            return sample, sample, sample, sample, class_label

        if self.opt.training.unsupervised:
            self.mask, self.inverse_mask = self.create_mask()
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        
        if self.opt.training.unsupervised:
            return pos_sample, neg_sample, sample, sample, class_label
        
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label
    
    def _get_pos_sample(self, sample, class_label):
        pos_sample = sample.clone()
        
        if not self.opt.training.unsupervised:
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(class_label), num_classes=self.num_classes
            )
            if self.opt.input.dataset == "nmnist":
                pos_sample[:, 0, 0, : self.num_classes] = one_hot_label
            else:
                pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        neg_sample = sample.clone()
        if not self.opt.training.unsupervised:
            # Create randomly sampled one-hot label.
            classes = list(range(self.num_classes))
            classes.remove(class_label)  # Remove true label from possible choices.
            wrong_class_label = np.random.choice(classes)
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(wrong_class_label), num_classes=self.num_classes
            )
            if self.opt.input.dataset == "nmnist":
                neg_sample[:, 0, 0, : self.num_classes] = one_hot_label
            else:
                neg_sample[0, 0, : self.num_classes] = one_hot_label
        else:
            while True:
                data, label = self.dataset[random.randint(0, len(self.dataset ) - 1)]
                # Check if the label is not 'x'
                if label != class_label:
                    neg_sample = neg_sample * self.mask + data * self.inverse_mask
                    break
        return neg_sample.float()

    def _get_neutral_sample(self, z):
        if self.opt.input.dataset == "nmnist":
            z[:, 0, 0, : self.num_classes] = self.uniform_label
        else:
            z[0, 0, : self.num_classes] = self.uniform_label
        return z
  
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes,) +  sample.shape)
        for i in range(self.num_classes):
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i] = sample.clone()
            if self.opt.input.dataset == "nmnist":
                all_samples[i, :, 0, 0, : self.num_classes] = one_hot_label
            else:
                all_samples[i, 0, 0, : self.num_classes] = one_hot_label
        return all_samples
    
    def create_mask(self):
        if self.opt.input.dataset == "nmnist":
            size = (8, 34, 34)
        elif self.opt.input.dataset == "mnist":
            size = (28, 28)
        else:
            raise Exception("Unknown dataset")
        
        random_bits = np.random.randint(2, size=size, dtype=np.uint8)

        random_image = random_bits * 255

        kernel = np.array([[ 0.0625, 0.125 , 0.0625],
                            [0.125 , 0.25  , 0.1250],
                            [0.0625, 0.125 , 0.0625]])

        for _ in range(0,10):
            random_image = cv.filter2D(random_image, -1, kernel)

        if self.opt.input.dataset == "nmnist":
            random_image = random_image[:,np.newaxis]
        return (random_image >= 127.5).astype(np.uint8), (random_image <= 127.5).astype(np.uint8)