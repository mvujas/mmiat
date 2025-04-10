import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

class AttackPredictionDataset(Dataset):
    """
    PyTorch Dataset that combines a training dataset and an non-training dataset 
    to create a new dataset for membership inference attacks. It assigns a membership label to each 
    sample, indicating whether the sample belongs to the training set (1) or the non-training set (0). 
    The combined dataset is shuffled to ensure randomness.
    
    Args:
        trainset (Dataset): The dataset representing the training data.
        nontrainset (Dataset): The dataset representing the non-training data.
    """
    def __init__(self, trainset, nontrainset):
        self.__data = ConcatDataset([trainset, nontrainset])
        self.__member_labels = torch.concat([torch.ones(len(trainset)), torch.zeros(len(nontrainset))]).long()
        shuffled_indices = torch.randperm(len(self.__data))
        self.__data = Subset(self.__data, shuffled_indices)
        self.__member_labels = self.__member_labels[shuffled_indices]
    
    @property
    def member_labels(self):
        return self.__member_labels
    
    def __len__(self):
        return len(self.__data)
    
    def __getitem__(self, idx):
        """
        Retrieve the data sample and its corresponding member label at the specified index.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - (x, y): The data sample, where `x` is the input data and `y` is the target label.
                - member_label: The membership label associated with the data sample. (1 for members, 0 for non-members)
        """
        x, y = self.__data[idx]
        member_label = self.__member_labels[idx]
        return (x, y), member_label