from typing import Tuple, Union
from torch.utils.data import Dataset, random_split


def torch_iid_partitioning(
        dataset: Dataset, 
        seen_size: Union[int, float]) -> Tuple[Dataset, Dataset]:
    """
    Partitions a given dataset into two subsets: seen and unseen, based on the specified size using uniform random sampling.

    Args:
        dataset (Dataset): The dataset to be partitioned.
        seen_size Union[int, float]: The size of the seen dataset. If an integer is provided, it represents the 
                                    number of samples in the seen dataset. If a float is provided, it represents 
                                    the fraction of the dataset to be included in the seen dataset.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the train and attack datasets.
    """
    if isinstance(seen_size, int):
        unseen_size: int = len(dataset) - seen_size
    else:
        unseen_size: float = 1 - seen_size
    train_dataset, attack_dataset = random_split(
        dataset, [seen_size, unseen_size])
    return train_dataset, attack_dataset