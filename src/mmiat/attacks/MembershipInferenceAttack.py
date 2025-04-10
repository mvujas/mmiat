from abc import ABC, abstractmethod

from torch import nn

from mmiat.utils.data import AttackPredictionDataset

class MembershipInferenceAttack(ABC):
    """
    Abstract base class for implementing membership inference attacks.
    """
    @abstractmethod
    def attack(self, model: nn.Module, attackprediction_dataset: AttackPredictionDataset, **kwargs):
        """
        Method to mount a membership inference attack on a given model.

        Args:
            model (nn.Module): The target model to attack.
            attackprediction_dataset (AttackPredictionDataset): The dataset used for the attack, 
                containing members and non-members with their corresponding labels.
            **kwargs: Additional keyword arguments that may be required for specific attack implementations.
        """
        raise NotImplementedError()
