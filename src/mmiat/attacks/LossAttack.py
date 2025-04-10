from mmiat.attacks.MembershipInferenceAttack import MembershipInferenceAttack
from mmiat.utils.data.AttackPredictionDataset import AttackPredictionDataset

import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from typing import Optional

class LossAttack(MembershipInferenceAttack):
    """
    Loss Attack implementation. (TODO: Add reference)
    
    Attributes:
        device (str): The device on which the computations will be performed (e.g., "cpu" or "cuda").
        batch_size (int): The batch size used for processing the dataset.
        criterion (torch.nn.Module): The loss function used to compute the loss values. Defaults to CrossEntropyLoss.
    """
    def __init__(self, device="cpu", batch_size: Optional[int]=8, criterion=None):
        self.device = device
        self.batch_size = batch_size
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss(reduction="none")

    def attack(self, model: nn.Module, attackprediction_dataset: AttackPredictionDataset):
        model.to(self.device)
        model.eval()
        
        membership_labels = []
        confidences = []
        
        with torch.no_grad():
            dataloader = tqdm(DataLoader(attackprediction_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=False),
                                desc="Loss Attack")
            for (x, y), member_label in dataloader:
                x, y = x.to(self.device), y.to(self.device)
            
                pred = model(x)
                loss = self.criterion(pred, y)
                confidence = -loss.detach().cpu()
        
                membership_labels.append(member_label.cpu())
                confidences.append(confidence.cpu())
        membership_labels = torch.cat(membership_labels).numpy()
        confidences = torch.cat(confidences).numpy()
        return membership_labels, confidences