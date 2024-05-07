import os
from typing import Callable, List, Optional, Tuple
import numpy as np
from math import sqrt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.autonotebook import tqdm

class Trainer():

    def __init__(
        self,
        dataset: Dataset,
        device: Optional[str] = 'cuda',
        validation_split: Optional[float] = 0.1,
        shuffle_dataset: Optional[bool] = True,
        dataset_size: Optional[int] = None
    ) -> None:
        
        # set device
        if device == 'cuda' and torch.cuda.is_available():
            print("Using cuda")
            torch.set_default_device('cuda')
        else:
            print("Using cpu")
            torch.set_default_device('cpu')
        
        random_seed = 0
        if not dataset_size:
            dataset_size = len(dataset)

        # train/validation split
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # train/validation loaders
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.train_loader = DataLoader(dataset, sampler=train_sampler)
        self.validation_loader = DataLoader(dataset, sampler=valid_sampler)

    def train(
        self,
        model: torch.nn.Module,
        loss_function: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        map_property: Callable[[torch.Tensor], torch.Tensor],
        aggregate_outputs: Callable[[torch.Tensor], torch.Tensor],
        max_epochs: int = 100,
        models_dir: str = None,
        checkpoint_frequency: Optional[int] = 10,
        property: str = None,
    ) -> Tuple[List[int], List[int]]:
        losses_train_mean = []
        losses_val_mean = []

        for epoch in tqdm(range(max_epochs)):
            
            losses_train = []
            model.train()
            for i, (z, r, t) in enumerate(self.train_loader):
                optimizer.zero_grad()
                
                outputs = model(z, r)
                pred = aggregate_outputs(outputs)
                target = map_property(t)

                loss = loss_function(pred, target)
                loss.backward(retain_graph=True)

                optimizer.step()

                losses_train.append(loss.detach().cpu().numpy().item())
            losses_train_mean.append(np.mean(losses_train))
            

            losses_val = []
            model.eval()
            for i, (z, r, t) in enumerate(self.validation_loader):  

                with torch.no_grad():
                    outputs = model(z, r)
                pred = aggregate_outputs(outputs)
                target = map_property(t)

                loss = loss_function(pred, target)
                    
                losses_val.append(loss.detach().cpu().numpy().item())
            losses_val_mean.append(np.mean(losses_val))

            if models_dir and (epoch+1) % checkpoint_frequency == 0:
                if not os.path.isdir(models_dir):
                    os.makedirs(models_dir)
                torch.save(model.state_dict(), os.path.join(models_dir, f'{property}-epoch-{epoch+1}.pt'))
                scheduler.step()

        torch.cuda.empty_cache()

        return losses_train_mean, losses_val_mean
    
    def load_trained(
        self, 
        model: torch.nn.Module,
        models_dir: str,
        property: str, 
        epoch: int
    ) -> torch.nn.Module:
        state = torch.load(os.path.join(models_dir, f'{property}-epoch-{epoch}.pt'))
        model.load_state_dict(state)
        return model
