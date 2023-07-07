
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from composer.core import Algorithm, Event


def _restore(
        model: Module,
        backup: dict) -> None:
    for name, param in model.named_parameters():
        if name in backup:
            param.data = backup[name]
    
def _save(
        model: Module,
        adv_param: str,
        adv_eps: float,
        backup: dict,
        backup_eps: dict) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and adv_param in name:
            if name not in backup:
                backup[name] = param.data.clone()
                grad_eps = adv_eps * param.abs().detach()
                backup_eps[name] = (
                    backup[name] - grad_eps,
                    backup[name] + grad_eps,
                )

def _attack_step(
        model: Module,
        adv_param: str,
        adv_lr: float,
        backup_eps: dict) -> None:
    e = 1e-6
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and adv_param in name:
            norm1 = torch.norm(param.grad)
            norm2 = torch.norm(param.data.detach())
            if norm1 != 0 and not torch.isnan(norm1):
                r_at = adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                param.data.add_(r_at)
                param.data = torch.min(
                    torch.max(
                        param.data, backup_eps[name][0]), backup_eps[name][1]
                )

def _attack_backward(
        model: Module,
        optimizer: Optimizer,
        batch, # the current state.batch
        adv_param: str,
        adv_lr: float,
        adv_eps: float,
        backup: dict,
        backup_eps: dict,
        apex: bool) -> Tensor:
    with torch.cuda.amp.autocast(enabled=apex):
        _save(
            model,
            adv_param,
            adv_eps,
            backup,
            backup_eps,
        )
        _attack_step(
            model,
            adv_param,
            adv_lr,
            backup_eps,
        )
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = {k: v for k, v in batch.items() if k == 'labels'}['labels']
        y_preds = model(inputs)
        adv_loss = model.loss(
            y_preds, batch)
        mask = (labels.view(-1, 1) != -1)
        adv_loss = torch.masked_select(adv_loss, mask).mean()
        optimizer.zero_grad()
    return adv_loss


class AWP(Algorithm):
    def __init__(self, 
                 start_epoch: int, 
                 adv_param: str = 'weight',
                 adv_lr: float = 1.0,
                 adv_eps: float = 0.01,
                 apex: bool = True):
        self.start_epoch = start_epoch
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}
    
    def match(self, event, state):
        return event == Event.AFTER_TRAIN_BATCH and state.timestamp.epoch >= self.start_epoch
    
    def apply(self, event, state, logger):
        state.loss = _attack_backward(
            state.model, 
            state.optimizers[0], 
            state.batch,
            self.adv_param,
            self.adv_lr,
            self.adv_eps,
            self.backup,
            self.backup_eps,
            self.apex)
        state.scaler.scale(state.loss).backward()
        # state.loss.backward()
        _restore(state.model, self.backup)
        self.backup, self.backup_eps = {}, {}