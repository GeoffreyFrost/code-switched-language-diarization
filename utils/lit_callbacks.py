
"""Modified callbacks for pl"""

from numpy import rec
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Any, Callable, Dict, Optional
from torch.optim.optimizer import Optimizer
import logging
from torch.nn import Module
import pytorch_lightning as pl

def multiplicative(epoch: int) -> float:
    return 2.0

log = logging.getLogger(__name__)

class BackboneFinetuning(BaseFinetuning):
    r"""Finetune a backbone model based on a learning rate user-defined scheduling.

    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Initial learning rate for the backbone.
            By default, we will use ``current_learning /  backbone_initial_ratio_lr``
        should_align: Whether to align with current learning rate when backbone learning
            reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial learning rate will
            ``current_learning_rate /  initial_denom_lr``.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone
        rounding: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }


    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        super().load_state_dict(state_dict)


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")


    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.backbone)


    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone.encoder,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
     def __init__(self, freeze_at_epoch=6, unfreeze_at_epoch=7):
         super().__init__()
         self._freeze_at_epoch = freeze_at_epoch
         self._unfreeze_at_epoch = unfreeze_at_epoch

     def freeze_before_training(self, pl_module):
        # freeze any module you want
        self.freeze(pl_module.soft_head)

     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, feature_extractor will start training.

        if current_epoch == self._freeze_at_epoch:
            for param in pl_module.backbone.parameters():
                param.requires_grad = False

            self.unfreeze_and_add_param_group(
                initial_denom_lr=1,
                modules=pl_module.soft_head,
                optimizer=optimizer,
                train_bn=True,
                )

        if current_epoch == self._unfreeze_at_epoch:
            for param in pl_module.backbone.parameters():
                param.requires_grad = True
            for param in pl_module.backbone.feature_extractor.parameters():
                param.requires_grad = False