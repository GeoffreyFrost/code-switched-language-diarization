from typing import OrderedDict
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer
import logging
from utils.transforms import interp_targets
from models.WavLM import WavLM, WavLMConfig
log = logging.getLogger(__name__)

def multiplicative(epoch: int) -> float:
    return 2.0

class LitCSDetector(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-4, label_smoothing=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.label_smoothing = label_smoothing
        assert backbone in ["base","large", "xlsr"], f'model: {backbone} not supported.'

        if backbone == "base":
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.backbone = bundle.get_model()
            self.head = nn.Linear(768, 2)

        if backbone == "large":
            bundle = torchaudio.pipelines.WAV2VEC2_LARGE
            self.backbone = bundle.get_model()
            self.head = nn.Linear(1024, 2)

        if backbone == "xlsr":
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
            self.backbone = bundle.get_model()
            self.head = nn.Linear(1024, 2)

            #self.head = nn.Sequential(OrderedDict([('linear1', nn.Linear(768, 4096)), ('linear2', nn.Linear(4096, 2))]))

        # if backbone == "wavlm":
        #     checkpoint = torch.load('/home/geoff/Documents/penguin/models/WavLM-Base.pt')
        #     cfg = WavLMConfig(checkpoint['cfg'])
        #     self.feature_extractor = WavLM(cfg)
        #     self.feature_extractor.load_state_dict(checkpoint['model'])
            
        #     self.head = nn.Sequential(OrderedDict([('linear1', nn.Linear(768, 4096)), ('linear2', nn.Linear(4096, 2))]))

    def forward(self, x, l):
        x, lengths = self.backbone(x, l)
        x = self.head(x)
        return x, lengths

    def training_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths, self.label_smoothing)
        self.log('train/loss', loss)
        return {'loss':loss, 'y_hat':y_hat, 'y':y, 'lengths':lengths}
    
    def training_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, 2)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float( y.size(0))
        self.log("train/train_acc", accuracy, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, x_l, y, y_l = batch
        y_hat, lengths = self.forward(x, x_l)
        loss, y = aggregate_bce_loss(y_hat, y, lengths, self.label_smoothing)
        self.log('val/loss', loss)
        return {'y_hat':y_hat, 'y':y, 'lengths':lengths}

    def validation_epoch_end(self, out):
        y_hat = torch.cat([x['y_hat'].view(-1, 2)[get_unpadded_idxs(x['lengths'])] for x in out])
        y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
        accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float( y.size(0))
        self.log("val/val_acc", accuracy, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), 
                                    lr=self.hparams.learning_rate, 
                                    weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler}

    # def freeze_backbone(self):
    #     for param in self.wav2vec2.parameters():
    #         param.requires_grad = False

    # def freeze_feature_extractor(self):
    #     for param in self.wav2vec2.feature_extractor.parameters():
    #         param.requires_grad = False

    # def unfreeze_backbone(self, freeze_feature_extractor=False):
    #     for param in self.wav2vec2.parameters():
    #         param.requires_grad = True
    #     if freeze_feature_extractor: self.freeze_feature_extractor()

def get_unpadded_idxs(lengths):
    max_len = torch.max(lengths)
    return torch.cat([torch.arange(max_len*i, max_len*i+l) for i, l in enumerate(lengths)]).to(torch.long)

def aggregate_bce_loss(y_hat, y, lengths, label_smoothing=0.0):
    eps = 1e-10
    y = interp_targets(y, torch.max(lengths))
    idxs = get_unpadded_idxs(lengths)
    loss = F.cross_entropy(y_hat.view(-1, 2)[idxs] + eps, y.view(-1)[idxs], label_smoothing=label_smoothing)
    return loss, y

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
     def __init__(self, unfreeze_at_epoch=10):
         super().__init__()
         self._unfreeze_at_epoch = unfreeze_at_epoch

     def freeze_before_training(self, pl_module):
        # freeze any module you want
         # Here, we are freezing `feature_extractor`
         self.freeze(pl_module.backbone)

     def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
         # When `current_epoch` is 10, feature_extractor will start training.
         if current_epoch == self._unfreeze_at_epoch:
             self.unfreeze_and_add_param_group(
                 modules=pl_module.backbone,
                 optimizer=optimizer,
                 train_bn=True,
             )
