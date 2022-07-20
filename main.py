
import argparse
from pyexpat import model
from trainer import Trainer, ExperimentConfig, TrainerConfig
from models.lit_cs_detector import ModelConfig
from setup import create_dfs

def set_configs(args):
    model_config = ModelConfig()
    trainer_config = TrainerConfig()
    experiment_config = ExperimentConfig()

    model_config.label_smoothing=args.label_smoothing
    model_config.backbone=args.backbone
    model_config.specaugment=args.specaugment
    model_config.freeze_feature_extractor=args.freeze_feature_extractor

    trainer_config.batch_size=args.batch_size
    trainer_config.accumulate_grad_batches=args.accumulate_grad_batches
    trainer_config.max_epochs=args.max_epochs
    trainer_config.grad_clip_val=args.grad_clip_val
    trainer_config.precision=args.precision
    trainer_config.learning_rate=args.learning_rate
    trainer_config.backbone_warmup=args.backbone_warmup
    trainer_config.unfreeze_at_epoch=args.unfreeze_at_epoch

    experiment_config.cs_pair=args.cs_pair
    experiment_config.routine=args.routine
    experiment_config.n_refinement_stages=args.n_refinement_stages
    experiment_config.unlabeled_ratio=args.unlabeled_ratio
    
    return model_config, trainer_config, experiment_config

def arg_paser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--accumulate-grad-batches', default=8, type=int)
    parser.add_argument('--max-epochs', default=16, type=int)
    parser.add_argument('--grad-clip-val', default=0.5, type=float)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--learning-rate', default=5e-5, type=float)
    parser.add_argument('--backbone-warmup', default=False, type=bool)
    parser.add_argument('--unfreeze-at-epoch', default=1, type=int)

    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--backbone', default='base')
    parser.add_argument('--specaugment', default=False, type=bool)
    parser.add_argument('--freeze-feature-extractor', default=True, type=bool)

    parser.add_argument('--cs-pair', default='all')
    parser.add_argument('--routine', default='semi-supervised')
    parser.add_argument('--n-refinement-stages', default=5, type=int)
    parser.add_argument('--unlabeled-ratio', default=0.2, type=float)

    parser.add_argument('--dataset_path', default='/home/gfrost/datasets')
    return parser

if __name__ == '__main__':
    parser = arg_paser()
    args = parser.parse_args()
    # Parse arguements to config objects
    model_config, trainer_config, experiment_config = set_configs(args)
    # Do setup check
    create_dfs(args.dataset_path)
    # Train
    trainer = Trainer(model_config, trainer_config, experiment_config)
    trainer.run_experiment()