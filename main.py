
import argparse
from trainer import Trainer, ExperimentConfig, TrainerConfig
from models.lit_cs_detector import ModelConfig
from setup import create_dfs, check_wavlm_checkpoints


def set_config(config, args):
    arg_fields = list(vars(args).keys())
    for field in config.__dataclass_fields__:
        if field in arg_fields: setattr(config, field, vars(args)[field])
    return config

def set_configs(args):
    model_config = set_config(ModelConfig(), args)
    trainer_config = set_config(TrainerConfig(), args)
    experiment_config = set_config(ExperimentConfig(), args)
    return model_config, trainer_config, experiment_config

def arg_paser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--accumulate-grad-batches', default=8, type=int)
    parser.add_argument('--max-epochs', default=16, type=int)
    parser.add_argument('--grad-clip-val', default=0.5, type=float)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--learning-rate', default=5e-5, type=float)
    parser.add_argument('--backbone-warmup', action='store_true')
    parser.add_argument('--unfreeze-at-epoch', default=1, type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.set_defaults(backbone_warmup=False)

    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--lr-warmup-steps', default=1000, type=int)
    parser.add_argument('--no-ssl-pretrain', default=False, action='store_true')
    parser.add_argument('--backbone', default='base')
    parser.add_argument('--specaugment', action='store_true')
    parser.add_argument('--freeze-feature-extractor', action='store_true')
    parser.add_argument('--soft-units', action='store_true')
    parser.add_argument('--fuzzy-cs-labels', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--audio-transforms', action='store_true')
    parser.add_argument('--weight-decay', default=None, type=float)
    parser.add_argument('--flatten-melspecs', default=False, action='store_true')

    parser.set_defaults(specaugment=False)
    parser.set_defaults(freeze_feature_extractor=False)
    parser.set_defaults(soft_units=False)
    parser.set_defaults(fuzzy_cs_labels=False)
    parser.set_defaults(custom_cross_entropy=False)
    parser.set_defaults(mixup=False)
    parser.set_defaults(audio_transforms=False)

    parser.add_argument('--final', default=False, action='store_true')
    parser.add_argument('--eng-other', default=False, action='store_true')
    parser.add_argument('--no-mono-eng', default=False, action='store_true')
    parser.add_argument('--lang-fams', default=False, action='store_true')
    parser.add_argument('--pretrained-eng-other', default=False, action='store_true')
    parser.add_argument('--pretrained-lang-fams', default=False, action='store_true')
    parser.add_argument('--filter-cs', default=False, action='store_true')
    parser.add_argument('--baseline', default=None)
    parser.add_argument('--cs-pair', default='all')
    parser.add_argument('--dataset_path', default='/home/gfrost/datasets')

    return parser

if __name__ == '__main__':

    # check if wavlm weights are downloaded
    check_wavlm_checkpoints()
    
    parser = arg_paser()
    args = parser.parse_args()
    # Parse arguements to config objects
    model_config, trainer_config, experiment_config = set_configs(args)
    # Do setup check
    create_dfs(args.dataset_path)
    # Train
    trainer = Trainer(model_config, trainer_config, experiment_config)
    trainer.run_experiment()