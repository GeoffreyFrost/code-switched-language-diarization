import argparse
from models.lit_cs_detector import LitCSDetector, get_unpadded_idxs
from models.lit_blstm_e2e import LitBLSTME2E
from models.lit_x_transformer_e2e import LitXSAE2E
from trainer import get_checkpoint_path
from utils.datasets import load_test_dfs, create_test_dataloader
import pytorch_lightning as pl
import torchmetrics.functional as FM
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from scipy.integrate import simps
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
from matplotlib import pyplot as plt
import os
import itertools

def load_trained_model(model_name, diarization_config):
    ckpt_path = get_checkpoint_path(f'logs/final/{model_name}/lightning_logs/version_{diarization_config}')
    if model_name == 'xsa': model = LitXSAE2E.load_from_checkpoint(ckpt_path)
    elif model_name == 'blstm': model = LitBLSTME2E.load_from_checkpoint(ckpt_path)
    else: model = LitCSDetector.load_from_checkpoint(ckpt_path)
    return model

def load_test_df(diarization_config):
    dfs = []
    supported_cs_pairs = ["engzul", "engxho", "engtsn", "engsot"]
    for cs_pair in supported_cs_pairs:
        data_df_root_dir = os.path.join('/home/gfrost/datasets', \
            f"soapies_balanced_corpora/cs_{cs_pair}_balanced/lang_targs_mult/")
        dfs.append(load_test_dfs(data_df_root_dir, cs_pair,
                                    eng_other=True if diarization_config == 0 else False,
                                    lang_fams=True if diarization_config == 1 else False,
                                    all_cs_pairs=True if diarization_config == 2 else False,
                                    )
                        )
    return pd.concat(dfs)

def load_test_dataloader(df, model_name, bs=8):
    tst_dataloader = create_test_dataloader(df, bs=bs,                                      
                        flatten_melspecs=True if model_name == 'blstm' else False,
                        melspecs=True if model_name in ['blstm', 'xsa'] else False,
                        stack_frames=True if model_name == 'xsa' else False)
    return tst_dataloader

def eval_loop_ssl_models(model, tst_dataloader):

    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    out = trainer.predict(model, tst_dataloader)

    y_hat = torch.cat([x['y_hat'].view(-1, 5)[get_unpadded_idxs(x['lengths'])] for x in out])
    y = torch.cat([x['y'].view(-1)[get_unpadded_idxs(x['lengths'])]for x in out])
    # accuracy = (torch.softmax(y_hat, dim=-1).argmax(dim=-1) == y.argmax(dim=-1)).sum().float() / float(y.size(0))
    # print(accuracy)

    return y_hat, y, out

def eval_loop_blstm_models(model, tst_dataloader):

    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    out = trainer.predict(model, tst_dataloader)

    y_hat = torch.cat([x['y_hat'] for x in out])
    y = torch.cat([x['y'] for x in out])
    
    return y_hat, y, out

def compute_predictions(model_name, model, tst_dataloader):
    if model_name in ['blstm']: y_hat, y, out = eval_loop_blstm_models(model, tst_dataloader)
    else: y_hat, y, out = eval_loop_ssl_models(model, tst_dataloader)
    return y_hat, y, out

def get_predictions(model_name, diarization_config, model, tst_dataloader):

    path = f'logs/results/{model_name}_predictions_dcfg_{diarization_config}.pt'
    # if os.path.exists(path): 
    #     y_hat, y, out = torch.load(path)
    # else: 
    y_hat, y, out = compute_predictions(model_name, model, tst_dataloader)
    torch.save((y_hat, y, out), path)
    return y_hat, y, out

def compute_metrics(y_hat, y, out, num_classes):

    accuracy = (F.softmax(y_hat, dim=-1).argmax(dim=-1) == y).sum().float() / float(y.size(0))
    cm = np.array(FM.confusion_matrix(y_hat.argmax(dim=-1), y, num_classes=num_classes))

    return 1-accuracy, cm

def plot_cm(cm, model_name, diarization_config, norm=True, save=False):

    if diarization_config == 2: target_names = ["English", "Zulu", "Xhosa", "Sesotho", "Setswana"]
    if diarization_config == 1: target_names = ["English", "Nguni", "Sotho–Tswana"]
    if diarization_config == 0:target_names = ["English", "Banu"]

    if norm: cm = np.round(cm / np.sum(cm, axis=1)[:, np.newaxis], 4)
    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('English-Zulu confusion matrix (wav2vec2-base, 100% train)')
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save: plt.savefig(f"figs/cm_{model_name}_dc_{diarization_config}.svg", bbox_inches='tight', dpi=500)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--diarization-config', default=2, type=int) # 0: eng-other, 1: lang-fams, 2: all
    parser.add_argument('--save-cm', default=False, action='store_true')
    args = parser.parse_args()

    model_name = args.model
    diarization_config = args.diarization_config

    model = load_trained_model(model_name, diarization_config)
    df = load_test_df(diarization_config)
    tst_dataloader = load_test_dataloader(df, model_name)

    y_hat, y, out = get_predictions(model_name, diarization_config, model, tst_dataloader)

    if diarization_config == 0: num_classes = 2
    if diarization_config == 1: num_classes = 3
    if diarization_config == 2: num_classes = 5

    ER, cm = compute_metrics(y_hat, y, out, num_classes)
    plot_cm(cm, model_name, diarization_config, save=args.save_cm)

    print(f'Error Rate: {ER:.4f}')