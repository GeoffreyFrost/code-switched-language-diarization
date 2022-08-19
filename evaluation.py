import itertools
import matplotlib.pyplot as plt
import torchmetrics.functional as FM
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm



def eval_run(model, dataloader):
    preds = []
    lengths = []
    wavs = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, x_l, y, _ = batch
            x = x.to(device)
            x_l = x_l.to(device)
            y = interp_targets(y, x.size(1))
            y = rnn_util.pack_padded_sequence(y, x_l.cpu(), batch_first=True, enforce_sorted=False).data.to(x.device)
            x = rnn_util.pack_padded_sequence(x, x_l.cpu(), batch_first=True, enforce_sorted=False)
            
            y_hat, embeddings = model.forward(x)
            # y = interp_targets(y, torch.max(_lengths))
            
            preds.append(y_hat.detach().cpu())
            labels.append(y.detach().cpu())
            lengths.append(_lengths.detach().cpu())
            lengths.append(x_l.detach().cpu())

def plot_cm(cm, target_names, norm=True, save=False, path='cm'):
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
    if save: plt.savefig(path+".pdf", dpi=500)
    plt.show()

if __name__ == '__main__':
    pass