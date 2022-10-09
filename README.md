# Fine-Tuned Self-Supervised Speech Representations for Language Diarization in Multilingual Code-Switched Speech

## About
TODO

## Getting Started

Download the corpora [soapies_balanced_corpora.tar.gz]() (TODO: add google drive link) and unzip it to a directory of your choice.

```bash
tar -xf  soapies_balanced_corpora.tar.gz -C /path/to/corpora
```

Set up your environment. This step is optional (the main dependencies are `PyTorch` and `Pytorch Lightning`), but you'll hit snags along the way, which may be a bother.

```bash
conda env create --name myenv -f enviornment.yml
conda activate myenv
```

## Training

All configs as presented in the paper can be found in `final_runs.txt`, which can be run sequentially with

```bash
python runs.py --cmds-path final_runs.txt
```

Altenertivly, `main.py` can be used to run any supported experiment configuration using a host of arguments.

### Arguments

A host of configuration arguments are offered in `main.py`, most of which are self-explanatory (see the script). Some of them are not relevant to the final paper but were used to explore fun ideas (soft units, data augmentation, and a few other random ideas). Some important ones:

- `--baseline`: Which baseline model to use ["blstm", "xsa"].
- `--backbone`: Which ssl transformer to use ["base","large", "xlsr", "wavlm-large",  "wavlm-base"]. "base","large" and "xlsr" all refer to the respective wav2vec2 models, although experiments with these were not presented.
- `--no-mono-eng`: Do not use any monolingual English utterances during training.
- `--pretrained-eng-other`: Initialise weights to a model trained on the English/Bantu diarization task.
- `--pretrained-lang-fams`: Initialise weights to a model trained on the language family dirazation task.
- `--pretrained-weights-path`: Path for a respective pre-trained model. **Important**, this is only necessary if the `--final` flag is *NOT* used.
- `--final`: Creates the file structure to run experiments smoothly (such as automatic path assignment for loading of pre-trained weights) and for `test.py` functionality.

Progress can be insepected with tensorboard, which includes per epoch development set performance. More in-depth development analysis and metrics are facilitated by the `devlopment_tinker.ipynb` notebook (found under the `notebooks/` directory).

## Final evaluation (testing)

It is time to test our developed models! For this to work, the following file structure is required (although modifications can be made to `test.py` to take absolute file paths)
```
final
|---blstm
|    ...
|---wavlm-large
    |---lightning_logs
        |---version_0 # English/other
        |   |---checkpoints
        |   |   |---*.ckpt
        |   |   ....
        |---version_1 # Language families
        |   |... 
        |---version_2 # All languages
            |... 
```
Run `test.py` with for a specific model and diarization config (0=English/other, 1=language families and 2=all languages)

```bash
python test.py --model wavlm-large --diarization-config 0 --save-cm 
```

Which should output the following and save the confusion matrix
```bash     
Global Error Rate: 0.1005
Mean Error Rate: 0.1194
```

## Citation
TODO