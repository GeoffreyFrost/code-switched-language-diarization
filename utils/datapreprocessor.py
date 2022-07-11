import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import re

class SoapiePreprocessor():
    def __init__(self, lang_pair, corpora_dir="/media/geoff/datasets/soapies_balanced_corpora/"):

        assert lang_pair in ["cs_engzul","cs_engxho","cs_engsot","cs_engtsn"]
        
        self.lang_pair = lang_pair
        self.corpora_dir = corpora_dir
        self.audio_dir = corpora_dir + lang_pair + "_balanced/audio"
        self.balanced_dir = corpora_dir + lang_pair + "_balanced"
        self.lst_dir = self.balanced_dir + "/lists"
        self.lang_tgts_dir = self.balanced_dir + "/lang_targs_mult"

        self.cosntruct_utt_audio_path_map()

    def cosntruct_utt_audio_path_map(self):
        spkr_utt_paths = {}
        with open(os.path.join(self.audio_dir, "audio_info.txt")) as f:
            for line in f: 
                if re.search("Input File", line): 
                    utt_path = eval(line.split(':')[-1])
                    spkr_utt_paths[utt_path.split('/')[-1].split('.')[0]] = self.audio_dir + utt_path[1:]
        self.spkr_utt_paths = spkr_utt_paths

    def get_split_utts(self, split):
        assert split in ["trn", "dev", "tst"]
        spkr_utt_paths_split = {}
        with open(os.path.join(self.lst_dir,f"{split}.lst")) as f:
            for line in f: 
                utt_id = line.split('.')[0][1:]
                spkr_utt_paths_split[utt_id] = self.spkr_utt_paths[utt_id]
        return spkr_utt_paths_split

    def generate_split_dataframes(self, save_csv=False):

        datadict_trn = construct_datadict(self.get_split_utts('trn'), self.lang_tgts_dir)
        datadict_dev = construct_datadict(self.get_split_utts('dev'), self.lang_tgts_dir)
        datadict_tst = construct_datadict(self.get_split_utts('tst'), self.lang_tgts_dir)

        df_trn = pd.DataFrame.from_dict(datadict_trn)
        df_dev = pd.DataFrame.from_dict(datadict_dev)
        df_tst = pd.DataFrame.from_dict(datadict_tst)

        if save_csv:
            path = self.balanced_dir + '/lang_targs_mult/'
            if not os.path.exists(path): os.makedirs(path)
            df_trn.to_pickle(path+self.lang_pair+"_trn.pkl")
            df_dev.to_pickle(path+self.lang_pair+"_dev.pkl")
            df_tst.to_pickle(path+self.lang_pair+"_tst.pkl")

        return df_trn, df_dev, df_tst

def load_utt_tgts(sample_utt_id, lang_tgts_dir):
    spkr = sample_utt_id.split('_')[0]
    with open(os.path.join(lang_tgts_dir, f"{spkr}/{sample_utt_id}.txt")) as f:
        for line in f: lang_targs = line
    lang_targs = np.array(list(lang_targs)).astype(int).astype(np.uint8)
    return lang_targs

def construct_datadict(spkr_utt_paths, lang_tgts_dir):
    datadict = {'audio_fpath':[], 'tgts':[]}
    for spkr_utt in tqdm(spkr_utt_paths):
        datadict['audio_fpath'].append(spkr_utt_paths[spkr_utt])
        datadict['tgts'].append(load_utt_tgts(spkr_utt, lang_tgts_dir))
    return datadict