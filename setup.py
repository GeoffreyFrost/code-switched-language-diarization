import os
from numpy import save
from utils.datapreprocessor import SoapiePreprocessor
import gdown

def check_cs_pairs_df_exist(dataset_root_dir):
    cs_pairs = ['engzul', 'engxho', 'engsot', 'engtsn']
    does_not_exist = []
    for cs_pair in cs_pairs:
        cs_dir = os.path.join(dataset_root_dir, \
            f'soapies_balanced_corpora/cs_{cs_pair}_balanced/lang_targs_mult/')
        if not os.path.exists(os.path.join(cs_dir, f'cs_{cs_pair}_trn.pkl')): does_not_exist.append(cs_pair)
    return does_not_exist

def create_dfs(dataset_root_dir, override=False):
    does_not_exist = check_cs_pairs_df_exist(dataset_root_dir)
    if override==True: does_not_exist = ['engzul', 'engxho', 'engsot', 'engtsn']
    if len(does_not_exist): 
        print(f"Dataset dataframes do not exist for the following cs pairs: {does_not_exist}")
        create = input("Do you want to create them (this might take a while)? [y/N] ").lower()
        create = create == 'y' or create == 'yes'
        if create:
            for cs_pair in does_not_exist:
                soapie_processor = SoapiePreprocessor('cs_'+cs_pair, os.path.join(dataset_root_dir, 'soapies_balanced_corpora'))
                _, _, _ = soapie_processor.generate_split_dataframes(save_pkl=True)

def check_wavlm_checkpoints():
    wavlm_large = 'models/weights/WavLM-Large.pt'
    wavlm_base = 'models/weights/WavLM-Base.pt'
    
    if not os.path.isfile(wavlm_large): download_wavlm(wavlm_large)
    if not os.path.isfile(wavlm_base): download_wavlm(wavlm_base)

def download_wavlm(wavlm_path):

    if wavlm_path.split('/')[-1] == 'wavlm-base': 
        URL = 'https://drive.google.com/u/0/uc?id=1PlbT_9_B4F9BsD_ija84sUTVw7almNX8&export=download'
        gdown.download(URL, 'models/weights/WavLM-Base.pt', quiet=False)
    elif wavlm_path.split('/')[-1] == 'wavlm-large': 

        URL = 'https://drive.google.com/u/0/uc?id=1rMu6PQ9vz3qPz4oIm72JDuIr5AHIbCOb&export=download'
        gdown.download(URL, 'models/weights/WavLM-Large.pt', quiet=False)

def create_directories():
    if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
        os.makedirs(os.path.join(os.getcwd(), 'logs/results'))
        os.makedirs(os.path.join(os.getcwd(), 'logs/final'))
        os.makedirs(os.path.join(os.getcwd(), 'logs/configs'))

if __name__ == '__main__':
    create_dfs('/home/gfrost/datasets', override=True)
