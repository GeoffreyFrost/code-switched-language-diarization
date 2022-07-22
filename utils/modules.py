import csv
import os
import numpy as np
import shutil
def config_logger(model_config, trainer_config, experiment_config, log_dir='logs/configs/'):

    check_config_overlap()
    log_dir, latest_exp = setup_config_dir(log_dir)
    config_csv_writer(model_config, log_dir, 'model_config')
    config_csv_writer(trainer_config, log_dir, 'trainer_config')
    config_csv_writer(experiment_config, log_dir, 'experiment_config')
    

def check_config_overlap(path1='logs/lightning_logs/', path2='logs/configs/'):

    exp_dirs = os.listdir(path1)
    config_dirs = os.listdir(path2)
    exp_vers = set(strip_version_number(exp_dirs))
    config_vers = set(strip_version_number(config_dirs))
    if len(config_vers)==0 or len(exp_vers)==0: return
    configs_to_delete = config_vers - exp_vers
    delete_indices = np.where(np.in1d(list(config_vers), list(configs_to_delete)))[0]
    for i in delete_indices: shutil.rmtree(os.path.join(path2, config_dirs[i]))

def strip_version_number(versions):
    if len(versions) == 0: return []
    return [eval(version.split('_')[-1]) for version in versions]

def setup_config_dir(log_dir, path='logs/lightning_logs/'):
    exp_dirs = os.listdir(path)
    latest_exp = 0
    for exp_dir in exp_dirs: 
        if eval(exp_dir.split('_')[-1]) > latest_exp:  latest_exp = eval(exp_dir.split('_')[-1])
    latest_exp+=1
    log_dir = os.path.join(log_dir, f'configs_version_{latest_exp}')
    if os.path.exists(log_dir): return log_dir, latest_exp
    os.makedirs(log_dir)
    return log_dir, latest_exp

def config_csv_writer(config, log_dir, config_name):
    f = open(f'{log_dir}/{config_name}.csv', 'w')
    writer = csv.writer(f)
    writer.writerow([f"---{config_name}---"])
    for field in config.__dataclass_fields__: 
        value = getattr(config, field)
        writer.writerow([field, value])
        