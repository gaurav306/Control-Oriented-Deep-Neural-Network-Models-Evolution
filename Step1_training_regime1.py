__author__ = "Gaurav Chaudhary"
__copyright__ = "Gaurav Chaudhary 2022"
__version__ = "1.0.0"
__license__ = "MIT"

##SBATCH --constraint="v100|a100"

import os
import datetime as dt
import sys, glob
from src.configs.configs_init import merge_configs
from src.data_processor import DataClass
from src.main_run import Main_Run_Class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymodconn import Model_Gen

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def og_case(ident, configs, runparams, model_path):
    batch, epoch, lr, lr_not, opt = runparams
    print('Current run case_batch-%s_epoch-%s_lr-%s_opt-%s' % (str(batch), str(epoch), str(lr_not), str(opt)))  
    
    configs['if_model_image']                   = 0
    configs['loss']                             = 6
    configs['training']['epochs']               = epoch
    configs['training']['EarlyStop_patience']   = 25
    configs['tl_case']['tl_type']               = 1

    configs['data']['save_results_dir']         = 'saved_results/All15_training/'
    
    configs['optimizer']                        = opt
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = batch
    
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)


if __name__ == '__main__':

    # wait random time
    import time
    sleeptime = np.random.randint(1, 100)/10
    print('sleeping for %s seconds' % str(sleeptime))
    time.sleep(sleeptime)

    start_script        = dt.datetime.now()
    data_config         = 'config_data.yaml'
    training_config     = 'config_training.yaml'
    
    SLURM_ARRAY_TASK_ID = int(sys.argv[1])
    flag = 0
    totalnumberoftimes = 10
    for i in range(1, totalnumberoftimes + 1):
        for j in range(1, 1 + 2): #<< change this to number of cases
            flag = flag + 1
            if flag == SLURM_ARRAY_TASK_ID:
                run_num = i
                RUNTHIS = j
                break
    
    dt_ident = '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])

    BATCH_SIZE = 512
    EPOCHS = 400

    if RUNTHIS == 1:
        model_config = 'pymodconn_2_1.yaml'
        TL_model_path = ''
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')

        configs_data['training']['run_number'] = run_num
        configs_data['training']['training_type'] = 3
        configs_data['training']['ifLRS'] = 0
        run_params = [BATCH_SIZE, EPOCHS, 0.0001, '1e-4', 'Adam']

        ident = 'biLSTM_' + 'Adam_1e-4_TR31-10_all15_%s' % str(run_num) + dt_ident
        print()
        print(TL_model_path)
        print(ident)
        print()
        og_case(ident, configs_data, run_params, TL_model_path) 

    if RUNTHIS == 2:
        model_config = 'config_model_RNN_MHA_CIT3.yaml'
        TL_model_path = ''
        allconfigs = [data_config, training_config, model_config]
        configs_data = merge_configs(allconfigs, rl_path='src/configs/')

        configs_data['training']['run_number'] = run_num
        configs_data['training']['training_type'] = 3
        configs_data['training']['ifLRS'] = 0
        run_params = [BATCH_SIZE, EPOCHS, 0.0001, '1e-4', 'Adam']

        ident = 'Adam_1e-4_TR31-10_all15_%s' % str(run_num) + dt_ident
        print()
        print(TL_model_path)
        print(ident)
        print()
        og_case(ident, configs_data, run_params, TL_model_path) 



    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')