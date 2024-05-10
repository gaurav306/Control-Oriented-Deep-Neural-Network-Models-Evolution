import datetime as dt
import os, sys, glob
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
    
    configs['if_model_image']                   = 0
    configs['loss']                             = 6
    configs['training']['epochs']               = epoch
    configs['training']['EarlyStop_patience']   = 25
    configs['tl_case']['tl_type']               = 1

    configs['data']['save_results_dir']         = 'saved_results/All15_models/'
    
    configs['optimizer']                        = opt
    configs[str(configs['optimizer'])]['lr']    = lr
    configs['training']['batch_size']           = batch
    
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)


if __name__ == '__main__':

    start_script        = dt.datetime.now()
    data_config         = 'config_data.yaml'
    training_config     = 'config_training.yaml'
    
    SLURM_ARRAY_TASK_ID = int(sys.argv[1])
    flag = 0
    totalnumberoftimes = 5
    for i in range(1,totalnumberoftimes+1):
        for j in range(1, 1 + 4):
            flag = flag + 1
            if flag == SLURM_ARRAY_TASK_ID:
                run_num = i
                RUNTHIS = j
                break
    
    run_params = [512, 400, 0.001, '1e-3', 'Adam']
    batch, epoch, lr, lr_not, opt = run_params
    
    dt_ident = '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])

    print('Current run case: ', RUNTHIS)
    print('Current run number: ', run_num)

    # wait random time
    import time
    sleeptime = np.random.randint(1, 100)/10
    print('sleeping for %s seconds' % str(sleeptime))
    time.sleep(sleeptime)
    
    """
    modelts = [ 'pymodconn_1_0.yaml', # no exchange of states from enc, cross mha, no self MHA, all LSTMs
                'pymodconn_1_1.yaml', # exchange of states from enc, cross mha, no self MHA, all LSTMs
                'pymodconn_2.yaml',   # exchange of states from enc, both MHA, all LSTMs
                'pymodconn_2_1.yaml', # exchange of states from enc, both MHA, all biLSTM
                'pymodconn_3.yaml',   # exchange of states from enc, both MHA, all GRUs
                'pymodconn_4.yaml',   # exchange of states from enc, both MHA, all biGRUs
                ]
    modelts = [ 'pymodconn_5.yaml'   # exchange of states from enc, both MHA, biGRU, MORE RESNET
                ]
    modelts = [ 
                'pymodconn_1_2.yaml', # exchange of states from enc, cross mha, Enc self MHA only
                'pymodconn_1_3.yaml', # exchange of states from enc, cross mha, Dec self MHA only
                ]
    modelts = [ 
                'pymodconn_3_1.yaml', # exchange of states from enc, both MHA, bottom LSTMs, no top GRU
                #'pymodconn_3_2.yaml', # exchange of states from enc, both MHA, bottom GRUs, no top GRU
                'pymodconn_5_0.yaml', # exchange of states from enc, both MHA, biLSTM, no RESNET
                'pymodconn_6.yaml',   # exchange of states from enc, both MHA, biLSTM, no RESNET, no mhaGRN
                'pymodconn_4_1.yaml',   # exchange of states from enc, both MHA, all biGRUs, 4 head instead of 8
                ]
    modelts = [ #'build_model_e1d1', 
                #'build_model_e1d1_attn', 
                'build_model_e1d1_wFuture1_1',
                'build_model_e1d1_wFuture1_2',
                'build_model_e1d1_wFuture1_1_GRU',
                'build_model_e1d1_wFuture1_2_GRU',
                #'build_model_e1d1_wFuture2_AddNorm',
                #'build_model_e1d1_wFuture2_Concat',
                'build_model_e1d1_wFuture2_Concat_attn',
                #'build_model_e1d1_wFuture3_AddNorm',
                'build_model_e1d1_wFuture3_Concat',
                #'build_model_e1d1_wFuture4_AddNorm',
                #'build_model_e1d1_wFuture4_Concat',
                #'build_model_e1d1_wFuture5_AddNorm',
                #'build_model_e1d1_wFuture5_Concat',
                'pymodconn_1_0.yaml', # no exchange of states from enc, cross mha, no self MHA, all LSTMs
                'pymodconn_1_1.yaml', # exchange of states from enc, cross mha, no self MHA, all LSTMs
                'pymodconn_1_2.yaml', # exchange of states from enc, cross mha, Enc self MHA only, all LSTMs
                'pymodconn_1_3.yaml', # exchange of states from enc, cross mha, Dec self MHA only, all LSTMs
                'pymodconn_2.yaml',   # exchange of states from enc, both MHA, all LSTMs, 8 heads
                'pymodconn_2_1.yaml',   # exchange of states from enc, both MHA, all biLSTMs, 8 heads
                'pymodconn_3.yaml',   # exchange of states from enc, both MHA, all GRUs, 8 heads
                'pymodconn_3_1.yaml', # exchange of states from enc, both MHA, bottom LSTMs, no top GRU
                'pymodconn_3_2.yaml',   # exchange of states from enc, both MHA, bottom GRUs, no top GRU
                'pymodconn_4.yaml',   # exchange of states from enc, both MHA, biGRU, 8 HEAD
                'pymodconn_4_1.yaml',   # exchange of states from enc, both MHA, all biGRUs, 4 head instead of 8
                'pymodconn_5.yaml',   # exchange of states from enc, both MHA, biGRU, MORE RESNET in pymodconn_4.yaml
                'pymodconn_5_0.yaml', # exchange of states from enc, both MHA, biLSTM, no RESNET
                'pymodconn_6.yaml',   # exchange of states from enc, both MHA, biLSTM, no RESNET, no mhaGRN
                ]
    modelts = [ 'pymodconn_3_4h.yaml',   # exchange of states from enc, both MHA, all GRUs, 4 heads
                'pymodconn_2_4h.yaml',   # exchange of states from enc, both MHA, all LSTMs, 4 heads
                'pymodconn_2_1_4h.yaml',   # exchange of states from enc, both MHA, all biLSTMs, 4 heads
                ]
    modelts = [ 
                'pymodconn_5.yaml',   # exchange of states from enc, both MHA, GRU, MORE RESNET in pymodconn_4.yaml
                'pymodconn_5_0.yaml', # exchange of states from enc, both MHA, GRU, no RESNET
                'pymodconn_6.yaml',   # exchange of states from enc, both MHA, GRU, no RESNET, no mhaGRN
                ]                           # 4 with merge method 2
    modelts = [ 
                'pymodconn_1_0_no.yaml', # no exchange of states from enc, cross mha, no self MHA, all LSTMs, no merging of states
                'pymodconn_1_1_no.yaml', # exchange of states from enc, cross mha, no self MHA, all LSTMs, no merging of states
                ]                           # 4 with merge method 2
    """
    modelts = [ 
                'pymodconn_3_addnorm.yaml',   # exchange of states from enc, both MHA, all GRUs, 8 heads
                'pymodconn_4_addnorm.yaml',   # exchange of states from enc, both MHA, biGRU, 8 HEAD
                'pymodconn_1_1_GRU.yaml', # exchange of states from enc, cross mha, no self MHA, all GRUs
                'pymodconn_1_1_biGRU.yaml', # exchange of states from enc, cross mha, no self MHA, all GRUs
                ]              
    
    modelt = modelts[RUNTHIS-1]

    TL_model_path = ''
    allconfigs = [data_config, training_config, modelt]
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')

    configs_data['training']['run_number'] = run_num
    configs_data['model_num'] = 'pymodconn'
    
    ident = '%s_%s' % (modelt, str(run_num)) + dt_ident
    print()
    print(TL_model_path)
    print(ident)
    print()
    og_case(ident, configs_data, run_params, TL_model_path) 
    """"""
    
    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')