__author__ = "Gaurav Chaudhary"
__copyright__ = "Gaurav Chaudhary 2022"
__version__ = "1.0.0"
__license__ = "MIT"
import datetime as dt
import os, sys
from src.configs.configs_init import merge_configs
from src.data_processor import DataClass
from src.main_run import Main_Run_Class

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def onecase(ident, configs, ifdata, model_path=None):
    ident = ident + str(dt.datetime.now().strftime('%d.%m-%H.%M.%f')[:-3])
    print('Current run case: ', ident)
    if ifdata:    
        print('Opening data class')
        data_class = DataClass(ident, configs)

    configs['Adam']['lr'] = 0.0001
    configs['training']['epochs'] = 50
    configs['if_model_image'] = 0

    print('Opening training pipeline class')
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)

if __name__ == '__main__':

    data_config = 'config_data.yaml'                #change data details here. 'src/configs/'
    training_config = 'config_training.yaml'

    #models = ['M0', 'M1', 'M2', 'M3low', 'M3med', 'M3high']
    models = ['M0']
    
    #cities    = ['Oslo', 'Bergen', 'Trondheim', 'Roros', 'Tromso']
    cities    = ['Steinkjer']

    # renames files
    for model in models:
        for city in cities:
            model_config = 'config_model_RNN_CIT3.yaml'  #can be any
            allconfigs = [data_config, training_config, model_config]
            configs_data = merge_configs(allconfigs, rl_path='src/configs/')

            configs_data['data']['input_EP_csv_files'][0] = '%s_%s.csv' %(model, city)
            ident = 'TL_ARG_RNN_CIT2_'
            onecase(ident, configs_data, True)
    
