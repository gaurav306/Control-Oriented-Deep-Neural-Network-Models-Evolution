import datetime as dt
import sys
from src.configs.configs_init import merge_configs
from src.main_run import Main_Run_Class
import numpy as np

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

def og_case(ident, configs, model_path):
    
    configs['if_model_image']                   = 0
    configs['training']['epochs']               = 200
    configs['data']['save_results_dir']         = 'saved_results/All15_models/Indiruns/'
    configs['optimizer']                        = 'Adam'
    configs[str(configs['optimizer'])]['lr']    = 0.0001
    configs['training']['batch_size']           = 512
    
    print('Current run case: ', ident)
    run_pipeline = Main_Run_Class(ident, configs, model_path)
    print('%s finished' % ident)


if __name__ == '__main__':
    start_script        = dt.datetime.now()
    sleeptime = np.random.randint(1, 5000)/1000
    print('sleeping for %s seconds' % str(sleeptime))
    #time.sleep(sleeptime)
    temparr = []
    for i in range(1, 1 +15):
        temparr.append('OneOf15-%s' % str(i))
    possible_types = temparr + ['OnlyIATs', 'OnlyPOWERs', 'OnlyCO2s', 'OnlyZN-1', 'OnlyZN-2', 'OnlyZN-3', 'OnlyZN-4', 'OnlyZN-5']
    types_indexes = [
                    [0], # OneOf15-1
                    [1], # OneOf15-2
                    [2], # OneOf15-3
                    [3], # OneOf15-4
                    [4], # OneOf15-5
                    [5], # OneOf15-6
                    [6], # OneOf15-7
                    [7], # OneOf15-8
                    [8], # OneOf15-9
                    [9], # OneOf15-10
                    [10], # OneOf15-11
                    [11], # OneOf15-12
                    [12], # OneOf15-13
                    [13], # OneOf15-14
                    [14], # OneOf15-15
                    [0, 1, 2, 3, 4], # OnlyIATs
                    [5, 6, 7, 8, 9], # OnlyPOWERs
                    [10, 11, 12, 13, 14], # OnlyCO2s
                    [0, 5, 10], # OnlyZN1
                    [1, 6, 11], # OnlyZN2
                    [2, 7, 12], # OnlyZN3
                    [3, 8, 13], # OnlyZN4
                    [4, 9, 14] # OnlyZN5
                    ]

    SLURM_ARRAY_TASK_ID = int(sys.argv[1])
    flag = 0
    totalnumberoftimes = 3
    for i in range(1,totalnumberoftimes+1):
        for j in range(0, len(possible_types)):
            flag = flag + 1
            if flag == SLURM_ARRAY_TASK_ID:
                run_num = i
                RUNTHIS = j
                break
    
    #RUNTHIS = SLURM_ARRAY_TASK_ID-1  # comment it out if you want to run it on slurm
    dt_ident = '_' + str(dt.datetime.now().strftime('%d.%H.%f')[:-2])

    allconfigs = ['config_data.yaml', 'config_training.yaml', 'pymodconn_3.yaml']
    configs_data = merge_configs(allconfigs, rl_path='src/configs/')

    configs_data['training']['run_number'] = run_num
    configs_data['individual_runs']['type'] = possible_types[RUNTHIS]
    configs_data['individual_runs']['type_index'] = types_indexes[RUNTHIS]

    ident = '%s_%s' % (possible_types[RUNTHIS], str(run_num)) + dt_ident 
    print(ident)
    
    if 'OneOf15-' in possible_types[RUNTHIS]:
        variable_num = int(possible_types[RUNTHIS].split('-')[1])
        if variable_num in [1, 6, 11]:
            zone_num = 1
        elif variable_num in [2, 7, 12]:
            zone_num = 2
        elif variable_num in [3, 8, 13]:
            zone_num = 3
        elif variable_num in [4, 9, 14]:
            zone_num = 4
        elif variable_num in [5, 10, 15]:
            zone_num = 5
        
        # only one variable
        x_wea    = [0,  1,  2,  3,  4]
        x_occu   = [5,  6,  7,  8,  9]
        x_time   = [20, 21, 22, 23, 24, 25]
        x_window = [26 + zone_num-1]
        x_hvac   = [31 + zone_num-1]
        x_hsp    = [36 + zone_num-1]
        x_csp    = [41 + zone_num-1]
        all_y    = [46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71]
        y_variable = [all_y[variable_num-1]]

        configs_data['data_processor']['columnno_X']            = x_wea + x_occu + x_time + x_window + x_hvac + x_hsp + x_csp + y_variable
        configs_data['data_processor']['columnno_X_future']     = [0]            + x_time + x_window + x_hvac + x_hsp + x_csp
        configs_data['data_processor']['columnno_y']                                                                          = y_variable
        configs_data['known_past_features']         = len(configs_data['data_processor']['columnno_X'])
        configs_data['known_future_features']       = len(configs_data['data_processor']['columnno_X_future'])
        configs_data['unknown_future_features']     = len(configs_data['data_processor']['columnno_y'])


    if 'OnlyZN-' in possible_types[RUNTHIS]:
        zone_num = int(possible_types[RUNTHIS][-1])
        # only one zone
        x_wea    = [0,  1,  2,  3,  4]
        x_occu   = [5]
        x_time   = [20, 21, 22, 23, 24, 25]
        x_window = [26 + zone_num-1]
        x_hvac   = [31 + zone_num-1]
        x_hsp    = [36 + zone_num-1]
        x_csp    = [41 + zone_num-1]
        y_iat    = [46 + zone_num-1]
        y_power  = [61 + zone_num-1]
        y_co2    = [67 + zone_num-1]

        configs_data['data_processor']['columnno_X']            = x_wea + x_occu + x_time + x_window + x_hvac + x_hsp + x_csp + y_iat + y_power + y_co2
        configs_data['data_processor']['columnno_X_future']     = [0]            + x_time + x_window + x_hvac + x_hsp + x_csp
        configs_data['data_processor']['columnno_y']                                                                          = y_iat + y_power + y_co2
        configs_data['known_past_features']         = len(configs_data['data_processor']['columnno_X'])
        configs_data['known_future_features']       = len(configs_data['data_processor']['columnno_X_future'])
        configs_data['unknown_future_features']     = len(configs_data['data_processor']['columnno_y'])

    if 'OnlyIATs' in possible_types[RUNTHIS]:
        # ONLY IAT
        x_wea    = [0,  1,  2,  3,  4]
        x_occu   = [5,  6,  7,  8,  9]
        x_time   = [20, 21, 22, 23, 24, 25]
        x_window = [26, 27, 28, 29, 30]
        x_hvac   = [31, 32, 33, 34, 35]
        x_hsp    = [36, 37, 38, 39, 40]
        x_csp    = [41, 42, 43, 44, 45]
        y_iat    = [46, 47, 48, 49, 50]
        y_power  = [61, 62, 63, 64, 65]
        y_co2    = [67, 68, 69, 70, 71]

        configs_data['data_processor']['columnno_X']            = x_wea + x_occu + x_time + x_window + x_hvac + x_hsp + x_csp + y_iat 
        configs_data['data_processor']['columnno_X_future']     = [0]            + x_time + x_window + x_hvac + x_hsp + x_csp
        configs_data['data_processor']['columnno_y']                                                                          = y_iat
        configs_data['known_past_features']         = len(configs_data['data_processor']['columnno_X'])
        configs_data['known_future_features']       = len(configs_data['data_processor']['columnno_X_future'])
        configs_data['unknown_future_features']        = len(configs_data['data_processor']['columnno_y'])

    if 'OnlyPOWERs' in possible_types[RUNTHIS]:
        # ONLY POWER
        x_wea    = [0,  1,  2,  3,  4]
        x_occu   = [5,  6,  7,  8,  9]
        x_time   = [20, 21, 22, 23, 24, 25]
        x_window = [26, 27, 28, 29, 30]
        x_hvac   = [31, 32, 33, 34, 35]
        x_hsp    = [36, 37, 38, 39, 40]
        x_csp    = [41, 42, 43, 44, 45]
        y_iat    = [46, 47, 48, 49, 50]
        y_power  = [61, 62, 63, 64, 65]
        y_co2    = [67, 68, 69, 70, 71]

        configs_data['data_processor']['columnno_X']            = x_wea + x_occu + x_time + x_window + x_hvac + x_hsp + x_csp + y_power 
        configs_data['data_processor']['columnno_X_future']     = [0]            + x_time + x_window + x_hvac + x_hsp + x_csp
        configs_data['data_processor']['columnno_y']                                                                          = y_power 
        configs_data['known_past_features']         = len(configs_data['data_processor']['columnno_X'])
        configs_data['known_future_features']       = len(configs_data['data_processor']['columnno_X_future'])
        configs_data['unknown_future_features']        = len(configs_data['data_processor']['columnno_y'])


    if 'OnlyCO2s' in possible_types[RUNTHIS]:
        #ONLY CO2
        x_wea    = [0,  1,  2,  3,  4]
        x_occu   = [5,  6,  7,  8,  9]
        x_time   = [20, 21, 22, 23, 24, 25]
        x_window = [26, 27, 28, 29, 30]
        x_hvac   = [31, 32, 33, 34, 35]
        x_hsp    = [36, 37, 38, 39, 40]
        x_csp    = [41, 42, 43, 44, 45]
        y_iat    = [46, 47, 48, 49, 50]
        y_power  = [61, 62, 63, 64, 65]
        y_co2    = [67, 68, 69, 70, 71]

        configs_data['data_processor']['columnno_X']            = x_wea + x_occu + x_time + x_window + x_hvac + x_hsp + x_csp + y_co2
        configs_data['data_processor']['columnno_X_future']     = [0]            + x_time + x_window + x_hvac + x_hsp + x_csp
        configs_data['data_processor']['columnno_y']                                                                          = y_co2
        configs_data['known_past_features']         = len(configs_data['data_processor']['columnno_X'])
        configs_data['known_future_features']       = len(configs_data['data_processor']['columnno_X_future'])
        configs_data['unknown_future_features']        = len(configs_data['data_processor']['columnno_y'])


    print('columnno_X: ')
    print(configs_data['data_processor']['columnno_X'], len(configs_data['data_processor']['columnno_X']))
    print('columnno_X_future:')
    print(configs_data['data_processor']['columnno_X_future'], len(configs_data['data_processor']['columnno_X_future']))
    print('columnno_y:')
    print(configs_data['data_processor']['columnno_y'], len(configs_data['data_processor']['columnno_y']))

    og_case(ident, configs_data, '') 

    end_script = dt.datetime.now()
    print('Time taken: %s' % str(end_script-start_script))
    print('All Done :) Good Job!')