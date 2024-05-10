import datetime as dt
import numpy as np
import os
import tensorflow as tf
from src.configs.configs_init import read_write_yaml
from src.data_processor import DataClass
from pymodconn import Model_Gen
from src.train import TrainClass
import csv
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def training_pipeline_3_part1(ident, configs_data, all_data, loss_csv):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data

    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X_past, 		train_X_future, 	train_Y_future  	= all_data[0], all_data[1], all_data[2]
    validate_X_past, 	validate_X_future, 	validate_Y_future	= all_data[3], all_data[4], all_data[5]
    test_X_past, 		test_X_future, 		test_Y_future  		= all_data[6], all_data[7], all_data[8]

    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    
    
    configs['training']['epochs'] = configs['training']['prelim_runs_epochs']
    configs['training']['ifLRS'] = 0
    configs[str(configs['optimizer'])]['lr'] = 0.0001



    print("Training model %s" % ident)
    train_model = TrainClass(configs,ident,model_class.model)
    if configs['training']['if_validation']==1:
        train_model.train_fit_validate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
    if configs['training']['if_validation']==0:
        train_model.train_fit_novalidate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
    
    end_training_dt = dt.datetime.now()
    print("Training time: %s" % (end_training_dt - start_training_dt))

    with open(configs['data']['save_results_dir'] + ident + "restart_training_flag.txt", "r") as file:
        restart_training_flag_str = file.readline().strip()
    if restart_training_flag_str == 'True':
        print("Restarting training")
        return
    if restart_training_flag_str == 'False':
        print("Training finished")
        with open(loss_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if configs['training']['if_validation']==1:
                writer.writerow([ident, train_model.last_val_loss])
            if configs['training']['if_validation']==0:
                writer.writerow([ident, train_model.last_loss])


def training_pipeline_3_part2(ident, configs_data, all_data, bestmodel_ident):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X_past, 		train_X_future, 	train_Y_future  	= all_data[0], all_data[1], all_data[2]
    validate_X_past, 	validate_X_future, 	validate_Y_future	= all_data[3], all_data[4], all_data[5]
    test_X_past, 		test_X_future, 		test_Y_future  		= all_data[6], all_data[7], all_data[8]

    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    

    print("Checkpoint 2: Loading best model weights %s" % (bestmodel_ident))
    best_model_weights = os.path.join(configs['data']['save_results_dir'], '%s_modelcheckpoint_best.h5' % (bestmodel_ident))
    best_model_optimizer = os.path.join(configs['data']['save_results_dir'], '%s_optimizer_state.pkl' % (bestmodel_ident))
    model_class.load_model(best_model_weights)
    model_class.load_optimizer(best_model_optimizer)


    configs['training']['training_type'] = 1
    #configs['training']['ifLRS'] = 1


    print("Checkpoint 3: Training model")
    train_model = TrainClass(configs,ident,model_class.model)
    
    if configs['training']['if_validation']==1:
        train_model.train_fit_validate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
    if configs['training']['if_validation']==0:
        train_model.train_fit_novalidate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
    
    end_training_dt = dt.datetime.now()
    print(train_model.last_loss, train_model.last_val_loss,    str(end_training_dt-start_training_dt))
    
    configs['results']={}
    configs['results']['training run time taken']=str(end_training_dt-start_training_dt)
    configs['results']['last loss']=train_model.last_loss
    configs['results']['last val_loss']=train_model.last_val_loss
    configs['results']['Diff of loss and val_loss']=train_model.diff_val_loss_loss
    configs['results']['current_run_dt_ident']=str(ident)
    configs['results']['last epoch num']=train_model.last_epoch
    print('Saving results and model config in yaml file')
    yaml_name = os.path.join(configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident))  		
    read_write_yaml(filename=yaml_name, mode='w', data_yaml=configs)
    train_model.forget_model()
    
    model_class.load_model(train_model.save_hf5_name)

    print("Checkpoint 4: Predicting full sequence...\n")
    print('Shape of test_X_past \n', test_X_past.shape)
    print('Shape of test_X_future \n', test_X_future.shape)
    if configs['model_num'] == 'build_model_e1d1' or configs['model_num'] == 'build_model_e1d1_attn':
        print('[Predicting and testing] Concatenating past and padded future data')
        padding_shape = ((0, 0), (0, test_X_past.shape[1] - test_X_future.shape[1]), (0, 0))
        test_X_future_padded = np.pad(test_X_future, pad_width=padding_shape, mode='constant', constant_values=0)
        test_X_concatenated = np.concatenate((test_X_past, test_X_future_padded), axis=2)
        x_test = test_X_concatenated
        print('[testClass] padded shape of x_test: ', x_test.shape)
    
        predicted = model_class.model.predict(x_test)
    
    else:
        predicted = model_class.model.predict([test_X_past, test_X_future])
    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    data_class.scaleback_after_prediction(predicted, test_X_future, test_Y_future)
    
    del configs, all_data





def training_pipeline_1_2(ident, configs_data, all_data):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X_past, 		train_X_future, 	train_Y_future  	= all_data[0], all_data[1], all_data[2]
    validate_X_past, 	validate_X_future, 	validate_Y_future	= all_data[3], all_data[4], all_data[5]
    test_X_past, 		test_X_future, 		test_Y_future  		= all_data[6], all_data[7], all_data[8]

    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    LOAD_or_TRAIN_MODEL 		= configs['training']['LOAD_or_TRAIN_MODEL']

    if LOAD_or_TRAIN_MODEL == 'LOAD':
        print("Checkpoint 2: Loading model from %s" % (configs['training']['LOAD_MODEL_PATH']))
        model_class.load_model(configs['training']['LOAD_MODEL_PATH'])

    if LOAD_or_TRAIN_MODEL == 'TRAIN':
        print("Checkpoint 3: Training model")
        train_model = TrainClass(configs,ident,model_class.model)
        
        if configs['training']['if_validation']==1:
            train_model.train_fit_validate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
        if configs['training']['if_validation']==0:
            train_model.train_fit_novalidate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
        
        if configs['training']['training_type'] == 2:
            with open(configs['data']['save_results_dir'] + ident + "restart_training_flag.txt", "r") as file:
                restart_training_flag_str = file.readline().strip()
            if restart_training_flag_str == 'True':
                print("Restarting training")
                return

        end_training_dt = dt.datetime.now()
        print(train_model.last_loss, train_model.last_val_loss,    str(end_training_dt-start_training_dt))
        
        configs['results']={}
        configs['results']['training run time taken']=str(end_training_dt-start_training_dt)
        configs['results']['last loss']=train_model.last_loss
        configs['results']['last val_loss']=train_model.last_val_loss
        configs['results']['Diff of loss and val_loss']=train_model.diff_val_loss_loss
        configs['results']['current_run_dt_ident']=str(ident)
        configs['results']['last epoch num']=train_model.last_epoch
        print('Saving results and model config in yaml file')
        yaml_name = os.path.join(configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident))  		
        read_write_yaml(filename=yaml_name, mode='w', data_yaml=configs)
        train_model.forget_model()
    
    model_class.load_model(train_model.save_hf5_name)
    
    print("Checkpoint 4: Predicting full sequence...\n")
    print('Shape of test_X_past \n', test_X_past.shape)
    print('Shape of test_X_future \n', test_X_future.shape)
    predicted = model_class.model.predict([test_X_past, test_X_future])
    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    data_class.scaleback_after_prediction(predicted, test_X_future, test_Y_future)
    ret1 = configs['results']['training run time taken']
    ret2 = configs['results']['last loss']
    
    del configs, all_data
    return predicted.shape, ret1, ret2 



def training_pipeline_8_9(ident, configs_data, all_data):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X_past, 		train_X_future, 	train_Y_future  	= all_data[0], all_data[1], all_data[2]
    validate_X_past, 	validate_X_future, 	validate_Y_future	= all_data[3], all_data[4], all_data[5]
    test_X_past, 		test_X_future, 		test_Y_future  		= all_data[6], all_data[7], all_data[8]

    start_training_dt = dt.datetime.now()

    configs['if_attn_scores'] = 0

    print("Checkpoint 1")
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    LOAD_or_TRAIN_MODEL 		= configs['training']['LOAD_or_TRAIN_MODEL']

    if LOAD_or_TRAIN_MODEL == 'LOAD':
        print("Checkpoint 2: Loading model from %s" % (configs['training']['LOAD_MODEL_PATH']))
        model_class.load_model(configs['training']['LOAD_MODEL_PATH'])

    if LOAD_or_TRAIN_MODEL == 'TRAIN':
        print("Checkpoint 3: Training model")
        train_model = TrainClass(configs,ident,model_class.model)
        
        if configs['training']['if_validation']==1:
            train_model.train_fit_validate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
        if configs['training']['if_validation']==0:
            train_model.train_fit_novalidate([train_X_past,train_X_future], train_Y_future,[validate_X_past,validate_X_future], validate_Y_future)     #validation split =0.3
        
        if configs['training']['training_type'] == 9:
            with open(configs['data']['save_results_dir'] + ident + "restart_training_flag.txt", "r") as file:
                restart_training_flag_str = file.readline().strip()
            if restart_training_flag_str == 'True':
                print("Restarting training")
                return

        end_training_dt = dt.datetime.now()
        print(train_model.last_loss, train_model.last_val_loss,    str(end_training_dt-start_training_dt))
        
        configs['results']={}
        configs['results']['training run time taken']=str(end_training_dt-start_training_dt)
        configs['results']['last loss']=train_model.last_loss
        configs['results']['last val_loss']=train_model.last_val_loss
        configs['results']['Diff of loss and val_loss']=train_model.diff_val_loss_loss
        configs['results']['current_run_dt_ident']=str(ident)
        configs['results']['last epoch num']=train_model.last_epoch
        print('Saving results and model config in yaml file')
        yaml_name = os.path.join(configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident))  		
        read_write_yaml(filename=yaml_name, mode='w', data_yaml=configs)
        #train_model.forget_model()
    
    configs['if_attn_scores'] = 1
    print("Checkpoint 4: Predicting full sequence...\n")
    print('Shape of test_X_past \n', test_X_past.shape)
    print('Shape of test_X_future \n', test_X_future.shape)

    predicted = model_class.model.predict([test_X_past, test_X_future])

    attention = tf.keras.Model(inputs=model_class.model.input, 
                                outputs=model_class.model.get_layer("decoder_1_crossMHA-1").output)
    
    small_test_X_past = test_X_past       # Taking just one sample [:1]
    small_test_X_future = test_X_future   # Taking just one sample

    all_out = attention.predict([small_test_X_past, small_test_X_future])

    print('Shape of all_out')
    for all in all_out:
        print(all.shape)

    attention_scores = all_out[1]

    print('Saving attention scores')
    save_attention_scores = os.path.join(configs['data']['save_results_dir'], '%s_attention_scores.npy' % (ident))
    np.save(save_attention_scores, attention_scores)

    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    data_class.scaleback_after_prediction(predicted, test_X_future, test_Y_future)
    ret1 = configs['results']['training run time taken']
    ret2 = configs['results']['last loss']

    del configs, all_data
    return predicted.shape, ret1, ret2 


def training_pipeline_8_part1(ident, configs_data, all_data, loss_csv):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data

    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X, train_Y, validate_X, validate_Y, test_X, test_Y_future     = all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]

    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    configs['training']['epochs'] = configs['training']['prelim_runs_epochs']

    print("Training model %s" % ident)
    train_model = TrainClass(configs,ident,model_class.model)
    if configs['training']['if_validation']==1:
        train_model.train_fit_validate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3
    if configs['training']['if_validation']==0:
        train_model.train_fit_novalidate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3    
    end_training_dt = dt.datetime.now()
    print("Training time: %s" % (end_training_dt - start_training_dt))

    with open(configs['data']['save_results_dir'] + ident + "restart_training_flag.txt", "r") as file:
        restart_training_flag_str = file.readline().strip()
    if restart_training_flag_str == 'True':
        print("Restarting training")
        return
    if restart_training_flag_str == 'False':
        print("Training finished")
        with open(loss_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if configs['training']['if_validation']==1:
                writer.writerow([ident, train_model.last_val_loss])
            if configs['training']['if_validation']==0:
                writer.writerow([ident, train_model.last_loss])


def training_pipeline_8_part2(ident, configs_data, all_data, bestmodel_ident):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X, train_Y, validate_X, validate_Y, test_X, test_Y_future     = all_data[0], all_data[1], all_data[2], all_data[3], all_data[4], all_data[5]

    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    

    print("Checkpoint 2: Loading best model weights %s" % (bestmodel_ident))
    best_model_weights = os.path.join(configs['data']['save_results_dir'], '%s_modelcheckpoint_best.h5' % (bestmodel_ident))
    best_model_optimizer = os.path.join(configs['data']['save_results_dir'], '%s_optimizer_state.pkl' % (bestmodel_ident))
    model_class.load_model(best_model_weights)
    model_class.load_optimizer(best_model_optimizer)

    configs['training']['training_type'] = 1
    #configs['training']['epochs'] = configs['training']['final_runs_epochs']

    print("Checkpoint 3: Training model")
    train_model = TrainClass(configs,ident,model_class.model)
    
    if configs['training']['if_validation']==1:
        train_model.train_fit_validate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3
    if configs['training']['if_validation']==0:
        train_model.train_fit_novalidate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3    
    
    end_training_dt = dt.datetime.now()
    print(train_model.last_loss, train_model.last_val_loss,    str(end_training_dt-start_training_dt))
    
    configs['results']={}
    configs['results']['training run time taken']=str(end_training_dt-start_training_dt)
    configs['results']['last loss']=train_model.last_loss
    configs['results']['last val_loss']=train_model.last_val_loss
    configs['results']['Diff of loss and val_loss']=train_model.diff_val_loss_loss
    configs['results']['current_run_dt_ident']=str(ident)
    configs['results']['last epoch num']=train_model.last_epoch
    print('Saving results and model config in yaml file')
    yaml_name = os.path.join(configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident))  		
    read_write_yaml(filename=yaml_name, mode='w', data_yaml=configs)
    train_model.forget_model()
    
    model_class.load_model(train_model.save_hf5_name)
    
    print("Checkpoint 4: Predicting full sequence...\n")
    predicted = model_class.model.predict(test_X)
    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    data_class.scaleback_after_prediction(predicted, None, test_Y_future)
    
    del configs, all_data



def training_pipeline_6_7(ident, configs_data, all_data):     #has its own config as it takes config after some changes
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])
    
    train_X_past, 		train_X_future, 	train_Y_future  	= all_data[0], all_data[1], all_data[2]
    validate_X_past, 	validate_X_future, 	validate_Y_future	= all_data[3], all_data[4], all_data[5]
    test_X_past, 		test_X_future, 		test_Y_future  		= all_data[6], all_data[7], all_data[8]

    if configs['decoder_paper']['dec_type'] in [1, 2]:
        train_X    = [train_X_past,train_X_future]
        train_Y    = [train_Y_future]
        validate_X = [validate_X_past,validate_X_future]
        validate_Y = [validate_Y_future]

        test_X   = [test_X_past,test_X_future]

    if configs['decoder_paper']['dec_type'] == 222:
        train_Y_future1 = train_Y_future[:, :, [0, 1, 2, 3, 4]]
        train_Y_future2 = train_Y_future[:, :, [5, 6, 7, 8, 9]]
        train_Y_future3 = train_Y_future[:, :, [10, 11, 12, 13, 14]]

        validate_Y_future1 = validate_Y_future[:, :, [0, 1, 2, 3, 4]]
        validate_Y_future2 = validate_Y_future[:, :, [5, 6, 7, 8, 9]]
        validate_Y_future3 = validate_Y_future[:, :, [10, 11, 12, 13, 14]]

        train_X    = [train_X_past,train_X_future]
        train_Y    = [train_Y_future1, train_Y_future2, train_Y_future3]
        validate_X = [validate_X_past,validate_X_future]
        validate_Y = [validate_Y_future1, validate_Y_future2, validate_Y_future3]

        test_X   = [test_X_past,test_X_future]

    if configs['decoder_paper']['dec_type'] == 3:
        """
        train_Y_future1 = train_Y_future[:, :, [0, 5, 10]]
        train_Y_future2 = train_Y_future[:, :, [1, 6, 11]]
        train_Y_future3 = train_Y_future[:, :, [2, 7, 12]]
        train_Y_future4 = train_Y_future[:, :, [3, 8, 13]]
        train_Y_future5 = train_Y_future[:, :, [4, 9, 14]]

        validate_Y_future1 = validate_Y_future[:, :, [0, 5, 10]]
        validate_Y_future2 = validate_Y_future[:, :, [1, 6, 11]]
        validate_Y_future3 = validate_Y_future[:, :, [2, 7, 12]]
        validate_Y_future4 = validate_Y_future[:, :, [3, 8, 13]]
        validate_Y_future5 = validate_Y_future[:, :, [4, 9, 14]]
        """
        train_X_future1 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        train_X_future2 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        train_X_future3 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        train_X_future4 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        train_X_future5 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        validate_X_future1 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        validate_X_future2 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        validate_X_future3 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        validate_X_future4 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        validate_X_future5 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        train_X    = [train_X_past, train_X_future1, train_X_future2, train_X_future3, train_X_future4, train_X_future5]
        train_Y    = [train_Y_future]
        #train_Y    = [train_Y_future1, train_Y_future2, train_Y_future3, train_Y_future4, train_Y_future5]
        validate_X = [validate_X_past, validate_X_future1, validate_X_future2, validate_X_future3, validate_X_future4, validate_X_future5]
        validate_Y = [validate_Y_future]
        #validate_Y = [validate_Y_future1, validate_Y_future2, validate_Y_future3, validate_Y_future4, validate_Y_future5]

        test_X_future1 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        test_X_future2 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        test_X_future3 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        test_X_future4 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        test_X_future5 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        test_X    = [test_X_past, test_X_future1, test_X_future2, test_X_future3, test_X_future4, test_X_future5]
    
    if configs['decoder_paper']['dec_type'] == 4:
        train_X_future1 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11]]
        train_X_future2 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]]
        train_X_future3 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21]]
        train_X_future4 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 25, 26]]

        validate_X_future1 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11]]
        validate_X_future2 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]]
        validate_X_future3 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21]]
        validate_X_future4 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 25, 26]]

        train_X    = [train_X_past,train_X_future1, train_X_future2, train_X_future3, train_X_future4]
        train_Y    = [train_Y_future]
        validate_X = [validate_X_past,validate_X_future1, validate_X_future2, validate_X_future3, validate_X_future4]
        validate_Y = [validate_Y_future]

        test_X_future1 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11]]
        test_X_future2 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16]]
        test_X_future3 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21]]
        test_X_future4 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 22, 23, 24, 25, 26]]

        test_X    = [test_X_past,test_X_future1, test_X_future2, test_X_future3, test_X_future4]
    
    if configs['decoder_paper']['dec_type'] == 5:
        train_X_future1 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        train_X_future2 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        train_X_future3 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        train_X_future4 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        train_X_future5 = train_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        validate_X_future1 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        validate_X_future2 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        validate_X_future3 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        validate_X_future4 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        validate_X_future5 = validate_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        train_X    = [train_X_past,train_X_future1, train_X_future2, train_X_future3, train_X_future4, train_X_future5]
        train_Y    = [train_Y_future]
        validate_X = [validate_X_past,validate_X_future1, validate_X_future2, validate_X_future3, validate_X_future4, validate_X_future5]
        validate_Y = [validate_Y_future]

        test_X_future1 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12, 17, 22]]
        test_X_future2 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 8, 13, 18, 23]]
        test_X_future3 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 9, 14, 19, 24]]
        test_X_future4 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 10, 15, 20, 25]]
        test_X_future5 = test_X_future[:, :, [0, 1, 2, 3, 4, 5, 6, 11, 16, 21, 26]]

        test_X    = [test_X_past,test_X_future1, test_X_future2, test_X_future3, test_X_future4, test_X_future5]
        

    
    start_training_dt = dt.datetime.now()
    
    print("Checkpoint 1")
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    LOAD_or_TRAIN_MODEL 		= configs['training']['LOAD_or_TRAIN_MODEL']

    if LOAD_or_TRAIN_MODEL == 'LOAD':
        print("Checkpoint 2: Loading model from %s" % (configs['training']['LOAD_MODEL_PATH']))
        model_class.load_model(configs['training']['LOAD_MODEL_PATH'])

    if LOAD_or_TRAIN_MODEL == 'TRAIN':
        print("Checkpoint 3: Training model")
        train_model = TrainClass(configs,ident,model_class.model)
        
        if configs['training']['if_validation']==1:
            train_model.train_fit_validate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3
        if configs['training']['if_validation']==0:
            train_model.train_fit_novalidate(train_X, train_Y, validate_X, validate_Y)     #validation split =0.3
        
        if configs['training']['training_type'] == 7:
            with open(configs['data']['save_results_dir'] + ident + "restart_training_flag.txt", "r") as file:
                restart_training_flag_str = file.readline().strip()
            if restart_training_flag_str == 'True':
                print("Restarting training")
                return

        end_training_dt = dt.datetime.now()
        print(str(end_training_dt-start_training_dt))
        #print(train_model.last_loss, train_model.last_val_loss,    str(end_training_dt-start_training_dt))
        configs['results']={}
        configs['results']['training run time taken']=str(end_training_dt-start_training_dt)
        #configs['results']['last loss']=train_model.last_loss
        #configs['results']['last val_loss']=train_model.last_val_loss
        #configs['results']['Diff of loss and val_loss']=train_model.diff_val_loss_loss
        #configs['results']['current_run_dt_ident']=str(ident)
        #configs['results']['last epoch num']=train_model.last_epoch
        #print('Saving results and model config in yaml file')
        yaml_name = os.path.join(configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident))  		
        read_write_yaml(filename=yaml_name, mode='w', data_yaml=configs)
        train_model.forget_model()
    
    model_class.load_model(train_model.save_hf5_name)
    
    print("Checkpoint 4: Predicting full sequence...\n")
    predicted = model_class.model.predict(test_X)

    if configs['decoder_paper']['dec_type'] in [1, 2, 3]:
        predicted = predicted
    if configs['decoder_paper']['dec_type'] == 33:
        result = np.zeros((predicted[0].shape[0], predicted[0].shape[1], 15))
        result[:, :, [0, 5, 10]] = predicted[0]
        result[:, :, [1, 6, 11]] = predicted[1]
        result[:, :, [2, 7, 12]] = predicted[2]
        result[:, :, [3, 8, 13]] = predicted[3]
        result[:, :, [4, 9, 14]] = predicted[4]
        predicted = result
    if configs['decoder_paper']['dec_type'] == 4:
        predicted = predicted
    if configs['decoder_paper']['dec_type'] == 5:
        predicted = predicted

    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    data_class.scaleback_after_prediction(predicted, test_X_future, test_Y_future)
    ret1 = configs['results']['training run time taken']
    #ret2 = configs['results']['last loss']
    
    del configs, all_data
    return predicted.shape, ret1 #, ret2 



def just_testing_pipeline(ident, configs_data, test_data, modelpath, weeknum, cords, tuneindent):
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])

    test_X_past, test_X_future, test_Y_future = test_data[0], test_data[1], test_data[2]

    print("Test data shape, X_past, X_future, Y_future: ", test_X_past.shape, test_X_future.shape, test_Y_future.shape)

    
    if configs['training']['IF_multipleGPU']:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        configs['training']['batch_size'] = configs['training']['batch_size'] * strategy.num_replicas_in_sync
        with strategy.scope():
            model_class = Model_Gen(configs, ident)
            model_class.build_model()
    else:
        model_class = Model_Gen(configs, ident)
        model_class.build_model()
    
    model_class.load_model(modelpath)

    configs['IF_MAKE_DATA'] = 0
    data_class = DataClass(ident, configs)
    predicted = model_class.model.predict([test_X_past, test_X_future])
    print('Shape of predicted :', predicted.shape)
    data_class.scaleback_after_prediction_tuning(predicted, test_X_future, test_Y_future, weeknum, cords, tuneindent)
