import os
import tensorflow as tf
from src.data_processor import DataClass
from pymodconn import Model_Gen
from src.train import TrainClass
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def transfer_learning_pipeline(ident, configs_data, tune_data, test_data, modelpath, weeknum, cords, callback_signal, tuneindent): 
    tf.keras.backend.clear_session()
    configs = configs_data
    if not os.path.exists(configs['save_models_dir']): os.makedirs(configs['save_models_dir'])
    if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])

    tune_X_past, tune_X_future, tune_Y_future = tune_data[0], tune_data[1], tune_data[2]
    test_X_past, test_X_future, test_Y_future = test_data[0], test_data[1], test_data[2]
    print("Tuning data shape, X_past, X_future, Y_future: ", tune_X_past.shape, tune_X_future.shape, tune_Y_future.shape)
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

    
    tl_type = configs['tl_case']['tl_type']

    if tl_type == 3 or tl_type == 4: #training parameters
        configs['training']['batch_size']           = configs['tl_case']['batch_size']
        configs['optimizer']                        = configs['tl_case']['optimizer']
        configs['training']['epochs']               = configs['tl_case']['epochs']
        configs[str(configs['optimizer'])]['lr']    = configs['tl_case']['lr']
    
    if tl_type == 1:
        print("Transfer learning type 1: Full model is fine tuned")
        for i, layer in enumerate(model_class.model.layers):
            layer.trainable = True
    
    if tl_type == 2:
        print()
        print("Transfer learning type 2: 'some' layer/s is/are fine tuned")
        for layer in model_class.model.layers:
            layer.trainable = False
        
        some_layers = configs['tl_case']['some_layer']
        for some_layer in some_layers:
            some_layer = int(some_layer)
            some_layer = model_class.model.layers[some_layer]
            some_layer.trainable = True

        model_class.model = tf.keras.Model(model_class.model.input, model_class.model.output)
        model_class.compile_model(model_class.model)
    
    if tl_type == 3: #'some' layer is fresh trained
        print()
        print("Transfer learning type 3: 'some' layer/s is/are fresh trained")
        for layer in model_class.model.layers:
            layer.trainable = False
        
        some_layers = configs['tl_case']['some_layer']
        for some_layer in some_layers:
            some_layer = int(some_layer)
            some_layer = model_class.model.layers[some_layer]
            some_layer.trainable = True
        
        model_class.model = tf.keras.Model(model_class.model.input, model_class.model.output)
        model_class.compile_model(model_class.model)
    
    if tl_type == 4: #extra layer is added and trained
        print("-------->>>>>>>>>> Transfer learning type 4: Extra layer is added and trained")
        for layer in model_class.model.layers:
            layer.trainable = False
        
        x = model_class.model.output
        if configs['tl_case']['extra_layer_type'] == 'Dense-25':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['extra_layer_type'] == 'Dense-50':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['extra_layer_type'] == 'Dense-100':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['extra_layer_type'] == 'Dense-200':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['extra_layer_type'] == 'Dense-400':
            x = tf.keras.layers.Dense(400)(x)
        
        elif configs['tl_case']['extra_layer_type'] == 'LSTM':
            x = tf.keras.layers.LSTM(200, return_sequences=True)(x)
        
        elif configs['tl_case']['extra_layer_type'] == 'GRU-25':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['extra_layer_type'] == 'GRU-50':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['extra_layer_type'] == 'GRU-100':
            x = tf.keras.layers.GRU(100, return_sequences=True)(x)
        elif configs['tl_case']['extra_layer_type'] == 'GRU-200':
            x = tf.keras.layers.GRU(200, return_sequences=True)(x)
        elif configs['tl_case']['extra_layer_type'] == 'GRU-400':
            x = tf.keras.layers.GRU(400, return_sequences=True)(x)

        elif 'MHA-4_' in configs['tl_case']['extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-8_' in configs['tl_case']['extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-16_' in configs['tl_case']['extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=configs['known_past_features'])(x, x)
        
        output = tf.keras.layers.Dense(configs['unknown_future_features'])(x)
        model_class.model = tf.keras.Model(model_class.model.input, output) 
        model_class.compile_model(model_class.model)
    

    if tl_type in [1,2,3,4]:
        tune_model = TrainClass(configs,ident,model_class.model)
        tune_model.tune_model([tune_X_past, tune_X_future], tune_Y_future, weeknum, callback_signal, tuneindent)
        model_class.load_model(tune_model.tuned_save_hf5_name)

        configs['IF_MAKE_DATA'] = 0
        data_class = DataClass(ident, configs)
        predicted = model_class.model.predict([test_X_past, test_X_future])
        print('Shape of predicted :', predicted.shape)
        data_class.scaleback_after_prediction_tuning(predicted, test_X_future, test_Y_future, weeknum, cords, tuneindent)
        
        del configs, tune_data, test_data, tune_model, model_class, data_class, predicted, tune_X_past, tune_X_future, tune_Y_future, test_X_past, test_X_future, test_Y_future
    



    if tl_type == 5:
        # Step one, tune model with a layer
        configs['training']['batch_size']           = configs['tl_case']['tl_5_1_batch_size']
        configs['optimizer']                        = configs['tl_case']['tl_5_1_optimizer']
        configs['training']['epochs']               = configs['tl_case']['tl_5_1_epochs']
        configs[str(configs['optimizer'])]['lr']    = configs['tl_case']['tl_5_1_lr']
        configs['tl_case']['tl_5_current'] = 'tune'

        print()
        print("-------->>>>>>>>>> Transfer learning type 5: 'some' layer/s is/are fine tuned")
        print(configs['tl_case']['tl_5_1_layer'])
        print()
        for layer in model_class.model.layers:
            layer.trainable = False
        
        some_layers = configs['tl_case']['tl_5_1_layer']
        for some_layer in some_layers:
            some_layer = int(some_layer)
            some_layer = model_class.model.layers[some_layer]
            some_layer.trainable = True

        model_class.model = tf.keras.Model(model_class.model.input, model_class.model.output)
        model_class.compile_model(model_class.model)

        tune_model = TrainClass(configs,ident,model_class.model)
        tune_model.tune_model([tune_X_past, tune_X_future], tune_Y_future, weeknum, callback_signal, tuneindent)
        model_class.load_model(tune_model.tuned_save_hf5_name)

        # Step two, train model with a new layer
        configs['training']['batch_size']           = configs['tl_case']['tl_5_2_batch_size']
        configs['training']['epochs']               = configs['tl_case']['tl_5_2_epochs']
        configs['optimizer']                        = configs['tl_case']['tl_5_2_optimizer']
        configs[str(configs['optimizer'])]['lr']    = configs['tl_case']['tl_5_2_lr']

        configs['tl_case']['tl_5_current'] = 'train'
        
        print()
        print("-------->>>>>>>>>> Transfer learning type 5: Extra layer is added and trained")
        print(configs['tl_case']['tl_5_2_extra_layer_type'])
        print()
        for layer in model_class.model.layers:
            layer.trainable = False
        
        x = model_class.model.output
        if configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-25':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-50':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-100':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-200':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-400':
            x = tf.keras.layers.Dense(400)(x)
        
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'LSTM':
            x = tf.keras.layers.LSTM(200, return_sequences=True)(x)
        
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-25':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-50':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-100':
            x = tf.keras.layers.GRU(100, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-200':
            x = tf.keras.layers.GRU(200, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-400':
            x = tf.keras.layers.GRU(400, return_sequences=True)(x)

        elif 'MHA-4_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-8_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-16_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=configs['known_past_features'])(x, x)
        
        output = tf.keras.layers.Dense(configs['unknown_future_features'])(x)
        model_class.model = tf.keras.Model(model_class.model.input, output) 
        model_class.compile_model(model_class.model)
        
        tune_model = TrainClass(configs,ident,model_class.model)
        tune_model.tune_model([tune_X_past, tune_X_future], tune_Y_future, weeknum, callback_signal, tuneindent)
        model_class.load_model(tune_model.tuned_save_hf5_name)

        configs['IF_MAKE_DATA'] = 0
        data_class = DataClass(ident, configs)
        predicted = model_class.model.predict([test_X_past, test_X_future])
        print('Shape of predicted :', predicted.shape)
        data_class.scaleback_after_prediction_tuning(predicted, test_X_future, test_Y_future, weeknum, cords, tuneindent)
        
        del configs, tune_data, test_data, tune_model, model_class, data_class, predicted, tune_X_past, tune_X_future, tune_Y_future, test_X_past, test_X_future, test_Y_future


    if tl_type == 6:
        # Step one, train model with a new layer
        configs['training']['batch_size']           = configs['tl_case']['tl_5_2_batch_size']
        configs['training']['epochs']               = configs['tl_case']['tl_5_2_epochs']
        configs['optimizer']                        = configs['tl_case']['tl_5_2_optimizer']
        configs[str(configs['optimizer'])]['lr']    = configs['tl_case']['tl_5_2_lr']

        configs['tl_case']['tl_5_current'] = 'train'
        
        print()
        print("-------->>>>>>>>>> Transfer learning type 6: Extra layer is added and trained")
        print(configs['tl_case']['tl_5_2_extra_layer_type'])
        print()
        for layer in model_class.model.layers:
            layer.trainable = False
        
        x = model_class.model.output
        if configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-25':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-50':
            x = tf.keras.layers.Dense(50)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-100':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-200':
            x = tf.keras.layers.Dense(100)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'Dense-400':
            x = tf.keras.layers.Dense(400)(x)
        
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'LSTM':
            x = tf.keras.layers.LSTM(200, return_sequences=True)(x)
        
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-25':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-50':
            x = tf.keras.layers.GRU(50, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-100':
            x = tf.keras.layers.GRU(100, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-200':
            x = tf.keras.layers.GRU(200, return_sequences=True)(x)
        elif configs['tl_case']['tl_5_2_extra_layer_type'] == 'GRU-400':
            x = tf.keras.layers.GRU(400, return_sequences=True)(x)

        elif 'MHA-4_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-8_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=configs['known_past_features'])(x, x)
        elif 'MHA-16_' in configs['tl_case']['tl_5_2_extra_layer_type']:
            x = tf.keras.layers.MultiHeadAttention(num_heads=16, key_dim=configs['known_past_features'])(x, x)
        
        output = tf.keras.layers.Dense(configs['unknown_future_features'])(x)
        model_class.model = tf.keras.Model(model_class.model.input, output) 
        model_class.compile_model(model_class.model)
        
        tune_model = TrainClass(configs,ident,model_class.model)
        tune_model.tune_model([tune_X_past, tune_X_future], tune_Y_future, weeknum, callback_signal, tuneindent)
        model_class.load_model(tune_model.tuned_save_hf5_name)

        # Step two, tune model with a layer
        configs['training']['batch_size']           = configs['tl_case']['tl_5_1_batch_size']
        configs['optimizer']                        = configs['tl_case']['tl_5_1_optimizer']
        configs['training']['epochs']               = configs['tl_case']['tl_5_1_epochs']
        configs[str(configs['optimizer'])]['lr']    = configs['tl_case']['tl_5_1_lr']
        configs['tl_case']['tl_5_current'] = 'tune'

        print()
        print("-------->>>>>>>>>> Transfer learning type 6: 'some' layer/s is/are fine tuned")
        print(configs['tl_case']['tl_5_1_layer'])
        print()
        for layer in model_class.model.layers:
            layer.trainable = False
        
        some_layers = configs['tl_case']['tl_5_1_layer']
        for some_layer in some_layers:
            some_layer = int(some_layer)
            some_layer = model_class.model.layers[some_layer]
            some_layer.trainable = True

        model_class.model = tf.keras.Model(model_class.model.input, model_class.model.output)
        model_class.compile_model(model_class.model)

        tune_model = TrainClass(configs,ident,model_class.model)
        tune_model.tune_model([tune_X_past, tune_X_future], tune_Y_future, weeknum, callback_signal, tuneindent)
        model_class.load_model(tune_model.tuned_save_hf5_name)

        configs['IF_MAKE_DATA'] = 0
        data_class = DataClass(ident, configs)
        predicted = model_class.model.predict([test_X_past, test_X_future])
        print('Shape of predicted :', predicted.shape)
        data_class.scaleback_after_prediction_tuning(predicted, test_X_future, test_Y_future, weeknum, cords, tuneindent)
        
        del configs, tune_data, test_data, tune_model, model_class, data_class, predicted, tune_X_past, tune_X_future, tune_Y_future, test_X_past, test_X_future, test_Y_future

