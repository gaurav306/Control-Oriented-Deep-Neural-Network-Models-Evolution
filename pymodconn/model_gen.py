import pymodconn.model_gen_utils as Model_utils
from pymodconn.Encoder_class_layer import Encoder_class
from pymodconn.Decoder_class_layer import Decoder_class
from pymodconn.utils_layers import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import keras.backend as K
from keras.models import Model
import os
from tensorflow.keras.layers import Layer
from pymodconn.og_models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class CustomRearrange_M3(Layer):
    def __init__(self, **kwargs):
        super(CustomRearrange_M3, self).__init__(**kwargs)

    def call(self, inputs):
        # Assuming inputs is a list of tensors [predicted1, predicted2, ... , predicted5]

        # Split each tensor along the third axis
        splits1 = tf.split(inputs[0], 3, axis=2)
        splits2 = tf.split(inputs[1], 3, axis=2)
        splits3 = tf.split(inputs[2], 3, axis=2)
        splits4 = tf.split(inputs[3], 3, axis=2)
        splits5 = tf.split(inputs[4], 3, axis=2)

        # Reorder and concatenate
        reordered = [
            splits1[0], splits2[0], splits3[0], splits4[0], splits5[0],
            splits1[1], splits2[1], splits3[1], splits4[1], splits5[1],
            splits1[2], splits2[2], splits3[2], splits4[2], splits5[2]
        ]

        result = tf.concat(reordered, axis=2)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], 15)


class Model_Gen():
    def __init__(self, cfg, current_dt):
        self.cfg = cfg
        self.current_dt = current_dt

        if cfg['if_seed']:
            np.random.seed(cfg['seed'])
            tf.random.set_seed(cfg['seed'])
            tf.keras.utils.set_random_seed(cfg['seed'])
            tf.config.experimental.enable_op_determinism()
        
        self.n_past = cfg['n_past']
        self.n_future = cfg['n_future']
        self.known_past_features = cfg['known_past_features']
        self.unknown_future_features = cfg['unknown_future_features']
        self.known_future_features = cfg['known_future_features']
        self.control_future_cells = cfg['control_future_cells']

        self.all_layers_neurons = cfg['all_layers_neurons']
        self.all_layers_dropout = cfg['all_layers_dropout']
        
        self.model_type_prob = cfg['model_type_prob']
        self.loss_prob = cfg['loss_prob']
        self.q = cfg['quantiles']
        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(self.q)

        self.save_models_dir = cfg['save_models_dir']
        if not os.path.exists(cfg['save_models_dir']):
            os.makedirs(cfg['save_models_dir'])

    def build_model(self):
        
        if self.cfg['model_num'] == 'pymodconn':
            self.build_model_pymodconn()
        else:
            modenum = globals()[self.cfg['model_num']]
            self.model = modenum(self.cfg)
            self.compile_model(self.model)


    def build_model_pymodconn(self):
        timer = Model_utils.Timer()
        timer.start()
        print('[pymodconn] Model Compiling.....')

        # input for encoder_past
        encoder_inputs = tf.keras.layers.Input(
            shape=(self.n_past, self.known_past_features), name='encoder_past_inputs')
        
        encoder_outputs_seq, encoder_outputs_allstates = Encoder_class(
            self.cfg, str(1))(encoder_inputs, init_states=None)

        print('Before_encoder_outputs_seq.shape', encoder_outputs_seq.shape)
        print('Before_encoder_outputs_allstates', encoder_outputs_allstates)
        
        decoder_outputs_list = []
        if self.cfg['decoder_paper']['use_decoder_paper']:
            self.cfg['known_future_features'] = self.cfg['decoder_paper']['dec_in_out_%s' % self.cfg['decoder_paper']['dec_type']][0]
            self.cfg['unknown_future_features'] = self.cfg['decoder_paper']['dec_in_out_%s' % self.cfg['decoder_paper']['dec_type']][1]
        self.known_future_features = self.cfg['known_future_features']
        self.unknown_future_features = self.cfg['unknown_future_features']
        num_decoder = self.cfg['decoder_paper']['dec_in_out_%s' % self.cfg['decoder_paper']['dec_type']][2]

        if self.cfg['if_decrease_neurons_by_decoder']:
            neurons = self.cfg['all_layers_neurons']/num_decoder
            neurons = 8 * int(neurons/8)
            self.cfg['all_layers_neurons'] = neurons 
            print('self.all_layers_neurons-----------', self.cfg['all_layers_neurons'])

            #convert encoder ouputs to decoder inputs number of neurons
            encoder_outputs_seq = tf.keras.layers.Dense(units=self.cfg['all_layers_neurons'])(encoder_outputs_seq)
            new_encoder_outputs_allstates = []
            for item in encoder_outputs_allstates:
                item = tf.keras.layers.Dense(units=self.cfg['all_layers_neurons'])(item)
                new_encoder_outputs_allstates.append(item)
            encoder_outputs_allstates = new_encoder_outputs_allstates

            
            print('After_encoder_outputs_seq.shape', encoder_outputs_seq.shape)
            print('After_encoder_outputs_allstates', encoder_outputs_allstates)
        
        #-----------------------------------------------------------
        if self.cfg['decoder']['IF_TAKE_ENC_STATES'] == 0:
            encoder_outputs_allstates = None

        if self.cfg['decoder_paper']['dec_type'] == 1:
            i = 1
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)

            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            
            locals()[f"decoder_{i}_final_outputs"] = decoder_outputs4

            self.model = Model([encoder_inputs, locals()[f"decoder_1_inputs"]], [locals()[f"decoder_1_final_outputs"]])

            print('shapes of all inputs and outputs of model')
            print(self.model.input_shape)
            print(self.model.output_shape)

        if self.cfg['decoder_paper']['dec_type'] == 2:
            i = 0
            locals()[f"decoder_0_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_0_inputs")
            
            i = 1
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_0_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            locals()[f"decoder_{i}_final_outputs"] = decoder_outputs4

            i = 2
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_0_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            locals()[f"decoder_{i}_final_outputs"] = decoder_outputs4

            i = 3
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_0_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(locals()[f"decoder_{i}_outputs"])
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            locals()[f"decoder_{i}_final_outputs"] = decoder_outputs4
            
            final_output = tf.keras.layers.Concatenate()([locals()[f"decoder_1_final_outputs"],
                                                        locals()[f"decoder_2_final_outputs"],
                                                        locals()[f"decoder_3_final_outputs"]])

            self.model = Model([encoder_inputs, locals()[f"decoder_0_inputs"]], final_output)

        if self.cfg['decoder_paper']['dec_type'] == 3:
            for i in range(1, 6):
                print(i)
                locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
                locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)

                if self.model_type_prob == 'prob':
                    decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(locals()[f"decoder_{i}_outputs"])
                    decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                    decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                    if self.loss_prob == 'parametric':
                        decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
                elif self.model_type_prob == 'nonprob':
                    decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(locals()[f"decoder_{i}_outputs"])
                    decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                
                locals()[f"decoder_{i}_final_outputs"] = decoder_outputs4
            
            final_output = CustomRearrange_M3()([locals()[f"decoder_1_final_outputs"],
                                                locals()[f"decoder_2_final_outputs"],
                                                locals()[f"decoder_3_final_outputs"],
                                                locals()[f"decoder_4_final_outputs"],
                                                locals()[f"decoder_5_final_outputs"]])

            self.model = Model([encoder_inputs, 
                                locals()[f"decoder_1_inputs"],
                                locals()[f"decoder_2_inputs"],
                                locals()[f"decoder_3_inputs"],
                                locals()[f"decoder_4_inputs"],
                                locals()[f"decoder_5_inputs"]], 
                                final_output)


        if self.cfg['decoder_paper']['dec_type'] == 4:
            i = 1
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 2
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 3
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 4
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            decoder_outputs_all = MERGE_LIST(self.unknown_future_features)(decoder_outputs_list)
            
            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(decoder_outputs_all)
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(decoder_outputs_all)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            
            locals()[f"decoder_1_final_outputs"] = decoder_outputs4

            self.model = Model([encoder_inputs, locals()[f"decoder_1_inputs"],
                                                locals()[f"decoder_2_inputs"],
                                                locals()[f"decoder_3_inputs"],
                                                locals()[f"decoder_4_inputs"]], [locals()[f"decoder_1_final_outputs"]])

        if self.cfg['decoder_paper']['dec_type'] == 5:
            i = 1
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 2
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 3
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 4
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            i = 5
            locals()[f"decoder_{i}_inputs"] = tf.keras.layers.Input(shape=(self.n_future, self.known_future_features), name=f"decoder_{i}_inputs")
            locals()[f"decoder_{i}_outputs"] = Decoder_class(self.cfg, str(i))(locals()[f"decoder_{i}_inputs"], encoder_outputs_seq, encoder_states=encoder_outputs_allstates)
            decoder_outputs_list.append(locals()[f"decoder_{i}_outputs"])

            decoder_outputs_all = MERGE_LIST(self.unknown_future_features)(decoder_outputs_list)
            
            if self.model_type_prob == 'prob':
                decoder_outputs3 = tf.keras.layers.Dense(units=self.unknown_future_features * self.n_outputs_lastlayer)(decoder_outputs_all)
                decoder_outputs4 = tf.keras.layers.Reshape(target_shape=(self.n_future, self.unknown_future_features, self.n_outputs_lastlayer))(decoder_outputs3)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
                if self.loss_prob == 'parametric':
                    decoder_outputs4 = tf.keras.layers.Lambda(function=lambda x: tf.stack([x[:, :, :, 0], soft_relu(x[:, :, :, 1])], axis=-1))(decoder_outputs4)
            elif self.model_type_prob == 'nonprob':
                decoder_outputs4 = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -1, 1))(decoder_outputs_all)
                decoder_outputs4 = tf.keras.layers.Activation('linear', dtype='float32')(decoder_outputs4)
            
            locals()[f"decoder_1_final_outputs"] = decoder_outputs4

            self.model = Model([encoder_inputs, locals()[f"decoder_1_inputs"],
                                                locals()[f"decoder_2_inputs"],
                                                locals()[f"decoder_3_inputs"],
                                                locals()[f"decoder_4_inputs"],
                                                locals()[f"decoder_5_inputs"]], [locals()[f"decoder_1_final_outputs"]])



        self.compile_model(self.model)

    def compile_model(self, model):
        Model_utils.Build_utils(
            self.cfg, self.current_dt).postbuild_model(model)
        print('[pymodconn] Model compiled')

    def load_model(self, filepath):
        print('[pymodconn] Loading model from file %s' % filepath)
        # self.model = load_model(filepath)
        self.model.load_weights(filepath)
        # https://stackoverflow.com/a/69663259/6510598

    def load_optimizer(self, filepath):
        print('[pymodconn] Loading optimizer state from file %s' % filepath)
        with open(filepath, 'rb') as f:
            values = pickle.load(f)
            for var, value in zip(self.model.optimizer.variables(), values):
                var.assign(value)

    def forget_model(self):
        del self.model
        K.clear_session()
        print("Everything forgoten....maybe")
