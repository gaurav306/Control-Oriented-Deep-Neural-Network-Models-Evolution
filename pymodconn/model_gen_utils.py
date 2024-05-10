from keras.utils.layer_utils import count_params
from pymodconn.configs.configs_init import read_write_yaml
import os
from typing import *

import numpy as np
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import datetime as dt
K = tf.keras.backend

import numpy as np
from scipy.spatial.distance import euclidean
#from fastdtw import fastdtw
from tensorflow.python.ops import array_ops


class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        timetaken = end_dt - self.start_dt
        print('Time taken: %s' % (timetaken))
        # print("Current memory (MBs)",get_memory_info('GPU:0')['current'] / 1048576)
        print("")
        return timetaken

class Build_utils():
    """
    This class is used to compile, make sumamry and save as png.
    """

    def __init__(self, cfg, current_dt):
        self.cfg = cfg
        self.current_dt = current_dt
        self.batch_size = cfg['training']['batch_size']
        self.optimizer = cfg['optimizer']  # new
        self.SGD_lr = cfg['SGD']['lr']  # new
        self.SGD_mom = cfg['SGD']['momentum']  # new
        self.Adam_lr = cfg['Adam']['lr']
        self.Adam_b1 = cfg['Adam']['b1']
        self.Adam_b2 = cfg['Adam']['b2']
        self.Adam_epsi = cfg['Adam']['epsi']
        self.possible_loss = cfg['possible_loss']
        
        self.model_type_prob = cfg['model_type_prob']
        self.loss_prob = cfg['loss_prob']
        self.q = cfg['quantiles']
        self.control_future_cells = cfg['control_future_cells']

        self.n_features_output_block = 1

        self.q = np.unique(np.array(self.q))
        if 0.5 not in self.q:
            self.q = np.sort(np.append(0.5, self.q))
        self.n_outputs_lastlayer = 2 if self.loss_prob == 'parametric' else len(
            self.q)

        self.if_save_model_image = cfg['if_model_image']
        self.if_model_summary = cfg['if_model_summary']

        self.save_models_dir = cfg['save_models_dir']
        self.save_modelimage_name = os.path.join(
            self.save_models_dir, '%s_modelimage.png' % (self.current_dt))
        self.save_modelsummary_name = os.path.join(
            self.save_models_dir, '%s_modelsummary.txt' % (self.current_dt))
        self.save_modelconfig = os.path.join(
            self.save_models_dir, '%s_modelconfig.yaml' % (self.current_dt))
        

    def CVRMSE_Q50_prob_nonparametric(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 3] - y_true)))/(K.mean(y_true))

    def CVRMSE_Q50_prob_parametric(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred[:, :, :, 0] - y_true)))/(K.mean(y_true))


    
    def postbuild_model(self, model) -> Tuple[float, int]:

        self.model = model
        if self.optimizer == 'Adam':
            optimizer_model = tf.keras.optimizers.Adam(
                learning_rate=self.Adam_lr,
                beta_1=self.Adam_b1,
                beta_2=self.Adam_b2,
                epsilon=self.Adam_epsi)

        if self.optimizer == 'SGD':
            optimizer_model = tf.keras.optimizers.SGD(
                learning_rate=self.SGD_lr,
                momentum=self.SGD_mom)

        if self.model_type_prob == 'prob':

            if self.loss_prob == "parametric":  # maximum likelihood estimation is applied to estimate mean and stddev
                self.model.compile(loss=parametric_loss,
                                   optimizer=optimizer_model,
                                   metrics=[self.CVRMSE_Q50_prob_parametric],
                                   run_eagerly=True)

            elif self.loss_prob == "nonparametric":  # non parametric approach where quantiles are predicted
                self.model.compile(loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, self.q),
                                   optimizer=optimizer_model,
                                   metrics=[
                                       self.CVRMSE_Q50_prob_nonparametric],
                                   run_eagerly=True)

        elif self.model_type_prob == 'nonprob':
            self.loss = self.possible_loss[0]
            print('loss function : %s' % self.loss)
            self.model.compile(loss=self.loss,
                               optimizer=optimizer_model,
                               metrics=self.possible_loss,
                               run_eagerly=True)

        if self.if_model_summary:
            self.model.summary()

        self.trainable_count = count_params(model.trainable_weights)
        '''
        if self.cfg['if_attn_scores'] == 0:
            self.GET_MODEL_SIZE_GB = get_model_memory_usage(
                self.batch_size, self.model)
        else:
        '''
        self.GET_MODEL_SIZE_GB = 0
        
        print('Trainable parameters in the model : %d' % self.trainable_count)
        ''''''
        with open(self.save_modelsummary_name, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
        with open(self.save_modelsummary_name, 'a') as f:
            f.write('_' * 25 + '\n')
            f.write('Model size in GB : %f' % self.GET_MODEL_SIZE_GB)
        ''''''
        if self.if_save_model_image:
            print('Saving model as %s' % self.save_modelimage_name)
            plot_model(self.model, to_file=self.save_modelimage_name,
                       show_shapes=True, show_layer_names=True, dpi=600, expand_nested=True)
        
        read_write_yaml(self.save_modelconfig, 'w', self.cfg)

        print("[ModelClass] Building postmodel DONE!! model can be used as self.model")


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem  = single_layer_mem * s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p)
                             for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p)
                                 for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * \
        (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + \
        internal_model_mem_count
    print('Memory usage for model: {} GB'.format(gbytes))
    return gbytes


def nonparametric_loss(y_true, y_pred, q):
    '''
    Nonparametric loss function, also pinball loss, or quantle loss, see Section 3.2.1 in the DeepTCN paper.

    # Parameters:
    y_true: tf.Tensor.
            Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
            the batch size, n_forecast is the decoder length and n_targets is the number of target time series.

    y_pred: tf.Tensor.
            Predicted quantiles of target time series, a tensor with shape (n_samples, n_forecast, n_targets, n_quantiles)
            where n_samples is the batch size, n_forecast is the decoder length, n_targets is the number of target time
            series and n_quantiles is the number of quantiles.

    q: tf.Tensor.
            Quantiles, a 1-dimensional tensor with length equal to the number of quantiles.

    # Returns:
    tf.Tensor.
            Loss value, a scalar tensor.
    '''

    y_true = tf.cast(tf.expand_dims(y_true, axis=3), dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    q = tf.cast(tf.reshape(q, shape=(1, len(q))), dtype=tf.float32)
    e = tf.subtract(y_true, y_pred)

    L = tf.multiply(q, tf.maximum(0.0, e)) + \
        tf.multiply(1.0 - q, tf.maximum(0.0, - e))

    return tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(L, axis=-1), axis=-1))


def parametric_loss(y_true, params):
    '''
    Parametric loss function, see Section 3.2.2 in the DeepTCN paper.

    # Parameters:
    y_true: tf.Tensor.
            Actual values of target time series, a tensor with shape (n_samples, n_forecast, n_targets) where n_samples is
            the batch size, n_forecast is the decoder length and n_targets is the number of target time series.

    params: tf.Tensor.
            Predicted means and standard deviations of target time series, a tensor with shape (n_samples, n_forecast,
            n_targets, 2) where n_samples is the batch size, n_forecast is the decoder length and n_targets is the
            number of target time series.

    # Returns:
    tf.Tensor.
            Loss value, a scalar tensor.
    '''

    y_true = tf.cast(y_true, dtype=tf.float32)
    params = tf.cast(params, dtype=tf.float32)

    mu = params[:, :, :, 0]
    sigma = params[:, :, :, 1]

    L = 0.5 * tf.math.log(2 * np.pi) + tf.math.log(sigma) + \
        tf.math.divide(tf.math.pow(y_true - mu, 2), 2 * tf.math.pow(sigma, 2))

    return tf.experimental.numpy.nanmean(tf.experimental.numpy.nanmean(tf.experimental.numpy.nansum(L, axis=-1), axis=-1))

def CVRMSE_Q50_nonprob(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))/(K.mean(y_true))

def hourly_MSE(y_true, y_pred):

    def mse(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4  
        shape_2 = tf.shape(y_true)[2]
        y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        mse_loss = K.mean(K.square(y_pred_hourly - y_true_hourly), axis=-1)
        return mse_loss
    return mse(y_true, y_pred)

def hourly_CVRMSE(y_true, y_pred):
    def cvrmse(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4
        shape_2 = tf.shape(y_true)[2]  
        y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        mse = K.mean(K.square(y_pred_hourly - y_true_hourly), axis=-1)
        rmse = K.sqrt(mse)
        mean_true = K.mean(y_true_hourly, axis=-1)
        cvrmse_loss = (rmse / K.clip(mean_true, K.epsilon(), None)) * 100
        return cvrmse_loss
    return cvrmse(y_true, y_pred)

def cs_CVRMSE(y_true, y_pred):
    def cvrmse(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4
        shape_2 = tf.shape(y_true)[2]  
        #y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        #y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        mse = K.mean(K.square(y_pred - y_true), axis=-1)
        rmse = K.sqrt(mse)
        mean_true = K.mean(y_true, axis=-1)
        cvrmse_loss = (rmse / K.clip(mean_true, K.epsilon(), None)) * 100
        return cvrmse_loss
    return cvrmse(y_true, y_pred)

def cs_huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss function, see https://en.wikipedia.org/wiki/Huber_loss.
    This loss function is less sensitive to outliers in data than mean squared error. 
    It is quadratic for small error values and linear for large values. 
    This can be beneficial if your energy usage data has a lot of sudden 
    spikes that might otherwise disproportionately affect the model's learning process.
    """
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    small_error_loss = tf.square(error) / 2
    big_error_loss = delta * (tf.abs(error) - (0.5 * delta))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

def hourly_huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber loss function, see https://en.wikipedia.org/wiki/Huber_loss.
    This loss function is less sensitive to outliers in data than mean squared error. 
    It is quadratic for small error values and linear for large values. 
    This can be beneficial if your energy usage data has a lot of sudden 
    spikes that might otherwise disproportionately affect the model's learning process.
    """
    shape_1 = tf.shape(y_true)[1] // 4
    shape_2 = tf.shape(y_true)[2]  
    y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
    y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
    error = y_true_hourly - y_pred_hourly
    is_small_error = tf.abs(error) <= delta
    small_error_loss = tf.square(error) / 2
    big_error_loss = delta * (tf.abs(error) - (0.5 * delta))
    return tf.where(is_small_error, small_error_loss, big_error_loss)


def hourly_MAE(y_true, y_pred):
    def mae(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4
        shape_2 = tf.shape(y_true)[2]  
        y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        mae_loss = K.mean(K.abs(y_pred_hourly - y_true_hourly), axis=-1)
        return mae_loss
    return mae(y_true, y_pred)

def hourly_MSLE(y_true, y_pred):
    def msle(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4
        shape_2 = tf.shape(y_true)[2]  
        y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))

        # Adding a small constant to avoid taking log of zero
        y_true_hourly = y_true_hourly + K.epsilon()
        y_pred_hourly = y_pred_hourly + K.epsilon()

        msle_loss = K.mean(K.square(K.log(y_true_hourly) - K.log(y_pred_hourly)), axis=-1)
        return msle_loss
    return msle(y_true, y_pred)

def hourly_MAPE(y_true, y_pred):
    def mape(y_true, y_pred):
        shape_1 = tf.shape(y_true)[1] // 4
        shape_2 = tf.shape(y_true)[2]  
        y_true_hourly = K.reshape(K.sum(K.reshape(y_true, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        y_pred_hourly = K.reshape(K.sum(K.reshape(y_pred, (-1, shape_1, 4, shape_2)), axis=2), (-1, shape_1, shape_2))
        
        # Avoid division by zero and change to absolute values for proper calculation
        y_true_hourly = K.clip(K.abs(y_true_hourly), K.epsilon(), None)
        mape_loss = K.mean(K.abs((y_true_hourly - y_pred_hourly) / y_true_hourly), axis=-1) * 100
        return mape_loss
    return mape(y_true, y_pred)

def cvrmse_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true - y_pred))  # mean square error
    rmse = K.sqrt(mse)  # root mean square error
    mean_true = K.mean(y_true)  # mean of the true values
    cvrmse = rmse / mean_true * 100  # coefficient of variation of the root mean square error
    return cvrmse


