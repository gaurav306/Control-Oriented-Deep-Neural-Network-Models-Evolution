import datetime as dt
import os
from typing import *
import time

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import Callback

os.environ['TF_CUDNN_USE_AUTOTUNE'] = '1'

class Timer():
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        timetaken = end_dt - self.start_dt
        print('Time taken: %s' % (timetaken))
        #print("Current memory (MBs)",get_memory_info('GPU:0')['current'] / 1048576)
        print("")
        return timetaken

class PrintLoggingCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        logs['epoch_duration'] = epoch_duration



def fill_nan(test_np): # fill nan with previous value # test_np should be 3D NumPy ndarray
    for i in range(test_np.shape[2]):
        arr = test_np[:,:,i]
        if np.isnan(arr).any():
            mask = np.isnan(arr)
            idx = np.where(~mask,np.arange(mask.shape[1]),0)
            np.maximum.accumulate(idx,axis=1, out=idx)
            out = arr[np.arange(idx.shape[0])[:,None], idx]
            test_np[:,:,i] = out

    return test_np


def average(lst: List[float]) -> float:
	"""Return the average of the list of numbers lst."""
	# Sum the numbers and divide by the number of numbers.
	return sum(lst) / len(lst)


class CustomEarlyStopping(Callback):
    def __init__(self, configs, ident, check_epoch_index, min_ERROR, error_type, **kwargs):
        super(CustomEarlyStopping, self).__init__(**kwargs)
        self.zero_epoch_ERROR = None
        self.check_epoch_ERROR = None
        self.check2_epoch_ERROR = None
        self.check_epoch_index = check_epoch_index
        self.check2_epoch_index = 7
        
        self.min_ERROR = min_ERROR
        self.error_type = error_type
        self.ident = ident
        self.configs = configs

    def on_epoch_end(self, epoch, logs=None):
        current_ERROR = logs.get(self.error_type)
        if epoch == 0:
            print("ERROR at 1st epoch is being recorded by CustomEarlyStopping: ", current_ERROR)
            self.zero_epoch_ERROR = current_ERROR
            if self.zero_epoch_ERROR > self.min_ERROR:
                self.model.stop_training = True
                print("Stopping training: ERROR at 1st epoch is greater than %s" % self.min_ERROR)
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        elif epoch == self.check_epoch_index:
            print("ERROR at check epoch is being recorded by CustomEarlyStopping: ", current_ERROR)
            self.check_epoch_ERROR = current_ERROR
        elif epoch == self.check2_epoch_index:
            self.check2_epoch_ERROR = current_ERROR
            
        if self.zero_epoch_ERROR and self.check_epoch_ERROR:
            if self.check_epoch_ERROR > self.zero_epoch_ERROR / 4:
                self.model.stop_training = True
                print("Stopping training: ERROR at check epoch is more than half of ERROR at 1st epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        if self.check_epoch_ERROR and self.check2_epoch_ERROR:
            if self.check2_epoch_ERROR > self.check_epoch_ERROR:
                self.model.stop_training = True
                print("Stopping training: ERROR at check epoch is more than ERROR at check2 epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        """
        with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
            file.write("False")
        """

class CustomEarlyStopping_tuning(Callback):
    def __init__(self, configs, ident, check_epoch_index, min_mape, error_type, **kwargs):
        super(CustomEarlyStopping_tuning, self).__init__(**kwargs)
        self.zero_epoch_mape = None
        self.check_epoch_mape = None
        self.check2_epoch_mape = None
        self.check_epoch_index = check_epoch_index
        self.check2_epoch_index = 3
        
        self.min_mape = min_mape
        self.error_type = error_type
        self.ident = ident
        self.configs = configs

    def on_epoch_end(self, epoch, logs=None):
        current_mape = logs.get(self.error_type)
        if epoch == 0:
            print("MAPE at 1st epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.zero_epoch_mape = current_mape
            if self.zero_epoch_mape > self.min_mape:
                self.model.stop_training = True
                print("Stopping training: MAPE at 1st epoch is greater than %s" % self.min_mape)
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        elif epoch == self.check_epoch_index:
            print("MAPE at check epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.check_epoch_mape = current_mape
        elif epoch == self.check2_epoch_index:
            print("MAPE at check epoch is being recorded by CustomEarlyStopping: ", current_mape)
            self.check2_epoch_mape = current_mape
            
        if self.zero_epoch_mape and self.check_epoch_mape:
            if self.check_epoch_mape > self.zero_epoch_mape :
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is more than MAPE at 1st epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        if self.check_epoch_mape and self.check2_epoch_mape:
            if round(self.check2_epoch_mape,5) == round(self.check_epoch_mape,5):
                self.model.stop_training = True
                print("Stopping training: MAPE at check epoch is equal to MAPE at check2 epoch")
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")
                raise SystemExit("Early Stopping Condition met - Restarting Training \n")
        """
        with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
            file.write("False")
        """

from tensorflow.keras.callbacks import TerminateOnNaN

class CustomTerminateOnNaN(TerminateOnNaN):
    def __init__(self, configs, ident):
        super(CustomTerminateOnNaN, self).__init__()
        self.ident = ident
        self.configs = configs

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'NaN or inf loss encountered at {batch}, terminating training and flagging for restart...')
                self.model.stop_training = True
                with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
                    file.write("True")


def mean_bias_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

def normalized_mean_bias_error(y_true, y_pred):
    return mean_bias_error(y_true, y_pred) / np.mean(y_true, axis=0) * 100

def coefficient_of_variation_root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)) / np.mean(y_true, axis=0) * 100

def root_mean_square_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=0))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred), axis=0)

def r_square(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.var(y_true) * len(y_true)
    return 1 - ss_res / ss_tot

def crps(y_true, y_pred_quantiles, quantiles=[0.05, 0.10, 0.50, 0.90, 0.95]):
    """
    Compute the Continuous Ranked Probability Score (CRPS).

    Parameters:
    - y_true : array-like of shape (n_samples, n_timesteps, n_features)
               True values.
    - y_pred_quantiles : array-like of shape (n_samples, n_timesteps, n_features, n_quantiles)
                         Predicted quantile values.

    Returns:
    - avg_crps : float
                 The average CRPS over all instances, timesteps, and features.
    """
    cum_obs = np.less_equal(y_true[..., np.newaxis], y_pred_quantiles).mean(axis=0)
    crps_values = (cum_obs - np.array(quantiles)) ** 2
    avg_crps = np.mean(crps_values)
    return avg_crps

def pinball_loss(y_true, y_pred, quantiles=[0.05, 0.10, 0.50, 0.90, 0.95]):
    """
    Compute the Pinball (Quantile) Loss.

    Parameters:
    - y_true : array-like of shape (n_samples, n_timesteps, n_features)
               True values.
    - y_pred : array-like of shape (n_samples, n_timesteps, n_features, n_quantiles)
               Predicted quantile values.
    - quantiles : list of float
                  The quantiles for which predictions are provided in y_pred.

    Returns:
    - avg_loss : float
                 The average Pinball loss over all instances, timesteps, features, and quantiles.
    """
    losses = np.empty((*y_true.shape, len(quantiles)))
    for idx, tau in enumerate(quantiles):
        errors = y_true - y_pred[..., idx]
        losses[..., idx] = np.where(errors >= 0, tau * errors, (1 - tau) * -errors)
    avg_loss = np.mean(losses)
    return avg_loss
