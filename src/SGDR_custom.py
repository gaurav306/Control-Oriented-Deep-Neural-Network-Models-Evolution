from typing import *
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import Callback

class SGDRScheduler_custom(Callback):       #modified to have warmup every restart
    '''
    Implements Stochastic Gradient Descent with Warm Restarts (SGDR) as a Keras callback function.
    
    Parameters:
        min_lr (float): Minimum learning rate during the entire training.
        max_lr (float): Maximum learning rate before each restart.
        steps_per_epoch (int): Number of steps per epoch. 
                                Is equal to number of training samples divided by batch size. = ceil(num_samples / batch_size)

        first_lr_drop_mult (float): Drop factor for learning rate after the first warmup.
        general_lr_decay (float): Decay factor for learning rate.

        if_warmup_or_cooldown_start (int): 0 for warmup start, 1 for cooldown start. Decide whether to start with warmup or cooldown step.

        init_cooldown_length (int): Initial number of epochs in a cooldown step.
        init_cooldown_mult_factor (float): Factor to grow the cooldown step length.

        warmup_length (int): Number of epochs in warmup period.
        warmup_mult_factor (float): Factor to grow the warmup length.
        if_post_warmup (int): 0 for no post warmup, 1 for post warmup. 
                                Decide whether to have post warmup step, i.e. after first warmup.


        number_of_cooldowns_before_switch (int): number of cooldowns when switch to a new cooldown length.
        new_cooldown_length (int): New cooldown length after switch epoch.
        new_cooldown_mult_factor (float): Factor to grow the new cooldown length.

        verbose (int): Verbosity mode.

    Attributes:
        Various internal state variables.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Cosine annealing learning rate as described in:
        Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. https://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 first_lr_drop_mult,
                 steps_per_epoch,
                 general_lr_decay,
                 if_warmup_or_cooldown_start,
                 init_cooldown_length,
                 init_cooldown_mult_factor,
                 warmup_length,
                 warmup_mult_factor,
                 if_post_warmup,
                 number_of_cooldowns_before_switch,
                 new_cooldown_length,
                 new_cooldown_mult_factor,
                 verbose = 0):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch

        self.first_lr_drop_mult = first_lr_drop_mult
        self.general_lr_decay = general_lr_decay

        self.if_warmup_or_cooldown_start = if_warmup_or_cooldown_start

        self.init_cooldown_length = init_cooldown_length
        self.init_cooldown_mult_factor = init_cooldown_mult_factor

        self.warmup_length = warmup_length
        self.warmup_mult_factor = warmup_mult_factor
        self.if_post_warmup = if_post_warmup

        self.number_of_cooldowns_before_switch = number_of_cooldowns_before_switch
        self.new_cooldown_length = new_cooldown_length
        self.new_cooldown_mult_factor = new_cooldown_mult_factor
        
        self.verbose = verbose        

        self.batch_since_restart = 0
        self.batch_since_warmup = 0
        self.warmup_cycle_mode = self.if_warmup_or_cooldown_start    #warmup =0, cooldown=1
        
        if self.warmup_cycle_mode==0:                   #warmup
            self.next_restart = warmup_length
        if self.warmup_cycle_mode==1:                   #cooldown
            self.next_restart = init_cooldown_length
        
        self.cycle_step_flag = 0                              #counts the number of warmup or cycle mode      
        self.flagswitch = 0

        self.history = {}

        print('[TrainClass] CWR added as callback')

    def cosine_cooldown_clr(self):
        '''Calculate the learning rate cosine fall.'''
        fraction_to_restart = (self.batch_since_restart+1) / (self.steps_per_epoch * self.init_cooldown_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        #lr = self.min_lr + (self.max_lr - self.min_lr) * (1-fraction_to_restart)
        if self.verbose == 1:
            print('[TrainClass] cosine_cooldown_clr - fraction_to_restart-', fraction_to_restart)
        return lr
    
    def warmup_clr(self):
        '''Calculate the learning rate during warmup.'''
        fraction_to_restart = (self.batch_since_warmup+1) / (self.steps_per_epoch * self.warmup_length)
        lr = self.min_lr + (self.max_lr - self.min_lr) * fraction_to_restart
        if self.verbose == 1:
            print('[TrainClass] warmup_clr - fraction_to_restart-', fraction_to_restart)
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        #K.set_value(self.model.optimizer.lr, self.max_lr) # when no warmup
        K.set_value(self.model.optimizer.lr, self.min_lr) # when warmup we start from lowest lr

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        if self.warmup_cycle_mode==0:
            K.set_value(self.model.optimizer.lr, self.warmup_clr()) 
            self.batch_since_warmup += 1  
        if self.warmup_cycle_mode==1:
            K.set_value(self.model.optimizer.lr, self.cosine_cooldown_clr())
            self.batch_since_restart += 1
        
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs={}):

        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch+1 == self.next_restart:
            self.warmup_cycle_mode = not self.warmup_cycle_mode # swap 0 to 1 and vice versa
            
            self.batch_since_restart = 0
            self.batch_since_warmup = 0
            
            if self.if_warmup_or_cooldown_start == 0:
                if self.warmup_cycle_mode==0: 
                    if self.cycle_step_flag == 1:                              
                        self.max_lr = self.max_lr * self.first_lr_drop_mult   # drop lr after first warmup or cooldown mode
                        if self.if_post_warmup == 0:
                            self.warmup_length = 1 
                            self.warmup_mult_factor = 1
                    else:
                        self.max_lr *= self.general_lr_decay

            if self.if_warmup_or_cooldown_start == 1:
                if self.warmup_cycle_mode==0: 
                    if self.cycle_step_flag == 0:                              
                        self.max_lr = self.max_lr * self.first_lr_drop_mult   # drop lr after first warmup or cooldown mode
                        if self.if_post_warmup == 0:
                            self.warmup_length = 1 
                            self.warmup_mult_factor = 1
                    else:
                        self.max_lr *= self.general_lr_decay
            
            if self.warmup_cycle_mode==0: 
                if self.verbose == 1:
                    print('Warmup length: %s' % self.warmup_length)
                    print('Warmup mult factor: %s' % self.warmup_mult_factor)
                self.warmup_length = np.ceil(self.warmup_length * self.warmup_mult_factor)
                self.next_restart += self.warmup_length
                if self.verbose == 1:
                    print('[TrainClass] Warmup restart at epoch %s, lr-' % (epoch+1), K.get_value(self.model.optimizer.lr))

            if self.warmup_cycle_mode==1: 
                if self.verbose == 1:
                    print('Cooldown length: %s' % self.init_cooldown_length)
                    print('Cooldown mult factor: %s' % self.init_cooldown_mult_factor)
                if self.flagswitch == self.number_of_cooldowns_before_switch:
                    self.init_cooldown_length = self.new_cooldown_length
                    self.init_cooldown_mult_factor = self.new_cooldown_mult_factor

                self.init_cooldown_length = np.ceil(self.init_cooldown_length * self.init_cooldown_mult_factor)
                self.next_restart += self.init_cooldown_length
                self.flagswitch = self.flagswitch + 1
                self.cycle_step_flag = self.cycle_step_flag + 1 
                if self.verbose == 1:
                    print('[TrainClass] Cycle restart at epoch %s, lr-' % (epoch+1), K.get_value(self.model.optimizer.lr))
        
                
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)  