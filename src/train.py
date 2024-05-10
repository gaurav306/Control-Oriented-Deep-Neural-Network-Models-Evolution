import os
import numpy as np
import pandas as pd
from src.utils import Timer, average, CustomEarlyStopping, \
	PrintLoggingCallback, CustomEarlyStopping_tuning, CustomTerminateOnNaN
from src.SGDR_custom import SGDRScheduler_custom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TerminateOnNaN
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from src.plots import PlotClass

import pickle


from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


class TrainClass():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self,configs,ident,model_to_be_trained):
		if configs['if_seed']:
			np.random.seed(configs['seed'])
			tf.random.set_seed(configs['seed'])
		
		self.epochs             =configs['training']['epochs']
		self.batch_size         =configs['training']['batch_size']
		self.save_models_dir    =configs['save_models_dir']
		self.save_results_dir   =configs['data']['save_results_dir']
		self.future_data_col    =configs['known_future_features']
		self.n_past             =configs['n_past']
		self.n_features_input   =configs['known_past_features'] + self.future_data_col

		self.all_layers_neurons =configs['all_layers_neurons']
		self.all_layers_dropout =configs['all_layers_dropout']

		self.n_future           =configs['n_future']
		self.n_features_output  =configs['unknown_future_features']

		self.Adam_lr            =configs['Adam']['lr']
		self.Adam_b1            =configs['Adam']['b1']
		self.Adam_b2            =configs['Adam']['b2']
		self.Adam_epsi          =configs['Adam']['epsi']
		self.loss_prob          =configs['loss_prob']
		self.q                  =configs['quantiles']
		
		self.q                  =np.unique(np.array(self.q))
		if 0.5 not in self.q:
			self.q = np.sort(np.append(0.5, self.q))
		self.n_outputs_lastlayer= 2 if self.loss_prob == 'parametric' else len(self.q)

		self.model_type_prob    =configs['model_type_prob']
		self.if_validation      =configs['training']['if_validation']
		self.seq_len            =configs['training']['seq_len']
		self.if_save_model_image=configs['if_model_image']
		self.if_model_summary   =configs['if_model_summary']
		self.training_verbose   =configs['training']['training_verbose']
		self.ifEarlyStop        =configs['training']['ifEarlyStop']
		self.EarlyStop_patience =configs['training']['EarlyStop_patience']
		self.EarlyStop_min_delta=configs['training']['EarlyStop_min_delta']
		self.EarlyStop_start_from_epoch =configs['training']['EarlyStop_start_from_epoch']
		self.ifLRS				=configs['training']['ifLRS']
		self.model              =model_to_be_trained
		self.ident 				=ident
		self.configs 			=configs

		self.save_training_history_file = os.path.join(self.save_results_dir, '%s.csv' % (ident))
		self.save_training_history_file_validation = os.path.join(self.save_results_dir, '%s_validation.csv' % (ident))
		self.save_training_history      = os.path.join(self.save_results_dir, '%s_history.png' % (ident))
		self.save_data_ndarray_pred     = os.path.join(self.save_results_dir, '%s_ndarray_predicted.npy' % (ident))
		self.save_data_ndarray_pastx    = os.path.join(self.save_results_dir, '%s_ndarray_pastx.npy' % (ident))
		self.save_data_ndarray_futurex  = os.path.join(self.save_results_dir, '%s_ndarray_futurex.npy' % (ident))
		self.save_data_ndarray_futurey  = os.path.join(self.save_results_dir, '%s_ndarray_futurey.npy' % (ident))
		self.save_hf5_name              = os.path.join(self.save_results_dir, '%s_modelcheckpoint_best.h5' % (ident))
		self.save_optimizer_state	  	= os.path.join(self.save_results_dir, '%s_optimizer_state.pkl' % (ident))
		self.save_modelimage_name       = os.path.join(self.save_models_dir, '%s_modelimage.png' % (ident))

		self.model_num = configs['model_num']		
	
	def callbacks_list(self, y_train_len):	
		callbacks = []
		def lr_scheduler_increase(epoch, lr):
			initial_lrate = 0.00001
			increase = 1.08
			lrate = initial_lrate * math.pow(increase,epoch)
			return lrate    
				
		self.callback1      =SGDRScheduler_custom(
									min_lr 						= self.configs['CosineWarmRestart']['min_lr'], 
									max_lr 						= self.configs['CosineWarmRestart']['max_lr'],
									steps_per_epoch 			= self.configs['CosineWarmRestart']['steps_per_epoch'],
									first_lr_drop_mult 			= self.configs['CosineWarmRestart']['first_lr_drop_mult'],
									general_lr_decay 			= self.configs['CosineWarmRestart']['general_lr_decay'],
									if_warmup_or_cooldown_start = self.configs['CosineWarmRestart']['if_warmup_or_cooldown_start'],
									init_cooldown_length 		= self.configs['CosineWarmRestart']['init_cooldown_length'],
									init_cooldown_mult_factor 	= self.configs['CosineWarmRestart']['init_cooldown_mult_factor'],
									warmup_length 				= self.configs['CosineWarmRestart']['warmup_length'],
									warmup_mult_factor 			= self.configs['CosineWarmRestart']['warmup_mult_factor'],
									if_post_warmup 				= self.configs['CosineWarmRestart']['if_post_warmup'],
									number_of_cooldowns_before_switch = self.configs['CosineWarmRestart']['number_of_cooldowns_before_switch'],
									new_cooldown_length 		= self.configs['CosineWarmRestart']['new_cooldown_length'],
									new_cooldown_mult_factor 	= self.configs['CosineWarmRestart']['new_cooldown_mult_factor']
									)

		self.callback2      =CustomTerminateOnNaN(self.configs, self.ident)
		
		if self.if_validation:
			self.callback3      =EarlyStopping(monitor='val_loss', 
											min_delta=self.configs['training']['EarlyStop_min_delta'], 
											patience=self.configs['training']['EarlyStop_patience'])
		else:
			self.callback3      =EarlyStopping(monitor='loss', 
											min_delta=self.configs['training']['EarlyStop_min_delta'], 
											patience=self.configs['training']['EarlyStop_patience'])			
		'''
		self.callback4      =ModelCheckpoint(filepath=self.save_hf5_name,
				       						monitor = "val_loss",
										    save_best_only=False,  
				       						save_weights_only=True,
										    mode='min',
										    verbose=1, 
											save_format="h5")
		'''
		self.callback4      =ModelCheckpoint(filepath=self.save_hf5_name,
				       						save_best_only=False,  
				       						save_weights_only=True,
										    verbose=1, 
											save_format="h5")
		self.callback5      =CSVLogger(self.save_training_history_file, append=True, separator=',')
		
		callbacks.append(PrintLoggingCallback())
		if self.configs['training']['ifLRS']:
			callbacks.append(self.callback1)
		callbacks.append(self.callback2)
		if self.configs['training']['ifEarlyStop']:
			callbacks.append(self.callback3)
		callbacks.append(self.callback4)
		callbacks.append(self.callback5)
		
		"""
		if self.configs['training']['training_type'] in [2,3]:
			callbacks.append(CustomEarlyStopping(self.configs,
												self.ident,
												self.configs['CustomEarlyStopping']['check_epoch_index'], 
												self.configs['CustomEarlyStopping']['min_error'],
												self.configs['CustomEarlyStopping']['error_type']))
		"""
		
		return callbacks


	def train_fit(self, x_data, y_data,vsplit): #validate_split = 0.25 #25% of data will be used for validation
		timer = Timer()
		timer.start()

		print('[TrainClass] Spliting data')
		
		x_data = np.asarray(x_data)
		y_data = np.asarray(y_data)

		x_train, x_validate = np.split(x_data,[round(len(x_data)*(1-vsplit))])
		y_train, y_validate = np.split(y_data,[round(len(y_data)*(1-vsplit))])

		print('[TrainClass] Training Started')
		print('[TrainClass] %s epochs, %s batch size' % (self.epochs, self.batch_size))
		
		self.history_model=self.model.fit(
				x_train, y_train,
				epochs=self.epochs,
				validation_data=(x_validate, y_validate),
				batch_size=self.batch_size,
				verbose=self.training_verbose,
				callbacks=self.callbacks_list(len(y_train)))
		
		#self.model.save_weights(self.save_hf5_name)   

		self.last_epoch=len(self.history_model.history['loss'])
		self.last_loss=average(self.history_model.history['loss'][-7:])
		self.last_val_loss=average(self.history_model.history['val_loss'][-7:])
		self.diff_val_loss_loss= abs(self.last_loss-self.last_val_loss)
		print("loss: "+str(round(self.last_loss,3)))
		print("val_loss: "+str(round(self.last_val_loss,3)))
		print("Diff of loss and val_loss: "+str(round(self.diff_val_loss_loss,3)))

		#self.last_mape = self.history_model.history['mape'][-1]
		
		print('[TrainClass] Training Completed. Model saved as %s' % self.save_hf5_name)
		
		with open(self.save_optimizer_state, 'wb') as f:
			pickle.dump([var.numpy() for var in self.model.optimizer.variables()], f)
		
		timer.stop()
	
	def train_fit_validate(self, x_train, y_train, x_validate, y_validate):
		timer = Timer()
		timer.start()

		print('[TrainClass] Training Started')
		print('[TrainClass] %s epochs, %s batch size' % (self.epochs, self.batch_size))
		
		if self.model_num == 'build_model_e1d1' or self.model_num == 'build_model_e1d1_attn':
			print('[TrainClass] Concatenating past and padded future data')
			train_X_past  = x_train[0]
			train_X_future= x_train[1]
			padding_shape = ((0, 0), (0, train_X_past.shape[1] - train_X_future.shape[1]), (0, 0))
			train_X_future_padded = np.pad(train_X_future, pad_width=padding_shape, mode='constant', constant_values=0)
			train_X_concatenated = np.concatenate((train_X_past, train_X_future_padded), axis=2)
			x_train = train_X_concatenated
			print('[TrainClass] padded shape of x_train: ', x_train.shape)

			validate_X_past  = x_validate[0]
			validate_X_future= x_validate[1]
			padding_shape = ((0, 0), (0, validate_X_past.shape[1] - validate_X_future.shape[1]), (0, 0))
			validate_X_future_padded = np.pad(validate_X_future, pad_width=padding_shape, mode='constant', constant_values=0)
			validate_X_concatenated = np.concatenate((validate_X_past, validate_X_future_padded), axis=2)
			x_validate = validate_X_concatenated
			print('[TrainClass] padded shape of x_validate: ', x_validate.shape)

		self.history_model=self.model.fit(
				x_train, y_train,
				epochs=self.epochs,
				validation_data=(x_validate, y_validate),
				batch_size=self.batch_size,
				verbose=self.training_verbose,
				callbacks=self.callbacks_list(len(y_train)))
		
		''''''
		#self.model.save_weights(self.save_hf5_name)
		if self.configs['training']['ifLRS']:
			plots_all = PlotClass(self.ident,self.configs)
			plots_all.history_plot_model_fit(self.history_model)
			print('Plots saved')
		
		self.last_epoch=len(self.history_model.history['loss'])
		self.last_loss=average(self.history_model.history['loss'][-7:])
		self.last_val_loss=average(self.history_model.history['val_loss'][-7:])
		self.diff_val_loss_loss= abs(self.last_loss-self.last_val_loss)
		print("loss: "+str(round(self.last_loss,3)))
		print("val_loss: "+str(round(self.last_val_loss,3)))
		print("Diff of loss and val_loss: "+str(round(self.diff_val_loss_loss,3)))
		self.last_mape = 0
		'''
		self.last_epoch=0
		self.last_loss=0
		self.last_val_loss=0
		self.diff_val_loss_loss= abs(self.last_loss-self.last_val_loss)
		print("loss: "+str(round(self.last_loss,3)))
		print("val_loss: "+str(round(self.last_val_loss,3)))
		print("Diff of loss and val_loss: "+str(round(self.diff_val_loss_loss,3)))
		self.last_mape = 0		
		print('[TrainClass] Training Completed. Model saved as %s' % self.save_hf5_name)
		'''
		with open(self.save_optimizer_state, 'wb') as f:
			pickle.dump([var.numpy() for var in self.model.optimizer.variables()], f)

		timer.stop()

	
	def train_fit_novalidate(self, x_train, y_train, x_validate, y_validate):
		timer = Timer()
		timer.start()

		print('[TrainClass] Training Started - train_fit_novalidate')
		print('[TrainClass] %s epochs, %s batch size' % (self.epochs, self.batch_size))
		
		self.history_model=self.model.fit(
				x_train, y_train,
				epochs=self.epochs,
				batch_size=self.batch_size,
				verbose=self.training_verbose,
				callbacks=self.callbacks_list(len(y_train)))
		
		#self.model.save_weights(self.save_hf5_name)   

		"""
		fig = plt.figure(figsize=(25, 2))
		plt.plot(self.history_model.history['loss'])
		plt.title("Model Loss")
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend(['Train'])
		plt.savefig(self.save_training_history)
		"""
		if self.configs['training']['ifLRS']:
			plots_all = PlotClass(self.ident,self.configs)
			plots_all.history_plot_model_fit_novalidate(self.history_model)

		self.last_epoch=len(self.history_model.history['loss'])
		self.last_loss=average(self.history_model.history['loss'][-7:])
		self.last_val_loss=0
		self.diff_val_loss_loss= abs(self.last_loss-self.last_val_loss)
		print("loss: "+str(round(self.last_loss,3)))
		print("val_loss: "+str(round(self.last_val_loss,3)))
		print("Diff of loss and val_loss: "+str(round(self.diff_val_loss_loss,3)))

		#self.last_mape = self.history_model.history['mape'][-1]
		
		print('[TrainClass] Training Completed. Model saved as %s' % self.save_hf5_name)

		with open(self.save_optimizer_state, 'wb') as f:
			pickle.dump([var.numpy() for var in self.model.optimizer.variables()], f)

		timer.stop()


	def tune_model(self, x_tune, y_tune, weeknum, callback_signal, tuneindent):
		timer = Timer()
		timer.start()
		print('[TrainClass] tuning Started for %s %s' % (tuneindent, weeknum))
		callback_all = [CustomEarlyStopping_tuning(self.configs, self.ident,2,200,self.configs['CustomEarlyStopping']['error_type']),
						CustomTerminateOnNaN(self.configs, self.ident),
						EarlyStopping(monitor='loss', 
							min_delta=self.configs['training']['EarlyStop_min_delta'], 
							patience=self.configs['training']['EarlyStop_patience'])]

		if self.configs[str(self.configs['optimizer'])]['lr'] == 0.00001:
			if self.configs['tl_case']['tl_type'] == 3 or self.configs['tl_case']['tl_type'] == 4:
				step_per_epoch = int(len(y_tune)/self.batch_size) + 1
				self.configs['CosineWarmRestart']['CWR_steps_per_epoch'] = step_per_epoch

				CWR_min_lr             =self.configs['CosineWarmRestart']['CWR_min_lr']
				CWR_max_lr             =self.configs['CosineWarmRestart']['CWR_max_lr']
				CWR_steps_per_epoch    =self.configs['CosineWarmRestart']['CWR_steps_per_epoch']
				CWR_lr_decay           =self.configs['CosineWarmRestart']['CWR_lr_decay']
				CWR_cycle_length       =self.configs['CosineWarmRestart']['CWR_cycle_length']
				CWR_cycle_mult_factor  =self.configs['CosineWarmRestart']['CWR_cycle_mult_factor']
				CWR_warmup_length      =self.configs['CosineWarmRestart']['CWR_warmup_length']
				CWR_warmup_mult_factor =self.configs['CosineWarmRestart']['CWR_warmup_mult_factor']
				
				callback1      =SGDRScheduler_warmup(
											min_lr=CWR_min_lr ,
											max_lr=CWR_max_lr,
											steps_per_epoch=CWR_steps_per_epoch,
											lr_decay=CWR_lr_decay,
											cycle_length=CWR_cycle_length,
											cycle_mult_factor=CWR_cycle_mult_factor,
											warmup_length=CWR_warmup_length,
											warmup_mult_factor=CWR_warmup_mult_factor)
				callback_all.append(callback1)
		
		if self.configs['tl_case']['tl_type'] == 5 or self.configs['tl_case']['tl_type'] == 6:
			if self.configs['tl_case']['tl_5_current'] == 'train':
				step_per_epoch = int(len(y_tune)/self.batch_size) + 1
				self.configs['CosineWarmRestart']['CWR_steps_per_epoch'] = step_per_epoch
				CWR_min_lr             =self.configs['CosineWarmRestart']['CWR_min_lr']
				#CWR_max_lr             =self.configs['CosineWarmRestart']['CWR_max_lr']
				CWR_max_lr             =self.configs['tl_case']['tl_5_2_lr']
				CWR_steps_per_epoch    =self.configs['CosineWarmRestart']['CWR_steps_per_epoch']
				CWR_lr_decay           =self.configs['CosineWarmRestart']['CWR_lr_decay']
				CWR_cycle_length       =self.configs['CosineWarmRestart']['CWR_cycle_length']
				CWR_cycle_mult_factor  =self.configs['CosineWarmRestart']['CWR_cycle_mult_factor']
				CWR_warmup_length      =self.configs['CosineWarmRestart']['CWR_warmup_length']
				CWR_warmup_mult_factor =self.configs['CosineWarmRestart']['CWR_warmup_mult_factor']
				
				callback1      =SGDRScheduler_warmup(
											min_lr=CWR_min_lr ,
											max_lr=CWR_max_lr,
											steps_per_epoch=CWR_steps_per_epoch,
											lr_decay=CWR_lr_decay,
											cycle_length=CWR_cycle_length,
											cycle_mult_factor=CWR_cycle_mult_factor,
											warmup_length=CWR_warmup_length,
											warmup_mult_factor=CWR_warmup_mult_factor)
				
				callback_all.append(callback1)
		
		if callback_signal == 1:
			if x_tune[0].shape[0] > 0:
				self.history_model=self.model.fit(
						x_tune, y_tune,
						epochs=self.epochs,
						batch_size=self.batch_size,
						verbose=self.training_verbose,
						callbacks=callback_all)

			else: 
				print('[TrainClass] No data to tune for week %s' % weeknum)
		
		if callback_signal == 0:
			if x_tune[0].shape[0] > 0:
				self.history_model=self.model.fit(
						x_tune, y_tune,
						epochs=self.epochs,
						batch_size=self.batch_size,
						verbose=self.training_verbose)
			else: 
				print('[TrainClass] No data to tune for week %s' % weeknum)
		
		self.tuned_save_hf5_name = os.path.join(self.save_results_dir, '%s_%s-%s.h5' % (self.ident, tuneindent, weeknum))

		self.model.save_weights(self.tuned_save_hf5_name)   

		timer.stop()



	def forget_model(self):
		del self.model
		K.clear_session()
