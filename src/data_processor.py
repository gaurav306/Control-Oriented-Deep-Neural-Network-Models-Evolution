import datetime as dt
import os
from typing import *

import numpy as np
import pandas as pd
from numpy import save

from src.data_processor_utils import DataClass_utils
from src.plots import PlotClass, calculate_metrics_nonprob, calculate_metrics_prob
from src.utils import fill_nan

import polars as pl     #<------------------------

class DataClass():
	def __init__(self,ident,configs):
		self.noise_sd 					= configs['data']['noise_sd']
		self.save_results_dir 			= configs['data']['save_results_dir']
		self.DEBUG_coloumns 			= configs['DEBUG_coloumns']
		self.data_type 					= configs['data_type_code']
		self.train_split 				= configs['data']['train_split']
		self.vaidate_test_split 		= configs['data']['vaidate_test_split']
		self.model_type_prob 			= configs['model_type_prob']
		self.csv_files 					= configs['data']['input_EP_csv_files']
		self.input_csv_path 			= configs['data']['input_csv_path']
		self.npy_save_path 				= configs['data']['npy_save_path']
		self.q 							= configs['quantiles']

		if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])

		self.save_data_scaledback_pred = os.path.join(
			self.save_results_dir, '%s_scaledback_predicted.npy' % (ident))
		self.save_data_scaledback_pastx = os.path.join(
			self.save_results_dir, '%s_scaledback_pastx.npy' % (ident))
		self.save_data_scaledback_futurex = os.path.join(
			self.save_results_dir, '%s_scaledback_futurex' % (ident))
		self.save_data_scaledback_futurey = os.path.join(
			self.save_results_dir, '%s_scaledback_futurey.npy' % (ident))

		self.ident = ident
		self.configs = configs
		
		#DataClass_utils.__init__(self, self.ident, self.configs)
		#DataClass_utils.make_coloumns(self, IFPRINT=False)
		self.data_precessor_utils = DataClass_utils(self.ident,self.configs)

		if self.configs['IF_MAKE_DATA']:
			self.make_data()
	

	def make_data(self):
		"""
		Make data from csv files and save them as npy files
		"""
		WAY_TO_SPLIT_DATA 	= self.configs['data']['data_split']

		if WAY_TO_SPLIT_DATA=='1a':
			train_col 		= self.configs['data']['data_split_1a'][0]

			filetoconvert 	= self.input_csv_path + self.csv_files[train_col]
			converted_ouput = self.npy_save_path +  self.csv_files[train_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '1a')

		if WAY_TO_SPLIT_DATA=='2a':
			train_col 		= self.configs['data']['data_split_2a'][0]
			valid_test_col 	= self.configs['data']['data_split_2a'][1]

			filetoconvert 	= self.input_csv_path + self.csv_files[train_col]
			converted_ouput = self.npy_save_path +  self.csv_files[train_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '1a')

			filetoconvert 	= self.input_csv_path + self.csv_files[valid_test_col]
			converted_ouput = self.npy_save_path +  self.csv_files[valid_test_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '2a')
		
		if WAY_TO_SPLIT_DATA=='2b':
			train_col 		= self.configs['data']['data_split_2a'][0]
			valid_test_col 	= self.configs['data']['data_split_2a'][1]

			filetoconvert 	= self.input_csv_path + self.csv_files[train_col]
			converted_ouput = self.npy_save_path +  self.csv_files[train_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '1a')

			filetoconvert 	= self.input_csv_path + self.csv_files[valid_test_col]
			converted_ouput = self.npy_save_path +  self.csv_files[valid_test_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '2b')
		
		if WAY_TO_SPLIT_DATA=='3a':
			train_valid_test_col 	= self.configs['data']['data_split_3a'][0]

			filetoconvert 	= self.input_csv_path + self.csv_files[train_valid_test_col]
			converted_ouput = self.npy_save_path +  self.csv_files[train_valid_test_col]
			self._csv_to_npy_func(filetoconvert,
								converted_ouput,
								SPLIT_INTO_FILES = '3a')
	

	def _csv_to_npy_func(self,NAME_OF_CSV, PATH_TO_NPY, SPLIT_INTO_FILES):
		
		print("-------------"+NAME_OF_CSV+"-------------")
		dataframe2  = pd.read_csv(NAME_OF_CSV, sep=',', header=0)
		
		dataframe_avg = dataframe2.copy()
		runn_avg_columns 	= self.configs['data_processor']['running_avg_col']
		runn_avg_win 		= self.configs['data_processor']['running_avg_window']
		for col in runn_avg_columns:
			dataframe_avg.iloc[:, col] = dataframe_avg.iloc[:, col].rolling(window=runn_avg_win).mean()
		dataframe_avg.fillna(dataframe2, inplace=True)
		dataframe = dataframe_avg.copy()
		
		dataframe = self.data_precessor_utils.change_64to32(dataframe)
		del dataframe['Date/Time']
		print('Original shape of the CSV file:',dataframe.shape)

		dataframe = dataframe.iloc[: , self.data_precessor_utils.columnno_X].copy() 
		print('Shape of the CSV file when X_past coloumns are copied:',dataframe.shape)
		
		#data for debug
		if self.data_type == '_DEBUG_xs':
			dataframe.drop(range(self.DEBUG_coloumns,len(dataframe)), axis=0, inplace=True)  #for testing,to reduce file size
		
		PATH_TO_NPY = PATH_TO_NPY + self.data_type

		#adding future coloumns to dataframe
		for i in range(dataframe.shape[1]):
			dataframe.iloc[:, i] = self.data_precessor_utils.scaledown( dataframe.iloc[:, i], min_x = self.data_precessor_utils.columnno_X_min[i], max_x = self.data_precessor_utils.columnno_X_max[i])
		for col in self.data_precessor_utils.columnno_X_future:
			dataframe[dataframe.iloc[:, col].name+str("_future")]=dataframe.iloc[:, col]

		X_past, X_future, Y_future   = self.data_precessor_utils.prepare_1Xpast_1Xfuture_1Yfuture_data( dataframe.values)
		
		start_runmain_dt = dt.datetime.now()
		
		print('Making future weather_with polars')
		print('if nan in future weather', np.isnan(X_future).any())
		for ix in range(0,X_future.shape[0]):
			ont_error = min(abs(np.random.normal(loc=0.0, scale=self.noise_sd)),0.02)
			for i in range(0,len(self.data_precessor_utils.columnno_X_future_noise)):
				if self.data_precessor_utils.columnno_X_future_noise[i] ==1:
					original = pl.Series(X_future[ix,:,i])
					pct_return_periods = 8
					percent_change = original.pct_change(n=pct_return_periods).fill_null(0)
					original_index = pl.Series([i/len(original) for i in range(0,len(original))])
					new_withnoise = original+(percent_change*ont_error*original_index)
					X_future[ix,:,i] = np.clip(new_withnoise,-1,1)
		end_training_dt = dt.datetime.now()
		print('Time taken to make future weather_with polars',end_training_dt-start_runmain_dt)
		print('if nan in future weather', np.isnan(X_future).any())
		X_future = fill_nan(X_future)
		print('if nan in future weather after fillnan', np.isnan(X_future).any())

		start_runmain_dt = dt.datetime.now()
		
		self.data_precessor_utils.split_in_3_or_2_or_1_save(
							PATH_TO_NPY+'_X_past', 
		       				df = X_past, 
							split_type= SPLIT_INTO_FILES,
							v_split=self.train_split,
							t_split=self.train_split+self.vaidate_test_split,
							m_split=0.5)
		
		self.data_precessor_utils.split_in_3_or_2_or_1_save(
							PATH_TO_NPY+'_X_future',
							df = X_future,
							split_type= SPLIT_INTO_FILES,
							v_split=self.train_split,
							t_split=self.train_split+self.vaidate_test_split,
							m_split=0.5)
		
		self.data_precessor_utils.split_in_3_or_2_or_1_save(
							PATH_TO_NPY+'_Y_future',
							df = Y_future,
							split_type= SPLIT_INTO_FILES,
							v_split=self.train_split,
							t_split=self.train_split+self.vaidate_test_split,
							m_split=0.5)
		
		end_training_dt = dt.datetime.now()
		print('Time taken to save all',end_training_dt-start_runmain_dt)
		

	def scaleback_after_prediction(self, pred, X_future, Y_future):
		#if self.control_future_cells == 6:
		#	X_future = self.data_precessor_utils.merge_X_future_back(X_future)
		#if self.control_future_cells == 1:
		#	X_future = X_future[0]
		IFPROB=1
		for i in range(pred.shape[2]):
			if self.model_type_prob == 'prob':
				pred[:,:,i,:] = self.data_precessor_utils.scaleup(
										pred[:,:,i,:],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
			elif self.model_type_prob == 'nonprob':
				pred[:,:,i] = self.data_precessor_utils.scaleup(
										pred[:,:,i],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
			Y_future[:,:,i] = self.data_precessor_utils.scaleup(
										Y_future[:,:,i],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
		for i in range(X_future.shape[2]):
			X_future[:,:,i] = self.data_precessor_utils.scaleup(
										X_future[:,:,i],
										min_x = self.data_precessor_utils.columnno_X_future_min[i],
										max_x = self.data_precessor_utils.columnno_X_future_max[i],
										a=-1,b=1)

		#print("X_future.shape: ",X_future.shape)
		#print("Y_future.shape: ",Y_future.shape)
		#print("pred.shape: ",        pred.shape)
		
		save(self.save_data_scaledback_pred,      		pred)
		#save(self.save_data_scaledback_futurex ,    X_future)
		#save(self.save_data_scaledback_futurey,     Y_future)

		# plot the prediction
		print('Plotting.......')
		if self.model_type_prob == 'nonprob':
			filesavename = os.path.join(self.save_results_dir, '%s_METRICS_nonprob_bytimestep.npy' % (self.ident))
			calculate_metrics_nonprob(Y_future, pred, filesavename)
		if self.model_type_prob == 'prob':
			filesavename = os.path.join(self.save_results_dir, '%s_METRICS_prob_bytimestep.npy' % (self.ident))
			calculate_metrics_prob(Y_future, pred, filesavename, self.q)
		
		#plots_all = PlotClass(self.ident,self.configs)
		#plots_all.plot_1zone(pred, X_future, Y_future)
		#plots_all.plot_bytimestep(Y_future, pred)
		#del plots_all


	def scaleback_after_prediction_tuning(self, pred, X_future, Y_future, weeknum, cords, tuneindent):
		#if self.control_future_cells == 6:
		#	X_future = self.data_precessor_utils.merge_X_future_back(X_future)
		#if self.control_future_cells == 1:
		#	X_future = X_future[0]
		IFPROB=1
		for i in range(pred.shape[2]):
			if self.model_type_prob == 'prob':
				pred[:,:,i,:] = self.data_precessor_utils.scaleup(
										pred[:,:,i,:],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
			elif self.model_type_prob == 'nonprob':
				pred[:,:,i] = self.data_precessor_utils.scaleup(
										pred[:,:,i],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
			Y_future[:,:,i] = self.data_precessor_utils.scaleup(
										Y_future[:,:,i],
										min_x = self.data_precessor_utils.columnno_Y_min[i],
										max_x = self.data_precessor_utils.columnno_Y_max[i],
										a=-1,b=1)
		for i in range(X_future.shape[2]):
			X_future[:,:,i] = self.data_precessor_utils.scaleup(
										X_future[:,:,i],
										min_x = self.data_precessor_utils.columnno_X_future_min[i],
										max_x = self.data_precessor_utils.columnno_X_future_max[i],
										a=-1,b=1)
		
		self.ident = self.ident + '_%s-%s' % (tuneindent, weeknum)

		save(self.save_data_scaledback_pred + '_%s-%s' % (tuneindent, weeknum),      		pred)
		if 'OG_FULL' in self.ident:
			save(self.save_data_scaledback_futurex + '_%s-%s' % (tuneindent, weeknum),    X_future)
			save(self.save_data_scaledback_futurey + '_%s-%s' % (tuneindent, weeknum),    Y_future)

		# plot the prediction
		print('Plotting.......')
		plots_all = PlotClass(self.ident,self.configs)
		plots_all.plot_1zone(pred, X_future, Y_future, cords)
		plots_all.plot_bytimestep(Y_future, pred)
		
		del plots_all

