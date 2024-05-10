from typing import *
import numpy as np
import pandas as pd
from numpy import save

class DataClass_utils():
	def __init__(self, ident, configs):
		
		self.n_past 					= configs['n_past']
		self.n_future 					= configs['n_future']
		self.future_data_col 			= configs['known_future_features']
		self.n_features_input 			= configs['known_past_features']
		self.n_features_output 			= configs['unknown_future_features']
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

		self.ident = ident
		self.configs = configs
		self.make_coloumns(IFPRINT = False)

	def make_coloumns(self,IFPRINT):
		def dfif(df,iflist):
			'''
			df is Excel file with max and min of all coloumns from Data csv file.
			This function returns min and max of coloumns in iflist from df.
			'''
			returnlist_min=[]
			returnlist_max=[]
			ifindf = [df['Index_num'].isin(iflist)]

			for i in range(len(df)):
				if ifindf[0][i]:
					returnlist_min.append(int(df['MIN_ALL'][i]) if int(df['MIN_ALL'][i] != 'None') else None)
					returnlist_max.append(int(df['MAX_ALL'][i]) if int(df['MAX_ALL'][i] != 'None') else None)
					"""
					if pd.isna(df['MIN_ALL'][i]) == False or df['MIN_ALL'][i] != 'None':
						returnlist_min.append(int(float(df['MIN_ALL'][i])))
					else:
						returnlist_min.append(None)
					if pd.isna(df['MAX_ALL'][i]) == False or df['MAX_ALL'][i] != 'None':
						returnlist_max.append(int(float(df['MAX_ALL'][i])))
					else:
						returnlist_max.append(None)
					"""

			#print('len(returnlist_min),len(returnlist_max)',len(returnlist_min),len(returnlist_max))
			#print('returnlist_min',returnlist_min)
			#print('returnlist_max',returnlist_max)
			return returnlist_min, returnlist_max
		
		self.columnno_X              = self.configs['data_processor']['columnno_X']
		self.columnno_X_future       = self.configs['data_processor']['columnno_X_future']
		self.columnno_X_future_noise = [1,1,1,*[0]*(len(self.columnno_X_future)-3)] 					#1 for noise, 0 for no noise
		self.columnno_Y              = self.configs['data_processor']['columnno_Y']
		print(len(self.columnno_X), len(self.columnno_X_future), len(self.columnno_X_future_noise), len(self.columnno_Y))
		print(self.n_features_input, self.n_features_output)
		self.n_features_input = len(self.columnno_X)
		self.n_features_output = len(self.columnno_Y)
		assert self.n_features_input == len(self.columnno_X), "n_features_input==len(self.columnno_X)"
		assert self.n_features_output == len(self.columnno_Y), "n_features_output==len(self.columnno_Y)"
		assert len(self.columnno_X_future) == len(self.columnno_X_future_noise), "len(self.columnno_X_future) == len(self.columnno_X_future_noise)"
		
		max_min_path = self.input_csv_path+self.configs['data_processor']['max_min_file']
		
		all_max_min  = pd.read_excel(max_min_path)

		self.columnno_X_min,        self.columnno_X_max        = dfif(all_max_min,self.columnno_X)
		self.columnno_X_future_min, self.columnno_X_future_max = dfif(all_max_min,self.columnno_X_future)
		self.columnno_Y_min,        self.columnno_Y_max        = dfif(all_max_min,self.columnno_Y)

		#make list as if all other coloumsn in data file are dropped except self.columnno_X
		self.columnno_X_future_new = []
		if_columnno_X_future = pd.DataFrame(self.columnno_X).isin(self.columnno_X_future)
		for i in range(0,len(self.columnno_X)):
			if if_columnno_X_future[0][i]:
				self.columnno_X_future_new.append(i)

		self.columnno_Y_new = []
		if_columnno_Y = pd.DataFrame(self.columnno_X).isin(self.columnno_Y)
		for i in range(0,len(self.columnno_X)):
			if if_columnno_Y[0][i]:
				self.columnno_Y_new.append(i)
				
		self.columnno_X_future,self.columnno_Y = self.columnno_X_future_new,self.columnno_Y_new

		if IFPRINT:
			print(len(self.columnno_X)    ,self.columnno_X)
			print(len(self.columnno_X)    ,[*range(0,len(self.columnno_X))])
			print(len(self.columnno_X_min),self.columnno_X_min)
			print(len(self.columnno_X_max),self.columnno_X_max)
			print()
			print(len(self.columnno_X_future)      ,self.columnno_X_future)
			print(len(self.columnno_X_future_min)  ,self.columnno_X_future_min)
			print(len(self.columnno_X_future_max)  ,self.columnno_X_future_max)
			print(len(self.columnno_X_future_noise),self.columnno_X_future_noise)
			print()
			print(len(self.columnno_Y)    ,self.columnno_Y)
			print(len(self.columnno_Y_min),self.columnno_Y_min)
			print(len(self.columnno_Y_max),self.columnno_Y_max)

	
	def split_in_3_or_2_or_1_save(self, filename, df, split_type, v_split, t_split, m_split):
		'''
		to split in 3
		0|--------train-------|(v_split)|--------validate--------|(t_split)|--------test--------|1
		to split in 2
		0|--------validate--------|(m_split)|--------test--------|1
		split_type = 3a >> split in 3
		split_type = 2a >> split in 2
		split_type = 2b >> split in 2_smaller
		split_type = 1a >> split in 1
		'''
		def save_npy(filename, df, split_type, v_split, t_split, m_split):
			if split_type=='3a':
				df1, df2, df3 = np.split(df,[round(df.shape[0]*(v_split)) , round(df.shape[0]*(t_split))])
				save(filename+'_train.npy',     df1)
				save(filename+'_validate.npy',  df2)
				save(filename+'_test.npy',  df3)
				print(df.shape,'>>',df1.shape,df2.shape,df3.shape)
			
			if split_type=='2a':
				df1, df2 = np.split(df,[round(df.shape[0]*(m_split))])
				save(filename+'_validate.npy',  df1)
				save(filename+'_test.npy',      df2)
				print(df.shape,'>>',df1.shape,df2.shape)
			
			if split_type=='2b':
				df_all = np.array_split(df, 8)
				df1 = np.concatenate([df_all[0], df_all[3]], axis=0)
				df2 = np.concatenate([df_all[4], df_all[7]], axis=0)
				save(filename+'_validate.npy',  df1)
				save(filename+'_test.npy',      df2)
				print(df.shape,'>>',df1.shape,df2.shape)
			
			if split_type=='1a':
				save(filename+'_train.npy',     df)
				print(df.shape,'>>',df.shape)

		if type(df) == np.ndarray:
			save_npy(filename, df, split_type, v_split, t_split, m_split)
			print('saved all npy with prefix %s' %filename)
		else:
			print('Error: control_future_cells is not 1 or 6')

	def change_64to32(self, dataframe: pd.DataFrame) -> pd.DataFrame:
		"""
		Change the data type of the dataframe from 64 to 32
		Parameters
		----------
		dataframe : pandas dataframe
			Input dataframe
		Returns
		-------
		dataframe : pandas dataframe
			Output dataframe
		"""
		# Change the float data type of the dataframe from 64 to 32
		float64_cols = list(dataframe.select_dtypes(include='float64'))
		dataframe[float64_cols] = dataframe[float64_cols].astype('float32')
		# Change the int data type of the dataframe from 64 to 32
		int64_cols = list(dataframe.select_dtypes(include='int64'))
		dataframe[int64_cols] = dataframe[int64_cols].astype('int32')
		return dataframe

	def scaledown(self,x_in:np.ndarray,min_x:float,max_x:float,a:float=-1,b:float=1) -> np.ndarray:  
		"""
		Scale down the x_in data to a,b
		# a,b is for example 0 to 1 or -1 to 1. b>a always   # max_x,min_x is for example 0 to 100
		Parameters
		----------
		x_in : array-like
			Input data to be scaled down
		min_x : float
			Minimum value of the input data
		max_x : float
			Maximum value of the input data
		a : float
			Minimum value of the output data
		b : float
			Maximum value of the output data
		Returns
		-------
		x_down : array-like
			Scaled down data
		"""
		try:
			if min_x == None or max_x == None:
				return x_in
			else:
				if isinstance(x_in, list):
					x = x_in.copy()
					x = np.asarray(x)
				else:
					x = x_in 
				x_down = a + ((x-min_x)*(b-a))/(max_x-min_x)
				# clip any values outside of the range a,b
				x_down[x_down > b] = b   
				x_down[x_down < a] = a
				return x_down
		except Exception as e:
			print('Error in _scaledown',e)
			return x_in

	def scaleup(self,x_in:np.ndarray,min_x:float,max_x:float,a:float=-1,b:float=1)->np.ndarray:
		"""
		Scale up the x_in data to a,b
		# a,b is for example 0 to 1 or -1 to 1. b>a always   # max_x,min_x is for example 0 to 100
		Parameters
		----------
		x_in : array-like
			Input data to be scaled up
		min_x : float
			Minimum value of the input data
		max_x : float
			Maximum value of the input data
		a : float
			Minimum value of the output data
		b : float
			Maximum value of the output data
		Returns
		-------
		x_up : array-like
			Scaled up data
		"""
		try:
			if min_x == None or max_x == None:
				return x_in
			else:
				if isinstance(x_in, list):
					x = x_in.copy()
					x = np.asarray(x)
				else:
					x = x_in 
				x_up = min_x + ((x-a)*(max_x-min_x))/(b-a)
				return x_up
		except Exception as e:
			print('Error in _scaleup',e)
			return x_in

	def prepare_1Xpast_1Xfuture_1Yfuture_data(self, datanp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Prepare the data for 1 past and 2 future data, Xpast, Xfuture, Yfuture
		Parameters
		----------
		datanp : numpy array
			Input data
		Returns
		-------
		X_past : numpy array
			1 past data
		X_future : numpy array
			X future data
		Y_future : numpy array
			Y future data
		"""
		print('Before def _timeseries_to_supervised_learning_prepare_X_Y_data:  ', datanp.shape)
		X_data, y_data = self.timeseries_to_supervised_learning_prepare_X_Y_data(datanp)
		print('After def _timeseries_to_supervised_learning_prepare_X_Y_data:  ', X_data.shape, y_data.shape)
		X_past = X_data[:,:,[*range(0,self.n_features_input)]].copy()
		X_future = y_data[:,:,[*range(self.n_features_input, self.n_features_input+self.future_data_col)]].copy()
		Y_future = y_data[:,:,self.columnno_Y].copy()
		print('After def _prepare_1Xpast_1Xfuture_1Yfuture_data:  ',
		      X_past.shape, X_future.shape, Y_future.shape)
		X_past = np.array(X_past)
		X_future = np.array(X_future)
		Y_future = np.array(Y_future)
		return X_past, X_future, Y_future



	def timeseries_to_supervised_learning_prepare_X_Y_data(self,series:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
		"""Convert series to supervised learning. 
		Parameters
		----------
		series : ndarray of shape (n_samples, n_features)
			Training data
		Returns
		-------
		X : ndarray of shape (n_samples, n_timesteps, n_features)
			Feature set
		y : ndarray of shape (n_samples, n_timesteps, n_features)
			Target set
		"""
		X, Y = list(), list()
		for window_start in range(len(series)):
			past_end = window_start + self.n_past
			future_end = past_end + self.n_future
			if future_end > len(series):
				break
			past, future = series[window_start:past_end, :], series[past_end:future_end, :]
			X.append(past)
			Y.append(future)
		X = np.array(X)
		Y = np.array(Y)
		return X, Y
	
	def merge_X_future_back(self, X_future):
		"""
		merge X_future1, X_future2, X_future3, X_future4, X_future5, X_future6 back to X_future
		Parameters
		----------
		X_future : numpy array
			Input data
		Returns
		-------
		X_future : numpy array
			merged X_future numpy arrays
		"""
		X_future = X_future.copy()
		xn = X_future[0].shape[2]  # number of features in X_future

		X_future_merged = np.concatenate((X_future[0], 
				    					X_future[1][:,:,[xn-1]],
										X_future[2][:,:,[xn-1]],
										X_future[3][:,:,[xn-1]],
										X_future[4][:,:,[xn-1]],
										X_future[5][:,:,[xn-1]]), 
										axis=2)

		print('After def merge_X_future_back:  ', X_future_merged.shape)
		return X_future_merged