import numpy as np
from numpy import load
from typing import *

def modifydata(data_all, configs):
	print('---Modifying data---')
	train_X_past, 	train_X_future,		train_Y_future, \
	validate_X_past, validate_X_future, 	validate_Y_future, \
	test_X_past, 	test_X_future, 		test_Y_future = data_all

	print('Shapes before modification')
	print('train_X_past.shape', train_X_past.shape)
	print('train_X_future.shape', train_X_future.shape)
	print('train_Y_future.shape', train_Y_future.shape)

	train_X_past_new = []
	train_X_future_new = []
	train_Y_future_new = []

	validate_X_past_new = []
	validate_X_future_new = []
	validate_Y_future_new = []

	test_X_past_new = []
	test_X_future_new = []
	test_Y_future_new = []

	new_columnno_X 			= configs['data_processor']['columnno_X']
	new_columnno_X_future 	= configs['data_processor']['columnno_X_future']
	new_columnno_y 			= configs['data_processor']['columnno_y']
	og_columnno_X			= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71]
	og_columnno_X_future	= [0, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
	og_columnno_Y			= [46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71]

	indexes_coloumnno_X = [og_columnno_X.index(num) for num in new_columnno_X if num in og_columnno_X]
	indexes_coloumnno_X_future = [og_columnno_X_future.index(num) for num in new_columnno_X_future if num in og_columnno_X_future]
	indexes_coloumnno_Y = [og_columnno_Y.index(num) for num in new_columnno_y if num in og_columnno_Y]

	train_X_past_new = train_X_past[:, :, indexes_coloumnno_X]
	train_X_future_new = train_X_future[:, :, indexes_coloumnno_X_future]
	train_Y_future_new = train_Y_future[:, :, indexes_coloumnno_Y]

	validate_X_past_new = validate_X_past[:, :, indexes_coloumnno_X]
	validate_X_future_new = validate_X_future[:, :, indexes_coloumnno_X_future]
	validate_Y_future_new = validate_Y_future[:, :, indexes_coloumnno_Y]

	test_X_past_new = test_X_past[:, :, indexes_coloumnno_X]
	test_X_future_new = test_X_future[:, :, indexes_coloumnno_X_future]
	test_Y_future_new = test_Y_future[:, :, indexes_coloumnno_Y]

	train_X_past = train_X_past_new
	train_X_future = train_X_future_new
	train_Y_future = train_Y_future_new

	validate_X_past = validate_X_past_new
	validate_X_future = validate_X_future_new
	validate_Y_future = validate_Y_future_new

	test_X_past = test_X_past_new
	test_X_future = test_X_future_new
	test_Y_future = test_Y_future_new

	print('\nShapes after modification')
	print('train_X_past.shape', train_X_past.shape)
	print('train_X_future.shape', train_X_future.shape)
	print('train_Y_future.shape', train_Y_future.shape)

	alldata = [train_X_past, 	train_X_future,		train_Y_future,
			validate_X_past, validate_X_future, 	validate_Y_future,
			test_X_past, 	test_X_future, 		test_Y_future]

	return alldata

class Data_Reader_Class():
	def __init__(self,ident,configs):
		self.csv_files 				= configs['data']['input_EP_csv_files']
		print(self.csv_files, 'self.csv_files')

		self.configs 				= configs
		self.ident 					= ident
		self.data_split 			= self.configs['data']['data_split']
		self.DATAFOLDER 			= self.configs['data']['npy_save_path']
		self.IFDEBUG_CODE 			= self.configs['data_type_code']

	def __call__(self):
		if self.configs['data']['data_split']=='1a':
			data_all = self._readdata_1a()
		if self.configs['data']['data_split'] in ['2a','2b','3a']:
			data_all = self._readdata_2a_2b_3a()
		
		if self.configs['individual_runs']['use_individual_runs']==1:
			data_all = modifydata(data_all, self.configs)
		return data_all


	def _readdata_2a_2b_3a(self) -> List[np.ndarray]:
		"""Reads the data from the npy files and returns the data in the form of a list of numpy arrays
		
		Returns:
			list -- [train_X_past, train_X_future, train_Y_future, validate_X_past, validate_X_future, validate_Y_future, test_X_past, test_X_future, test_Y_future]

		train_X_future, validate_X_future, test_X_future are lists of numpy arrays. 
		
		Each numpy array is of shape (number of samples, number of future cells, number of features)
		"""	

		train_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][0]
		valid_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][1]
		test_col  = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][2]


		filename = self.DATAFOLDER + self.csv_files[train_col]
		train_X_past 		= load(filename+'%s_X_past_train.npy' %(self.IFDEBUG_CODE))
		train_X_future 		= load(filename+'%s_X_future_train.npy' %(self.IFDEBUG_CODE))
		train_Y_future	 	= load(filename+'%s_Y_future_train.npy' %(self.IFDEBUG_CODE))



		filename = self.DATAFOLDER + self.csv_files[valid_col]
		validate_X_past 	= load(filename+'%s_X_past_validate.npy' %(self.IFDEBUG_CODE))  
		validate_X_future 	= load(filename+'%s_X_future_validate.npy' %(self.IFDEBUG_CODE))
		validate_Y_future 	= load(filename+'%s_Y_future_validate.npy' %(self.IFDEBUG_CODE))  



		filename = self.DATAFOLDER + self.csv_files[test_col]
		test_X_past 		= load(filename+'%s_X_past_test.npy' %(self.IFDEBUG_CODE))  
		test_X_future 		= load(filename+'%s_X_future_test.npy' %(self.IFDEBUG_CODE))
		test_Y_future 		= load(filename+'%s_Y_future_test.npy' %(self.IFDEBUG_CODE))  

		alldata = [train_X_past, 	train_X_future,		train_Y_future,
				validate_X_past, validate_X_future, 	validate_Y_future,
				test_X_past, 	test_X_future, 		test_Y_future]

		return alldata

	def _readdata_1a(self) -> List[np.ndarray]:
		"""Reads the data from the npy files and returns the data in the form of a list of numpy arrays
		
		Returns:
			list -- [train_X_past, train_X_future, train_Y_future, validate_X_past, validate_X_future, validate_Y_future, test_X_past, test_X_future, test_Y_future]

		train_X_future, validate_X_future, test_X_future are lists of numpy arrays. 
		
		Each numpy array is of shape (number of samples, number of future cells, number of features)
		"""	

		train_col = self.configs['data']['data_split_%s' % self.configs['data']['data_split']][0]

		filename = self.DATAFOLDER + self.csv_files[train_col]
		train_X_past 		= load(filename+'%s_X_past_train.npy' %(self.IFDEBUG_CODE))
		train_X_future 		= load(filename+'%s_X_future_train.npy' %(self.IFDEBUG_CODE))
		train_Y_future	 	= load(filename+'%s_Y_future_train.npy' %(self.IFDEBUG_CODE))


		alldata = [train_X_past, 	train_X_future,		train_Y_future]

		return alldata