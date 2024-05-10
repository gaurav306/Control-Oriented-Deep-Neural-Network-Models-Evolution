import multiprocessing
import os
import csv
import numpy as np
from src.configs.configs_init import read_write_yaml
from src.data_reader import Data_Reader_Class
from src.training_pipiline import training_pipeline_1_2, training_pipeline_6_7, training_pipeline_8_9, training_pipeline_3_part1, training_pipeline_3_part2, just_testing_pipeline
from src.training_pipiline_finetune import transfer_learning_pipeline

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

class Main_Run_Class():
	def __init__(self,ident,configs, modelpath=None):
		self.csv_files 	= configs['data']['input_EP_csv_files']

		self.configs 		= configs
		self.ident 			= ident
		self.save_results_dir   =configs['data']['save_results_dir']

		if not os.path.exists(configs['data']['save_results_dir']): os.makedirs(configs['data']['save_results_dir'])

		if self.configs['training']['IF_TRAIN_AT_ALL']:
			if self.configs['training']['training_type'] in [1,2]: 
				self.training_type_1_2()                      # 1 is basic training. #2 is training with early stopping measures and restarts
			if self.configs['training']['training_type'] == 3:
				self.training_type_3()                        # 3 is training with multiple runs for small number of epochs and then training with best run for more epochs
			if self.configs['training']['training_type'] == 4:
				self.transfer_learning_case(modelpath)        # 4 is transfer learning case where tuning data is increased from 0 to 26 weeks
			if self.configs['training']['training_type'] == 41:
				self.transfer_learning_case_MONTHLY(modelpath)# 41 is transfer learning case where tuning data is increased from 0 to 6 months (monthly) keeping amount of data each 6 times equal
			if self.configs['training']['training_type'] == 42:
				self.transfer_learning_case_WEEKLEY(modelpath)# 41 is transfer learning case where tuning data is increased from 0 to 6 months (monthly) keeping amount of data each 6 times equal
			if self.configs['training']['training_type'] == 5:
				self.just_testing(modelpath)                  # 5 is just testing case where model is loaded and tested on test data
			if self.configs['training']['training_type'] in [6,7]: 
				self.training_type_6_7()                      #decoder paper # 6 is basic training. #7 is training with early stopping measures and restarts
			if self.configs['training']['training_type'] in [8,9]: 
				self.training_type_8_9()                      #attn_scores # 8 is basic training. #9 is training with early stopping measures and restarts

		else:
			print("----------#############--------configs['training']['IF_TRAIN_AT_ALL'] set to 0 so cannot run training at all")


	def just_testing(self, modelpath):
		print('just testing')
		data_all = Data_Reader_Class(self.ident, self.configs)()
		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())	
		print()

		test_data_all = []
		start = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']
		end = data_all[0].shape[0]

		for data in data_all:
			if type(data) == np.ndarray:
				test_data_all.append(data[start:end,:])
			elif type(data) == list:
				test_data_all.append([d[start:end,:] for d in data])

		print('-----------modelpath:-:'+modelpath)

		self.configs['training']['IF_MULTIPROCESS'] = 0
		
		if self.configs['training']['IF_MULTIPROCESS'] == 1:
			MultiprocessingWindow(just_testing_pipeline, (self.ident, self.configs, test_data_all, modelpath, 0, [start, end], 'week'))()
		if self.configs['training']['IF_MULTIPROCESS'] == 0:
			just_testing_pipeline(self.ident, self.configs, test_data_all, modelpath, 0, [start, end], 'week')


	def transfer_learning_case(self, modelpath):
		""""""
		# tuning data goes from none to 1 week, 2 weeks, 3 weeks..... 26 weeks
		data_all = Data_Reader_Class(self.ident, self.configs)()

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())	
		""""""
		print()
		start_sim = self.configs['tl_case']['start']
		end_sim = self.configs['tl_case']['end']
		for i in range (start_sim, end_sim):                   #<---------------------------------change this to 52 for full training
			tune_data_all = []
			test_data_all = []
			end = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']                  #17472 for 15 mins data = 96*7*26
			start = (24 * 26 * 7 * self.configs['data']['values_in_an_hour']) - ((i) * 24 * 7 * self.configs['data']['values_in_an_hour'])
			
			""""""
			print()
			print('%s week of tuning data, start:%s and end:%s' % (i, start, end)	)
			for data in data_all:
				if type(data) == np.ndarray:
					tune_data_all.append(data[start:end,:])
				elif type(data) == list:
					tune_data_all.append([d[start:end,:] for d in data])
			
			start = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']
			end = data_all[0].shape[0]

			for data in data_all:
				if type(data) == np.ndarray:
					test_data_all.append(data[start:end,:])
				elif type(data) == list:
					test_data_all.append([d[start:end,:] for d in data])
			""""""

			print('-----------modelpath-----------------'+modelpath)
			with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
				file.write("False")
			flag = 1
			while True:
				print("Training attempt %s" % (flag))
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'week'))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'week')

				with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "r") as file:
					restart_training = file.readline().strip()
				if restart_training == "True":
					with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
						file.write("False")		
					print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
					flag += 1
					continue # Restart the loop to begin training anew
				else:
					print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
					break # Training completed normally, so exit the loop.

	def transfer_learning_case_MONTHLY(self, modelpath):    # if self.configs['training']['training_type'] == 41:
		# tuning data goes from none to 1st month, 2nd month, 3rd month..... 6th month
		data_all = Data_Reader_Class(self.ident, self.configs)()

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())	
		print()
		for i in range (0,6):                   #<---------------------------------change this to 52 for full training
			tune_data_all = []

			end = 17472 - (i * 2912)                 #17472 for 15 mins data = 96*7*26
			start = 17472 - ((i+1) * 2912)

			print()
			print('%s month of tuning data, start:%s and end:%s' % (i, start, end)	)
			for data in data_all:
				if type(data) == np.ndarray:
					tune_data_all.append(data[start:end,:])
				elif type(data) == list:
					tune_data_all.append([d[start:end,:] for d in data])
			
			test_data_all = []
			start = 17472
			end = data_all[0].shape[0]

			for data in data_all:
				if type(data) == np.ndarray:
					test_data_all.append(data[start:end,:])
				elif type(data) == list:
					test_data_all.append([d[start:end,:] for d in data])
			
			"""
			if i == 0:
				modelpath = modelpath
			else:
				modelpath = os.path.join(self.save_results_dir, '%s_week-%s.h5' % (self.ident, i-1))
			"""
			print('-----------modelpath-----------------'+modelpath)
			'''
			if self.configs['training']['IF_MULTIPROCESS'] == 1:
				MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'month'))()
			if self.configs['training']['IF_MULTIPROCESS'] == 0:
				transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'month')	

			'''
			with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
				file.write("False")
			flag = 1
			while True:
				print("Training attempt %s" % (flag))
				'''
				if flag == 3:
					if self.configs['training']['IF_MULTIPROCESS'] == 1:
						MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'month'))()
					if self.configs['training']['IF_MULTIPROCESS'] == 0:
						transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'month')	
					with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
						file.write("False")	
					break				
				'''
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'month'))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'month')

				with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "r") as file:
					restart_training = file.readline().strip()
				if restart_training == "True":
					with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
						file.write("False")		
					print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
					flag += 1
					continue # Restart the loop to begin training anew
				else:
					print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
					break # Training completed normally, so exit the loop.


	def transfer_learning_case_WEEKLEY(self, modelpath):    # if self.configs['training']['training_type'] == 42:
		# tuning data goes from none to 1st WEEK, 2nd WEEK, 3rd WEEK..... 26th WEEK ONLY
		data_all = Data_Reader_Class(self.ident, self.configs)()

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())	
		print()
		for i in range (0,26):                   #<---------------------------------change this to 52 for full training
			tune_data_all = []

			end = 17472 - (i * 672)                 #17472 for 15 mins data = 96*7*26
			start = 17472 - ((i+1) * 672)

			print()
			print('%s w of tuning data, start:%s and end:%s' % (i, start, end)	)
			for data in data_all:
				if type(data) == np.ndarray:
					tune_data_all.append(data[start:end,:])
				elif type(data) == list:
					tune_data_all.append([d[start:end,:] for d in data])
			
			test_data_all = []
			start = 17472
			end = data_all[0].shape[0]

			for data in data_all:
				if type(data) == np.ndarray:
					test_data_all.append(data[start:end,:])
				elif type(data) == list:
					test_data_all.append([d[start:end,:] for d in data])
			
			"""
			if i == 0:
				modelpath = modelpath
			else:
				modelpath = os.path.join(self.save_results_dir, '%s_week-%s.h5' % (self.ident, i-1))
			"""
			print('-----------modelpath-----------------'+modelpath)
			'''
			if self.configs['training']['IF_MULTIPROCESS'] == 1:
				MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'weekly'))()
			if self.configs['training']['IF_MULTIPROCESS'] == 0:
				transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'weekly')	

			'''
			with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
				file.write("False")
			flag = 1
			while True:
				print("Training attempt %s" % (flag))
				'''
				if flag == 3:
					if self.configs['training']['IF_MULTIPROCESS'] == 1:
						MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'weekly'))()
					if self.configs['training']['IF_MULTIPROCESS'] == 0:
						transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 0, 'weekly')	
					with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
						file.write("False")	
					break				
				'''
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					MultiprocessingWindow(transfer_learning_pipeline, (self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'weekly'))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					transfer_learning_pipeline(self.ident, self.configs, tune_data_all, test_data_all, modelpath, i, [start, end], 1, 'weekly')

				with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "r") as file:
					restart_training = file.readline().strip()
				if restart_training == "True":
					with open(self.configs['data']['save_results_dir'] + self.ident + "restart_training_flag.txt", "w") as file:
						file.write("False")		
					print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
					flag += 1
					continue # Restart the loop to begin training anew
				else:
					print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
					break # Training completed normally, so exit the loop.


	def training_type_1_2(self):     #take input as number of simulation same runs
		num_of_run = self.configs['training']['num_of_same_run']
		all_return_loss 		= []
		all_return_vloss 		= []
		all_return_diff_loss 	= []

		data_all = Data_Reader_Class(self.ident, self.configs)()

		if self.configs['data']['half_training']:		
			half_data = []
			start = 0
			end = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']
			for data in data_all:
				if type(data) == np.ndarray:
					half_data.append(data[start:end,:])
				elif type(data) == list:
					half_data.append([d[start:end,:] for d in data])
			data_all = half_data

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())
		
		for ix in range(0,num_of_run):
			ident_run=self.ident+'_runNUMBER-'+str(ix+1)
			print('----------------------------'+ident_run)

			if self.configs['training']['training_type'] == 1:
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					return_v = MultiprocessingWindow_ret(training_pipeline_1_2, (ident_run, self.configs, data_all))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					return_v = training_pipeline_1_2(ident_run, self.configs, data_all)
				
				print('<<<<<<<<<<<< return_v >>>>>>>>>>>>>>>>',return_v)
			
			if self.configs['training']['training_type'] == 2:
				with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
					file.write("False")
				flag = 1
				while True:
					print("Training attempt %s" % (flag))
					if flag > self.configs['CustomEarlyStopping']['max_training_attempts']:
						raise Exception("Training failed after %s attempts. Exiting." % (flag))
					
					if self.configs['training']['IF_MULTIPROCESS'] == 1:
						MultiprocessingWindow(training_pipeline_1_2, (ident_run, self.configs, data_all))()
					if self.configs['training']['IF_MULTIPROCESS'] == 0:
						training_pipeline_1_2(ident_run, self.configs, data_all)
					
					with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "r") as file:
						restart_training = file.readline().strip()
					if restart_training == "True":
						with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
							file.write("False")		
						print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
						flag += 1
						continue # Restart the loop to begin training anew
					else:
						print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
						break # Training completed normally, so exit the loop.
			try:
				if self.configs['training']['LOAD_or_TRAIN_MODEL'] == 'TRAIN':
					yaml_name = os.path.join(self.configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident_run))
					print('Reading results from yaml file')
					results_configs = read_write_yaml(filename=yaml_name, mode='r', data_yaml=None)
					all_return_loss.append(results_configs['results']['last loss'])
					all_return_vloss.append(results_configs['results']['last val_loss'])
					all_return_diff_loss.append(results_configs['results']['Diff of loss and val_loss'])
					print(_average(all_return_loss), _average(all_return_vloss), _average(all_return_diff_loss))
			except:
				pass



	def training_type_6_7(self):     #take input as number of simulation same runs
		num_of_run = self.configs['training']['num_of_same_run']
		all_return_loss 		= []
		all_return_vloss 		= []
		all_return_diff_loss 	= []

		data_all = Data_Reader_Class(self.ident, self.configs)()

		if self.configs['data']['half_training']:		
			half_data = []
			start = 0
			end = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']
			for data in data_all:
				if type(data) == np.ndarray:
					half_data.append(data[start:end,:])
				elif type(data) == list:
					half_data.append([d[start:end,:] for d in data])
			data_all = half_data

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())
		
		for ix in range(0,num_of_run):
			run_number = self.configs['training']['run_number']
			ident_run=self.ident+'_runNUMBER-'+str(ix+1)+'_of_'+str(run_number)
			print('----------------------------'+ident_run)

			if self.configs['training']['training_type'] == 6:
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					return_v = MultiprocessingWindow_ret(training_pipeline_6_7, (ident_run, self.configs, data_all))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					return_v = training_pipeline_6_7(ident_run, self.configs, data_all)
				
				print('<<<<<<<<<<<< return_v >>>>>>>>>>>>>>>>',return_v)
			
			if self.configs['training']['training_type'] == 7:
				with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
					file.write("False")
				flag = 1
				while True:
					print("Training attempt %s" % (flag))
					if flag > self.configs['CustomEarlyStopping']['max_training_attempts']:
						raise Exception("Training failed after %s attempts. Exiting." % (flag))
					
					if self.configs['training']['IF_MULTIPROCESS'] == 1:
						MultiprocessingWindow(training_pipeline_6_7, (ident_run, self.configs, data_all))()
					if self.configs['training']['IF_MULTIPROCESS'] == 0:
						training_pipeline_6_7(ident_run, self.configs, data_all)
					
					with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "r") as file:
						restart_training = file.readline().strip()
					if restart_training == "True":
						with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
							file.write("False")		
						print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
						flag += 1
						continue # Restart the loop to begin training anew
					else:
						print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
						break # Training completed normally, so exit the loop.
			try:
				if self.configs['training']['LOAD_or_TRAIN_MODEL'] == 'TRAIN':
					yaml_name = os.path.join(self.configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident_run))
					print('Reading results from yaml file')
					results_configs = read_write_yaml(filename=yaml_name, mode='r', data_yaml=None)
					all_return_loss.append(results_configs['results']['last loss'])
					all_return_vloss.append(results_configs['results']['last val_loss'])
					all_return_diff_loss.append(results_configs['results']['Diff of loss and val_loss'])
					print(_average(all_return_loss), _average(all_return_vloss), _average(all_return_diff_loss))
			except:
				pass



	def training_type_8_9(self):     #take input as number of simulation same runs
		print('Startttttttt training_type_8_9: ', str(self.configs['training']['training_type']))

		num_of_run = self.configs['training']['num_of_same_run']
		all_return_loss 		= []
		all_return_vloss 		= []
		all_return_diff_loss 	= []

		data_all = Data_Reader_Class(self.ident, self.configs)()

		if self.configs['data']['half_training']:		
			half_data = []
			start = 0
			end = 24 * 26 * 7 * self.configs['data']['values_in_an_hour']
			for data in data_all:
				if type(data) == np.ndarray:
					half_data.append(data[start:end,:])
				elif type(data) == list:
					half_data.append([d[start:end,:] for d in data])
			data_all = half_data

		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())
		
		for ix in range(0,num_of_run):
			ident_run=self.ident+'_runNUMBER-'+str(ix+1)
			print('----------------------------'+ident_run)

			if self.configs['training']['training_type'] == 8:
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					return_v = MultiprocessingWindow_ret(training_pipeline_8_9, (ident_run, self.configs, data_all))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					return_v = training_pipeline_8_9(ident_run, self.configs, data_all)
				
				print('<<<<<<<<<<<< return_v >>>>>>>>>>>>>>>>',return_v)
			
			if self.configs['training']['training_type'] == 9:
				with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
					file.write("False")
				flag = 1
				while True:
					print("Training attempt %s" % (flag))
					if flag > self.configs['CustomEarlyStopping']['max_training_attempts']:
						raise Exception("Training failed after %s attempts. Exiting." % (flag))
					
					if self.configs['training']['IF_MULTIPROCESS'] == 1:
						MultiprocessingWindow(training_pipeline_8_9, (ident_run, self.configs, data_all))()
					if self.configs['training']['IF_MULTIPROCESS'] == 0:
						training_pipeline_8_9(ident_run, self.configs, data_all)
					
					with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "r") as file:
						restart_training = file.readline().strip()
					if restart_training == "True":
						with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
							file.write("False")		
						print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
						flag += 1
						continue # Restart the loop to begin training anew
					else:
						print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
						break # Training completed normally, so exit the loop.
			try:
				if self.configs['training']['LOAD_or_TRAIN_MODEL'] == 'TRAIN':
					yaml_name = os.path.join(self.configs['save_models_dir'], '%s_configs_with_results.yaml' % (ident_run))
					print('Reading results from yaml file')
					results_configs = read_write_yaml(filename=yaml_name, mode='r', data_yaml=None)
					all_return_loss.append(results_configs['results']['last loss'])
					all_return_vloss.append(results_configs['results']['last val_loss'])
					all_return_diff_loss.append(results_configs['results']['Diff of loss and val_loss'])
					print(_average(all_return_loss), _average(all_return_vloss), _average(all_return_diff_loss))
			except:
				pass




	def training_type_3(self):     #take input as number of simulation same runs
		
		self.configs['if_model_image'] = 0
		self.configs['training']['ifLRS'] = 0
		save_loss_allrun  = os.path.join(self.configs['data']['save_results_dir'], '%s_loss_allrun.csv' % (self.ident))
		data_all = Data_Reader_Class(self.ident, self.configs)()
		
		print('Check for nan in data')
		for data in data_all:
			if type(data) == np.ndarray:
				print(data.shape, np.isnan(data).any())
			elif type(data) == list:
				for d in data:
					print(d.shape, np.isnan(d).any())
		"""
		Part 1 where training is run for number of prelim_runs and loss is recorded for all runs. Model weights and 
		optimizer states are saved for each run.
		"""
		for ix in range(0, self.configs['training']['prelim_runs']):
			ident_run=self.ident+'_prelim_run_number-'+str(ix+1)
			print('----------------------------'+ident_run)
			
			with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
				file.write("False")
			
			while True:
				if self.configs['training']['IF_MULTIPROCESS'] == 1:
					MultiprocessingWindow(training_pipeline_3_part1, (ident_run, self.configs, data_all, save_loss_allrun))()
				if self.configs['training']['IF_MULTIPROCESS'] == 0:
					training_pipeline_3_part1(ident_run, self.configs, data_all, save_loss_allrun)
				
				with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "r") as file:
					restart_training = file.readline().strip()
				if restart_training == "True":
					with open(self.configs['data']['save_results_dir'] + ident_run + "restart_training_flag.txt", "w") as file:
						file.write("False")					
					print("restart_training_flag.txt says: %s, so restart the loop" % (restart_training))
					continue
				else:
					print("restart_training_flag.txt says: %s, meaning the training completed normally, so exit the loop" % (restart_training))
					break
		
		"""
		Part 2 where the best model is selected from the prelim runs and trained for the final run.
		"""
		self.configs['training']['ifLRS'] = 1
		with open(save_loss_allrun, 'r') as f:
			reader = csv.reader(f)
			loss_values = [float(row[1]) for row in reader]
			print('MAPE values of all prelim models', loss_values)
		
		# Find the index of the best MAPE
		best_run = loss_values.index(min(loss_values))
		best_model_ident = self.ident+'_prelim_run_number-'+str(best_run+1)
		print('Best prelim model is: %s' % (best_model_ident))

		ident_run=self.ident+'_best_run'
		print('----------------------------'+ident_run)
		if self.configs['training']['IF_MULTIPROCESS'] == 1:
			MultiprocessingWindow(training_pipeline_3_part2, (ident_run, self.configs, data_all, best_model_ident))()
		if self.configs['training']['IF_MULTIPROCESS'] == 0:
			training_pipeline_3_part2(ident_run, self.configs, data_all, best_model_ident)
		
		for ix in range(0, self.configs['training']['prelim_runs']):
			os.remove(os.path.join(self.configs['data']['save_results_dir'], '%s_prelim_run_number-%s_modelcheckpoint_best.h5' % (self.ident, ix+1)))
			os.remove(os.path.join(self.configs['data']['save_results_dir'], '%s_prelim_run_number-%s_optimizer_state.pkl' % (self.ident, ix+1)))
			#os.remove(os.path.join(self.configs['data']['save_results_dir'], '%s_prelim_run_number-%s.csv' % (self.ident, ix+1)))



def _average(lst):
	return sum(lst) / len(lst)

class MultiprocessingWindow():
	'''
	this class is to create a multiprocessing window that can run a function
	and clear the memory
	'''
	def __init__(self, function_to_run, its_args):
		self.function_to_run = function_to_run
		self.its_args = its_args
		print("Multiprocessing class created")

	def __call__(self):
		'''
		this function is to run a function in a separate process and clear the memory
		'''
		try:
			multiprocessing.set_start_method('spawn', force=True)
			print("Multiprocessing window spawned")
		except RuntimeError:
			pass
		p = multiprocessing.Process(target=self.function_to_run,args=self.its_args)
		p.start()
		p.join()


class MultiprocessingWindow_ret():
    '''
    this class is to create a multiprocessing window that can run a function
    and clear the memory AND RETURN ITS VALUE
    '''
    def __init__(self, function_to_run, its_args):
        self.function_to_run = function_to_run
        self.its_args = its_args
        print("Multiprocessing class created")
    
    @staticmethod
    def wrapper(queue, func, args):
        ret_val = func(*args)
        queue.put(ret_val)

    def __call__(self):
        '''
        this function is to run a function in a separate process and clear the memory
        '''
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("Multiprocessing window spawned")
        except RuntimeError:
            pass
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.wrapper, args=(q, self.function_to_run, self.its_args))
        p.start()
        result = q.get()
        p.join()
        return result
