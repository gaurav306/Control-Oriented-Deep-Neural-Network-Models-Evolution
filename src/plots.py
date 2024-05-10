import os
import matplotlib.pyplot as plt           #<---------------------------
import numpy as np
import csv
import cv2
from sklearn.metrics import r2_score
from src.utils import coefficient_of_variation_root_mean_square_error, normalized_mean_bias_error, root_mean_square_error, mean_absolute_error, mean_squared_error
from src.utils import crps, pinball_loss

def calculate_metrics_nonprob(y_true, y_pred, savename):
	metrics = np.empty((y_true.shape[1], y_true.shape[2], 5))  # 5 metrics
	for i in range(y_true.shape[1]):
		for j in range(y_true.shape[2]):
			metrics[i, j, 0] = coefficient_of_variation_root_mean_square_error(y_true[:, i, j], y_pred[:, i, j])
			metrics[i, j, 1] = mean_absolute_error(y_true[:, i, j], y_pred[:, i, j])
			metrics[i, j, 2] = mean_squared_error(y_true[:, i, j], y_pred[:, i, j])
			metrics[i, j, 3] = root_mean_square_error(y_true[:, i, j], y_pred[:, i, j])
			metrics[i, j, 4] = r2_score(y_true[:, i, j], y_pred[:, i, j])
	np.save(savename, metrics)
	return metrics

def calculate_metrics_prob(y_true, y_pred, savename, q):
    metrics = np.empty((y_true.shape[1], y_true.shape[2], 2))  # 2 metrics: CRPS and Pinball loss
    for i in range(y_true.shape[1]):
        for j in range(y_true.shape[2]):
            metrics[i, j, 0] = crps(y_true[:, i, j], y_pred[:, i, j, :], q)
            metrics[i, j, 1] = pinball_loss(y_true[:, i, j], y_pred[:, i, j, :], q)
    np.save(savename, metrics)
    return metrics


class PlotClass():
	def __init__(self, ident, configs):
		self.future_data_col    =configs['known_future_features']
		self.n_past             = configs['n_past']
		self.n_features_input   = configs['known_past_features']
		self.all_layers_neurons = configs['all_layers_neurons']
		self.all_layers_dropout = configs['all_layers_dropout']
		self.n_future           = configs['n_future']
		self.n_features_output  = configs['unknown_future_features']
		self.noise_sd			= configs['data']['noise_sd']
		self.save_results_dir   = configs['data']['save_results_dir']
		#self.train_split		= configs['data']['train_split']
		#self.vaidate_test_split	= configs['data']['vaidate_test_split']
		#self.zone_number 		= configs['data_processor']['zone_number']
		self.model_type_prob 	= configs['model_type_prob']
		self.ident 				= ident
		self.configs 			= configs

	def plot_5zone_single_ts(self, pred, X_future, Y_future, IFPROB=False, ts=0):
		#ts=0
		start = 0
		end   = len(pred)
		fig = plt.figure(dpi=150)
		widthofpng=25
		
		fig.set_size_inches(widthofpng, pred.shape[2]*2.5, True)
		plt.subplots_adjust(hspace=0.05)

		for i in range(pred.shape[2]):
			ax = fig.add_subplot(pred.shape[2],1,i+1)
			ax.plot(range(start,end,1),Y_future[start:end,ts,i], label='Actual IAT', color='black', linewidth = 2, linestyle='dashed', alpha=0.6)
			if self.model_type_prob == 'prob':
				ax.plot(range(start,end,1),pred[start:end,ts,i,2],label='Predicted IAT_mean  [P50]', color='red', linewidth = 1, alpha=1)
				ax.fill_between(range(start,end,1), 
							   pred[start:end,ts,i,0], 
							   pred[start:end,ts,i,4],
							   color = 'green', label = '95% confidence interval [P05,P95]', alpha=0.1,
							   linewidth = 0)

				ax.fill_between(range(start,end,1), 
							   pred[start:end,ts,i,1], 
							   pred[start:end,ts,i,3],
							   color = 'green', label = '90% confidence interval [P10,P90]', alpha=0.2,
							   linewidth = 0)    
			elif self.model_type_prob == 'nonprob':
				ax.plot(range(start,end,1),pred[start:end,ts,i],label='Predicted IAT_mean  [P50]', color='red', linewidth = 1, alpha=1)

			#ax.plot(range(start,end,1),Y_future[:,ts,i], label='Actual IAT', color='black', linewidth = 2, linestyle='dashed', alpha=0.3)
			#ax.plot(range(start,end,1),X_future[:,ts,12+i], color='black',linewidth = 1, alpha=0.3)    #, label='Heating Setpoint'
			#ax.plot(X_future[:,ts,16+i],  color='gray',linewidth = 1, alpha=0.3)  #, label='Cooling Setpoint'
			ax.legend(loc=1)
			if i<5:    
				ax.axis(ymin= 10,ymax=30)
				ax.plot(range(start,end,1),X_future[start:end,ts,11]*13, color='black', linewidth = 0.3,marker='.', markersize=7)
				ax.plot(range(start,end,1),X_future[start:end,ts,12]*13, color='black', linewidth = 0.3,marker='.', markersize=7)
				ax.plot(range(start,end,1),X_future[start:end,ts,13], color='black', linewidth = 1, alpha=0.3)
				ax.plot(range(start,end,1),X_future[start:end,ts,24], color='black', linewidth = 1, alpha=0.3)
				
			ax.margins(x=0)
			#plt.xticks([])
			if i==0:
				ax.title.set_text('%s-step prediction over time || Ident <%s>' % (str(ts+1),str(self.ident)))
				
			#ax.set_xticks([0,2500,5000])
			#ax.set_xticks(range(0,5001,500))
			
			#if i!=0:
			#	ax.set_yticklabels([])
			ax.set_ylabel("Indoor Air Temperature\nSPACE%s-1 (IAT(°C))" %(str(i+1)), fontsize=10)

		self.save_prediction_png	= os.path.join(self.save_results_dir, '%s_ts_%s_prediction.png' % (self.ident,ts,))
		plt.savefig(self.save_prediction_png, bbox_inches='tight', dpi=150)

	
	def plot_1zone(self, pred, X_future, Y_future, cords = None):

		print('plotting output variable and control variable')
		
		#start = 0
		#end   = len(pred)
		if cords is None:
			start = 0
			end   = len(pred)
		else:
			start = cords[0]
			end   = cords[1]

		print('start: ', start)
		print('end: ', end)
		print('pred.len: ', len(pred))

		plot_ts = 12

		fig = plt.figure(dpi=150)
		widthofpng=75

		window_opeing_index = 7
		hvac_mode_index = 8
		heating_setpoint_index = 9
		cooling_setpoint_index = 10
		tot_subplots = Y_future.shape[2] + 1

		fig.set_size_inches(widthofpng, tot_subplots*4, True)
		plt.subplots_adjust(hspace=0.1, wspace=0.05)
		colorsa=['red','blue','orange','black','darkgreen']
		for ix in range(Y_future.shape[2]):
			ax = fig.add_subplot(tot_subplots,1,ix+1)
			ax.plot(range(start,end,1),Y_future[0:len(pred),plot_ts,ix], label='Actual', color=colorsa[ix], linewidth = 2, linestyle='dashed', alpha=0.6)
			if self.model_type_prob == 'prob':
				ax.plot(range(start,end,1),pred[0:len(pred),plot_ts,ix,2],label='Predicted', color=colorsa[ix], linewidth = 1, alpha=1)
				ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,heating_setpoint_index],label='heating_setpoint', color='grey', linewidth = 1, alpha=0.5)
				ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,cooling_setpoint_index],label='cooling_setpoint', color='grey', linewidth = 1, linestyle='dashed', alpha=0.5)
				#ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,0],label='Forecasted OAT', color='black', linewidth = 1)
				
				ax.fill_between(range(start,end,1), 
							pred[0:len(pred),plot_ts,ix,0], 
							pred[0:len(pred),plot_ts,ix,4],
							color = 'green', label = '95% confidence interval [P05,P95]', alpha=0.1,
							linewidth = 0)

				ax.fill_between(range(start,end,1), 
							pred[0:len(pred),plot_ts,ix,1], 
							pred[0:len(pred),plot_ts,ix,3],
							color = 'green', label = '90% confidence interval [P10,P90]', alpha=0.2,
							linewidth = 0)    
			elif self.model_type_prob == 'nonprob':
				ax.plot(range(start,end,1),pred[0:len(pred),plot_ts,ix],label='Predicted', color=colorsa[ix], linewidth = 1, alpha=1)
				#ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,heating_setpoint_index],label='heating_setpoint', color='grey', linewidth = 1, alpha=0.5)
				#ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,cooling_setpoint_index],label='cooling_setpoint', color='grey', linewidth = 1, linestyle='dashed', alpha=0.5)
				#ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,0],label='Forecasted OAT', color='black', linewidth = 1)
			ax.legend(loc=1, fontsize=12) 				
			ax.margins(x=0)
			"""
			if ix == 0:
				ax.set_ylim([0, 32])
			elif ix == 1:
				ax.set_ylim([0, 7000])
			"""
			#locs = ax.get_xticks()
			#ax.set_xticks(locs)
			#ax.set_xticklabels(np.linspace(cords[0],cords[1], len(locs), dtype=int))
			#ax.set_xticklabels(generate_numbers(cords[0],cords[1], len(locs)))
			ax.title.set_text('Ident <%s>: Output Variable %s' % (str(self.ident), str(ix+1)))
		
		ax = fig.add_subplot(tot_subplots,1,Y_future.shape[2]+1)
		ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,window_opeing_index], label = 'Window Opening', color='green', linestyle='dashed', linewidth = 2)
		ax.plot(range(start,end,1),X_future[0:len(pred),plot_ts,hvac_mode_index], label = 'HVAC Mode', color='brown', linestyle='dotted', linewidth = 2)
		ax.margins(x=0)
		ax.legend(loc=1, fontsize=12)
		ax.set_ylim([-0.5, 3.5])
		ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 2, 3])
		#locs = ax.get_xticks()
		#ax.set_xticks(locs)
		#ax.set_xticklabels(np.linspace(cords[0],cords[1], len(locs), dtype=int))
		#ax.set_xticklabels(generate_numbers(cords[0],cords[1], len(locs)))
		ax.grid(axis='y', alpha=0.5)

		self.save_prediction_png	= os.path.join(self.save_results_dir, '%s_prediction_mult_ts.png' % (self.ident))
		plt.savefig(self.save_prediction_png, bbox_inches='tight', dpi=150)


	
	def plot_bytimestep(self, y_true, y_pred):
		print('plotting CVRMSE by timestep for each output variable')

		filesavename = os.path.join(self.save_results_dir, '%s_cvrmse_r2_nmbe_bytimestep.npy' % (self.ident))
		all_metrics = calculate_metrics_nonprob(y_true, y_pred, filesavename)

		#all_metrics_name = ['CVRMSE(%)', 'RMSE', 'MAE', 'MSE', 'Rsquared', 'DTW.Distance']
		all_metrics_name = ['CVRMSE(%)', 'RMSE', 'MAE', 'MSE', 'Rsquared']
		
		
		for metric_index in range(0,len(all_metrics_name)):
			metric_name = all_metrics_name[metric_index]
			print('metric_name: ', metric_name)
			fig = plt.figure(figsize=(15, 5))
			fig.subplots_adjust(wspace=0.1, hspace=0.5)
			ax1 = fig.add_subplot(1,2,2)
			colorsa=['red','blue','orange','black','darkgreen']
			markersa=['o','v','^','s','*']
			for i in range(y_true.shape[2]):
				ax1.plot(all_metrics[:, i, metric_index], alpha=1, color=colorsa[i], label='Output Variable:'+str(i+1), linewidth = 1, marker=markersa[i], markersize=3)
			ax1.set_ylabel(metric_name)
			ax1.grid(axis = 'y',alpha=0.5)
			ax1.margins(x=0)
			ax1.set_xlabel("Nth Step-ahead", fontsize=12)
			ax1.legend(loc=2, bbox_to_anchor=(1, 0.7))
			ax1.set_title('%s of all Actual vs Predicted at Nth Step-ahead' % metric_name, fontsize=13)
			self.save_prediction_png	= os.path.join(self.save_results_dir, '%s_%s_bytimestep.png' % (self.ident, str(metric_index)))
			plt.savefig(self.save_prediction_png, bbox_inches='tight', dpi=150)

		print('Stacking both plots horizontally and deleting the individual plots')

		first 	= cv2.imread(os.path.join(self.save_results_dir, '%s_0_bytimestep.png' % (self.ident)))
		second 	= cv2.imread(os.path.join(self.save_results_dir, '%s_1_bytimestep.png' % (self.ident)))
		third 	= cv2.imread(os.path.join(self.save_results_dir, '%s_2_bytimestep.png' % (self.ident)))
		fourth 	= cv2.imread(os.path.join(self.save_results_dir, '%s_3_bytimestep.png' % (self.ident)))
		fifth 	= cv2.imread(os.path.join(self.save_results_dir, '%s_4_bytimestep.png' % (self.ident)))
		#sixth 	= cv2.imread(os.path.join(self.save_results_dir, '%s_5_bytimestep.png' % (self.ident)))

		h1, w1, c1 = first.shape
		h2, w2, c2 = second.shape
		h3, w3, c3 = third.shape
		h4, w4, c4 = fourth.shape
		h5, w5, c5 = fifth.shape
		#h6, w6, c6 = sixth.shape
		#h6, w6, c6 = 0,0,0

		#h, w = max(h1, h2, h3, h4, h5, h6), w1 + w2 + w3 + w4 + w5 + w6   #h, w = max(h1, h2, h3), w1 + w2 + w3
		h, w = max(h1, h2, h3, h4, h5), w1 + w2 + w3 + w4 + w5   #h, w = max(h1, h2, h3), w1 + w2 + w3
		out_image = np.zeros((h, w, c1))
		out_image[:h1, :w1] = first
		out_image[:h2, w1:w1+w2] = second
		out_image[:h3, w1+w2:w1+w2+w3] = third
		out_image[:h4, w1+w2+w3:w1+w2+w3+w4] = fourth
		out_image[:h5, w1+w2+w3+w4:w1+w2+w3+w4+w5] = fifth
		#out_image[:h6, w1+w2+w3+w4+w5:w1+w2+w3+w4+w5+w6] = sixth
		self.save_prediction_png	= os.path.join(self.save_results_dir, '%s_bytimestep.png' % (self.ident))
		cv2.imwrite(self.save_prediction_png, out_image)

		print('Stacking both plots vertically and deleting the individual plots')

		first = cv2.imread(os.path.join(self.save_results_dir, '%s_prediction_mult_ts.png' % (self.ident)))
		second = cv2.imread(os.path.join(self.save_results_dir, '%s_bytimestep.png' % (self.ident)))
		h1, w1, c1 = first.shape
		h2, w2, c2 = second.shape
		h, w= h1+h2, max(w1, w2)
		out_image = np.zeros((h,w,c1))
		out_image[:h1,:w1, ] = first
		out_image[h1:h1+h2,:w2, ] = second
		self.save_prediction_png	= os.path.join(self.save_results_dir, '%s_merged.png' % (self.ident))
		cv2.imwrite(self.save_prediction_png, out_image)

		os.remove(os.path.join(self.save_results_dir, '%s_prediction_mult_ts.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_bytimestep.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_0_bytimestep.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_1_bytimestep.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_2_bytimestep.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_3_bytimestep.png' % (self.ident)))
		os.remove(os.path.join(self.save_results_dir, '%s_4_bytimestep.png' % (self.ident)))
		#os.remove(os.path.join(self.save_results_dir, '%s_5_bytimestep.png' % (self.ident)))

	
	def history_plot_train_on_batch(self, df_loss, df_v_loss, df_lr, df_loss_index, df_v_loss_index):
		fig, ax1 = plt.subplots(figsize=(15,5))
		ax2 = ax1.twinx()

		ax2.plot(df_loss_index,   df_lr,     label='LR',              color='b', alpha=0.5, lw = 0.6)
		ax1.plot(df_loss_index,   df_loss,   label='Loss',            color='g', lw = 0.6)
		if sum(df_v_loss) > 0:
			ax1.plot(df_v_loss_index, df_v_loss, label='Validation_Loss', color='g', marker = 'o', markersize = 2, lw = 0.6)

		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Loss', color='g')
		ax2.set_ylabel('Learning rate', color='b')
		plt.title("Model Loss vs Learning Rate")
		ax1.legend(loc=3)
		ax2.legend(loc=1)
		self.save_training_history      = os.path.join(self.save_results_dir, '%s_history.png' % (self.ident))
		plt.savefig(self.save_training_history, bbox_inches='tight',dpi=150)

	def history_plot_model_fit(self, history):
		df_loss = history.history['loss']
		df_v_loss = history.history['val_loss']
		df_lr = history.history['lr']
		df_loss_index = range(1,len(df_loss)+1)
		df_v_loss_index = range(1,len(df_v_loss)+1)

		self.history_plot_train_on_batch(df_loss, df_v_loss, df_lr, df_loss_index, df_v_loss_index)

	def history_plot_model_fit_novalidate(self, history):
		df_loss = history.history['loss']
		df_lr = history.history['lr']
		df_loss_index = range(1,len(df_loss)+1)
		df_v_loss_index = df_loss_index
		df_v_loss = [0 for i in range(len(df_loss))]

		self.history_plot_train_on_batch(df_loss, df_v_loss, df_lr, df_loss_index, df_v_loss_index)