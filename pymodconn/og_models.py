
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Input, RepeatVector, concatenate, Activation, dot, TimeDistributed, Concatenate
from keras.models import Model
import tensorflow as tf


def build_model_e1d1(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']
	
	encoder_inputs = Input(shape=(n_past, n_features_input+ future_data_col), name='encoder_past_inputs')
	
	encoder_l1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs1 = encoder_l1(encoder_inputs)
	encoder_states1 = encoder_outputs1[1:]
	
	decoder_inputs = RepeatVector(n_future)(encoder_outputs1[0])
	
	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = encoder_states1)
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)

	model = Model(encoder_inputs,decoder_outputs1)
	
	return model

def build_model_e1d1_attn(CONFIGS): #to edit
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	input_train = Input(shape=(n_past, n_features_input + future_data_col), name='encoder_past_inputs')
	
	encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(all_layers_neurons, dropout=all_layers_dropout, return_state=True, return_sequences=True, name = 'LSTM_enc_past')(input_train)

	decoder_input = RepeatVector(n_future)(encoder_last_h)
	decoder_stack_h = LSTM(all_layers_neurons, dropout=all_layers_dropout, return_state=False, return_sequences=True, name='LSTM_dec_future')(decoder_input, initial_state=[encoder_last_h, encoder_last_c])

	attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
	attention = Activation('softmax')(attention)

	context = dot([attention, encoder_stack_h], axes=[2,1])

	decoder_combined_context = concatenate([context, decoder_stack_h])

	out = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_combined_context)

	model = Model(inputs=input_train, outputs=out)
	
	return model


def build_model_e1d1_wFuture0(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	"""
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	"""
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(encoder_outputs_future1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model


def build_model_e1d1_wFuture1_1(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input ), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_1_Dense(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input ), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_1_Dense2(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input ), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons)(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons)(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_1_GRU(CONFIGS):
    n_past 				= CONFIGS['n_past']
    n_future 			= CONFIGS['n_future']
    n_features_input   	= CONFIGS['known_past_features']
    future_data_col    	= CONFIGS['known_future_features']
    n_features_output  	= CONFIGS['unknown_future_features']
    all_layers_neurons 	= CONFIGS['all_layers_neurons']
    all_layers_dropout 	= CONFIGS['all_layers_dropout']

    encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

    encoder_past = GRU(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='GRU_enc_past')
    encoder_outputs_past, encoder_state_past = encoder_past(encoder_past_inputs)

    encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

    encoder_future = GRU(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='GRU_enc_future')
    encoder_outputs_future, encoder_state_future = encoder_future(encoder_future_inputs)

    decoder = GRU(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='GRU_dec_future')(encoder_outputs_future, initial_state=encoder_state_past)

    decoder_outputs = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder)

    model = Model([encoder_past_inputs, encoder_future_inputs], decoder_outputs)
    return model

def build_model_e1d1_wFuture1_2(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_2_Dense(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_2_Dense2(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons)(encoder_past_inputs)
	#encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs1)
	

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons)(encoder_future_inputs)
	#encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs1)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	decoder_l1 = LSTM(all_layers_neurons, 
				   return_sequences=True, 
				   dropout=all_layers_dropout, 
				   name='LSTM_dec_future')(encoder_outputs_future1_h,
							   				initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])
	
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture1_2_GRU(CONFIGS):
    n_past = CONFIGS['n_past']
    n_future = CONFIGS['n_future']
    n_features_input = CONFIGS['known_past_features']
    future_data_col = CONFIGS['known_future_features']
    n_features_output = CONFIGS['unknown_future_features']
    all_layers_neurons = CONFIGS['all_layers_neurons']
    all_layers_dropout = CONFIGS['all_layers_dropout']

    encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

    encoder_past = GRU(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='GRU_enc_past')
    encoder_outputs_past, encoder_state_past = encoder_past(encoder_past_inputs)

    encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

    encoder_future = GRU(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='GRU_enc_future')
    encoder_outputs_future, encoder_state_future = encoder_future(encoder_future_inputs, initial_state=encoder_state_past)

    decoder = GRU(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='GRU_dec_future')(encoder_outputs_future, initial_state=encoder_state_past)

    decoder_outputs = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder)

    model = Model([encoder_past_inputs, encoder_future_inputs], decoder_outputs)
    return model

def build_model_e1d1_wFuture2_AddNorm(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)
	
	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	adnorm_c 			= add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	#concat_c 			 = concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h,adnorm_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture2_Concat(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)
	
	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	#adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h, concat_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture2_Concat_Dense(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)
	
	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	#adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h, concat_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture2_Concat_attn(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[1]
	encoder_outputs_past1_stack = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')

	encoder_outputs_future1 = encoder_future1(encoder_future_inputs,initial_state = [encoder_outputs_past1_h,encoder_outputs_past1_c])

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[1]
	encoder_outputs_future1_stack = encoder_outputs_future1[0]

	#adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_stack,initial_state = [encoder_outputs_past1_h,concat_c])

	attention = dot([decoder_l1, encoder_outputs_past1_stack], axes=[2, 2])
	attention = Activation('softmax')(attention)
	context = dot([attention, encoder_outputs_past1_stack], axes=[2,1])
	decoder_combined_context = concatenate([context, decoder_l1])

	attention1 = dot([decoder_l1, encoder_outputs_future1_stack], axes=[2, 2])
	attention1 = Activation('softmax')(attention1)
	context1 = dot([attention1, encoder_outputs_future1_stack], axes=[2,1])
	decoder_combined_context1 = concatenate([context1, decoder_l1])

	combined_attn = concatenate([decoder_combined_context, decoder_combined_context1])

	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(combined_attn)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture3_AddNorm(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	#concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h, adnorm_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture3_Concat(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	#adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h, concat_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture3_Concat_Dense(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, return_sequences=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1)

	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	#adnorm_c 			=  add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	concat_c 			= concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(encoder_outputs_future1_h,initial_state = [encoder_outputs_past1_h, concat_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model


def build_model_e1d1_wFuture4_AddNorm(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	encoder_combined_c = add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	encoder_combined_h = add_and_norm([encoder_outputs_past1_h, encoder_outputs_future1_h])
	decoder_inputs = RepeatVector(n_future)(encoder_combined_h)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = [encoder_combined_h,encoder_combined_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture4_Concat(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	encoder_combined_c = concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)
	encoder_combined_h = concat_layers([encoder_outputs_past1_h, encoder_outputs_future1_h], all_layers_neurons)
	decoder_inputs = RepeatVector(n_future)(encoder_combined_h)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = [encoder_combined_h,encoder_combined_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture4_Concat_Dense(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_past_inputs)

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs1)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future_inputs1 = Dense(all_layers_neurons, activation='relu')(encoder_future_inputs)

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs1)
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	encoder_combined_c = concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)
	encoder_combined_h = concat_layers([encoder_outputs_past1_h, encoder_outputs_future1_h], all_layers_neurons)
	decoder_inputs = RepeatVector(n_future)(encoder_combined_h)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = [encoder_combined_h,encoder_combined_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture5_AddNorm(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	encoder_combined_c = add_and_norm([encoder_outputs_past1_c, encoder_outputs_future1_c])
	encoder_combined_h = add_and_norm([encoder_outputs_past1_h, encoder_outputs_future1_h])
	decoder_inputs = RepeatVector(n_future)(encoder_combined_h)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = [encoder_outputs_past1_h,encoder_combined_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model

def build_model_e1d1_wFuture5_Concat(CONFIGS):
	n_past 				= CONFIGS['n_past']
	n_future 			= CONFIGS['n_future']
	n_features_input 	= CONFIGS['known_past_features']
	future_data_col 	= CONFIGS['known_future_features']
	n_features_output 	= CONFIGS['unknown_future_features']
	all_layers_neurons 	= CONFIGS['all_layers_neurons']
	all_layers_dropout 	= CONFIGS['all_layers_dropout']

	encoder_past_inputs = Input(shape=(n_past, n_features_input), name='encoder_past_inputs')

	encoder_past1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_past')
	encoder_outputs_past1 = encoder_past1(encoder_past_inputs)

	encoder_outputs_past1_c = encoder_outputs_past1[2]
	encoder_outputs_past1_h = encoder_outputs_past1[0]

	encoder_future_inputs = Input(shape=(n_future, future_data_col), name='encoder_future_inputs')

	encoder_future1 = LSTM(all_layers_neurons, return_state=True, dropout=all_layers_dropout, name='LSTM_enc_future')
	encoder_outputs_future1 = encoder_future1(encoder_future_inputs)
	encoder_outputs_future1_c = encoder_outputs_future1[2]
	encoder_outputs_future1_h = encoder_outputs_future1[0]

	encoder_combined_c = concat_layers([encoder_outputs_past1_c, encoder_outputs_future1_c], all_layers_neurons)
	encoder_combined_h = concat_layers([encoder_outputs_past1_h, encoder_outputs_future1_h], all_layers_neurons)
	decoder_inputs = RepeatVector(n_future)(encoder_combined_h)

	decoder_l1 = LSTM(all_layers_neurons, return_sequences=True, dropout=all_layers_dropout, name='LSTM_dec_future')(decoder_inputs,initial_state = [encoder_outputs_past1_h,encoder_combined_c])
	decoder_outputs1 = TimeDistributed(Dense(n_features_output, activation=lambda x: tf.clip_by_value(x, -1, 1)))(decoder_l1)
	
	model = Model([encoder_past_inputs,encoder_future_inputs],decoder_outputs1)
	return model


def add_and_norm(x_list):
	Add = tf.keras.layers.Add
	LayerNorm = tf.keras.layers.LayerNormalization
	tmp = Add()(x_list)
	tmp = LayerNorm()(tmp)
	return tmp

def concat_layers(x_list, layer_size):
	Concatenate = tf.keras.layers.Concatenate
	tmp = Concatenate()(x_list)
	tmp = Dense(layer_size, activation='relu')(tmp)

	return tmp
