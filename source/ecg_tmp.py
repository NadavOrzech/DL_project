# import tensorflow as tf
# from keras.layers import Dense,Activation,Dropout
# from keras.layers import LSTM,Bidirectional, GRU #could try TimeDistributed(Dense(...))
# from keras.models import Sequential, load_model
# from keras import optimizers,regularizers
# from keras.layers.normalization import BatchNormalization
# import keras.backend.tensorflow_backend as KTF
# from keras.metrics import categorical_accuracy

## MITDB
import os
import wfdb
import numpy as np
import glob
import math
import sys

np.random.seed(42)


def get_ecg_data_nl(datfile):
	## convert .dat/q1c to numpy arrays
	recordname=os.path.basename(datfile).split(".dat")[0]
	recordpath=os.path.dirname(datfile)
	cwd=os.getcwd()
	os.chdir(recordpath) ## somehow it only works if you chdir. 
	record = wfdb.rdsamp(recordname)
	Vctrecord=np.transpose(record.p_signals)
	Vctrecord=np.transpose(Vctrecord)
	return Vctrecord

# functions
def get_ecg_data(datfile): 
    ## convert .dat/q1c to numpy arrays
    recordname=os.path.basename(datfile).split(".dat")[0]
    recordpath=os.path.dirname(datfile)
    cwd=os.getcwd()
    os.chdir(recordpath) ## somehow it only works if you chdir. 

    annotator='atr'
    annotation = wfdb.rdann(recordname, extension=annotator)
    record = wfdb.rdsamp(recordname, sampfrom=0,sampto = None) #wfdb.showanncodes()

    Vctrecord=np.transpose(record.p_signals)
    VctAnnotationHot=np.zeros( (2,len(Vctrecord[1])), dtype=np.int)
    VctAnnotationHot[1] = 1;
    
    #print("ecg, 2 lead of shape" , Vctrecord.shape) 
    #print("VctAnnotationHot of shape" , VctAnnotationHot.shape) 
    #print('plotting extracted signal with annotation')
    wfdb.plotrec(record, annotation=annotation, title='Record 100 from MIT-BIH Arrhythmia Database', timeunits = 'seconds')

    VctAnnotations=list(zip(annotation.sample,annotation.symbol)) ## zip coordinates + annotations (N),(t) etc)
    #print(VctAnnotations)
    for i in range(len(VctAnnotations)):
        if( VctAnnotations[i][1] == 'N' or 
            VctAnnotations[i][1] == 'L' or  
            VctAnnotations[i][1] == 'R' or 
            VctAnnotations[i][1] == 'B' or 
            VctAnnotations[i][1] == 'A' or
            VctAnnotations[i][1] == 'a' or 
            VctAnnotations[i][1] == 'J' or 
            VctAnnotations[i][1] == 'S' or 
            VctAnnotations[i][1] == 'V' or  
            VctAnnotations[i][1] == 'r' or  
            VctAnnotations[i][1] == 'F' or 
            VctAnnotations[i][1] == 'e' or 
            VctAnnotations[i][1] == 'j' or
            VctAnnotations[i][1] == 'n' or 
            VctAnnotations[i][1] == 'E' or 
            VctAnnotations[i][1] == '/' or 
            VctAnnotations[i][1] == 'f' or
            VctAnnotations[i][1] == 'Q' or  
            VctAnnotations[i][1] == '?'):
            VctAnnotationHot[0][VctAnnotations[i][0]] = 1;  
            VctAnnotationHot[1][VctAnnotations[i][0]] = 0;  
    VctAnnotationHot=np.transpose(VctAnnotationHot)
    Vctrecord=np.transpose(Vctrecord) # transpose to (timesteps,feat)

    os.chdir(cwd)
    return Vctrecord, VctAnnotationHot



def splitseq(x,n,o):
	n = int(n)
	o = int(o)
	#split seq; should be optimized so that remove_seq_gaps is not needed. 
	upper= int(math.ceil( x.shape[0] / n) *n)
	print("splitting on",n,"with overlap of ",o,	"total datapoints:",x.shape[0],"; upper:",upper)
	for i in range(0,upper,n):
		#print(i)
		if i==0:
			padded=np.zeros( ( o+n+o,x.shape[1])   ) ## pad with 0's on init
			padded[o:,:x.shape[1]] = x[i:i+n+o,:]
			xpart=padded
		else:
			xpart=x[i-o:i+n+o,:]
		if xpart.shape[0]<i:

			padded=np.zeros( (o+n+o,xpart.shape[1])  ) ## pad with 0's on end of seq
			padded[:xpart.shape[0],:xpart.shape[1]] = xpart
			xpart=padded

		xpart=np.expand_dims(xpart,0)## add one dimension; so that you get shape (samples,timesteps,features)
		try:
			xx=np.vstack(  (xx,xpart) )
		except UnboundLocalError: ## on init
			xx=xpart
	print("output: ",xx.shape)
	return(xx)

def remove_seq_gaps(x,y):
	#remove parts that are not annotated <- not ideal, but quickest for now.
	window=150
	c=0
	cutout=[]
	include=[]
	print("filterering.")
	print("before shape x,y",x.shape,y.shape)
	for i in range(y.shape[0]):
		
		c=c+1
		if c<window :
			include.append(i)
		if sum(y[i,0:5])>0:
			c=0 
		if c >= window:
			#print ('filtering')
			pass
	x,y=x[include,:],y[include,:]
	print(" after shape x,y",x.shape,y.shape)
	return(x,y)


def normalize_new(x):
    for i in range(x.shape[0]):
        x[i] = scale( x[i], axis=0, with_mean=True, with_std=True, copy=True )
    return x

def normalizesignal(x):
	x=st.zscore(x, ddof=0)
	return x
def normalizesignal_array(x):
	for i in range(x.shape[0]):
		x[i]=st.zscore(x[i], axis=0, ddof=0)
	return x

def plotecg(x,y,begin,end):
	#helper to plot ecg
	plt.figure(1,figsize=(11.69,8.27))
	plt.subplot(211)
	plt.plot(x[begin:end,0])
	plt.subplot(211)
	plt.plot(y[begin:end,0])
	plt.subplot(211)
	plt.plot(y[begin:end,1])
	plt.subplot(211)
	plt.plot(y[begin:end,2])
	plt.subplot(211)
	plt.plot(y[begin:end,3])
	plt.subplot(211)
	plt.plot(y[begin:end,4])
	plt.subplot(211)
	plt.plot(y[begin:end,5])

	plt.subplot(212)
	plt.plot(x[begin:end,1])
	plt.show()

def plotecg_validation(x,y_true,y_pred,begin,end):
	#helper to plot ecg
	plt.figure(1,figsize=(11.69,8.27))
	plt.subplot(211)
	plt.plot(x[begin:end,0])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,0])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,1])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,2])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,3])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,4])
	plt.subplot(211)
	plt.plot(y_pred[begin:end,5])

	plt.subplot(212)
	plt.plot(x[begin:end,1])
	plt.subplot(212)
	plt.plot(y_true[begin:end,0])
	plt.subplot(212)
	plt.plot(y_true[begin:end,1])
	plt.subplot(212)
	plt.plot(y_true[begin:end,2])
	plt.subplot(212)
	plt.plot(y_true[begin:end,3])
	plt.subplot(212)
	plt.plot(y_true[begin:end,4])
	plt.subplot(212)
	plt.plot(y_true[begin:end,5])

    
def LoaddDatFiles(datfiles):  
    for datfile in datfiles:
        print(datfile)
        if basename(datfile).split(".",1)[0] in exclude:
            continue
        qf=os.path.splitext(datfile)[0]+'.atr'
        if os.path.isfile(qf):
            #print("yes",qf,datfile)
            x,y=get_ecg_data(datfile)

            x,y=splitseq(x,1000,0),splitseq(y,1000,0) ## create equal sized numpy arrays of n size and overlap of o 

            x = normalize_new(x)
            ## todo; add noise, shuffle leads etc. ?
            try: ## concat
                xx=np.vstack(  (xx,x) )
                yy=np.vstack(  (yy,y) )
            except NameError: ## if xx does not exist yet (on init)
                xx = x
                yy = y
    return(xx,yy)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]




if __name__=="__main__":
    qtdbpath="C:\\Users\\Dell\\Desktop\\Technion\\DeepLearning\\project_data\\mit-bih\\files"#sys.argv[1] ## first argument = qtdb database from physionet. 

    # load data
    datfiles=glob.glob(os.path.join(qtdbpath,"*.dat"))
    # print(datfiles)

        
    # load data
    perct=0.81 #percentage training
    percv=0.19 #percentage validation
    exclude = set()

    wfdb.show_ann_labels()

    for datfile in datfiles:
        print(datfile)
        if os.path.basename(datfile).split(".",1)[0] in exclude:
            continue
        
        qf=os.path.splitext(datfile)[0]+'.atr'
        if os.path.isfile(qf):
            #print("yes",qf,datfile)
            x,y=get_ecg_data(datfile)
            print(x)
            # for yy in y:
            #     if yy[0] != 0 or yy[1] != 1:
            #         print("FOUND ONE")
            print(y)
            # x,y=splitseq(x,1000,0),splitseq(y,1000,0) ## create equal sized numpy arrays of n size and overlap of o 

            # x = normalize_new(x)
            # ## todo; add noise, shuffle leads etc. ?
            # try: ## concat
            #     xx=np.vstack(  (xx,x) )
            #     yy=np.vstack(  (yy,y) )
            # except NameError: ## if xx does not exist yet (on init)
            #     xx = x
            #     yy = y





# def get_session(gpu_fraction=0.8):
# 	#allocate % of gpu memory.
# 	num_threads = os.environ.get('OMP_NUM_THREADS')
# 	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
# 	if num_threads:
# 		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
# 	else:
# 		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# from keras import backend as K

# import numpy as np
# from keras.activations import softmax
# from keras.objectives import categorical_crossentropy

# def weighted_categorical_crossentropy(weights):
#     """
#     A weighted version of keras.objectives.categorical_crossentropy
    
#     Variables:
#         weights: numpy array of shape (C,) where C is the number of classes
    
#     Usage:
#         weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
#         loss = weighted_categorical_crossentropy(weights)
#         model.compile(loss=loss,optimizer='adam')
#     """
    
#     weights = K.variable(weights)
        
#     def loss(y_true, y_pred):
#         # scale predictions so that the class probas of each sample sum to 1
#         y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
#         # clip to prevent NaN's and Inf's
#         y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
#         # calc
#         loss = y_true * K.log(y_pred) * weights
#         loss = -K.sum(loss, -1)
#         return loss
    
#     return loss

# def build_model_gru():
#     #data_dim = X_data.shape[2]
#     #timesteps = X_data.shape[1]
#     model = Sequential()
#     model.add(BatchNormalization(input_shape=(seqlength, features)))  
#     model.add(GRU(output_dim=50,init ='glorot_normal',
#          return_sequences=True, W_regularizer=regularizers.l2(0.00),U_regularizer=regularizers.l1(0.01),dropout_W =0.2 ))
#     model.add(GRU(output_dim=50,init ='glorot_normal',
#         return_sequences=True,W_regularizer=regularizers.l2(0.00),U_regularizer=regularizers.l1(0.01),dropout_W =0.2))
#     model.add(GRU(50,init ='glorot_normal',return_sequences=False,dropout_W =0.01, W_regularizer=regularizers.l2(0.00),U_regularizer=regularizers.l1(0.01)))
#     model.add(Dense(dimout, init='glorot_normal'))
#     model.add(Activation('softmax'))

#     #weights = np.ones((dimout,))
#     weights = np.array([0.8, 500.0])
#     model.compile(loss=weighted_categorical_crossentropy(weights),optimizer='Adam')
#     return model

# def getmodel_simple():
#     # create model
#     model = Sequential()
#     model.add(Dense(1000,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
#     model.add(Dense(2000, activation='relu'))
#     model.add(Dense(dimout, activation='softmax'))
#     # Compile model
#     weights = np.array([128., 0.1])
#     adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
#     model.compile(loss=weighted_categorical_crossentropy(weights),optimizer=adam, metrics=['accuracy']) #(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
#     return model

# def getmodel():
#     model = Sequential()
#     model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(Dense(32, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(Dense(dimout, activation='softmax'))
#     adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     weights = np.array([64.0, 0.1,])
#     model.compile(loss=weighted_categorical_crossentropy(weights),optimizer=adam, metrics=['accuracy']) #(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
#     print(model.summary())
#     return(model)

# def f1(y_true, y_pred):
#     y_pred = K.round(y_pred)
#     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())

#     f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return K.mean(f1)

# def f1_loss(y_true, y_pred):
    
#     tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
#     tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
#     fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
#     fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())

#     f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     return 1 - K.mean(f1)


# def getmodel_one():
#     model = Sequential()
#     model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
#     model.add(Bidirectional(LSTM(64, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(Dense(32, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(Dense(dimout, activation='sigmoid'))
#     adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     weights = np.array([64.0, 0.1,])
#     model.compile(loss=f1_loss, #loss=weighted_categorical_crossentropy(weights), 
#                   optimizer=adam, 
#                   metrics=['categorical_accuracy', f1_m,precision_m, recall_m]) #(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
#     print(model.summary())
#     return(model)


# def recall_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

# def precision_m(y_true, y_pred):
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision

# def f1_m(y_true, y_pred):
    # precision = precision_m(y_true, y_pred)
    # recall = recall_m(y_true, y_pred)
    # return 2*((precision*recall)/(precision+recall+K.epsilon()))