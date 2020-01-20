from dataloader import load_data, balanced_sampler, unbalanced_sampler
import numpy as np
from cm import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def unbalanced_cross_validation_data(dataset, cnt, emotions, K):
	balanced_data, count, e_cnts = unbalanced_sampler(dataset, cnt, emotions)
	curr_size = 0
	increment = count//K
	k_data = {}
	for emotion in emotions:
		k_data[emotion] = []
	for k in range(K):
		for emotion in emotions:
			k_data[emotion].append(balanced_data[emotion][curr_size:curr_size+increment])
		curr_size = curr_size+increment
	rest_per_emotion = (count - curr_size)
	k = 0
	for emotion in emotions:
		for next_id in range(curr_size,len(balanced_data[emotion])):
			k_data[emotion][k%K].append(balanced_data[emotion][next_id])
			k=k+1
	
	Xs = []
	Ys = []
	img_dim = np.shape(k_data[emotions[0]][0])
	img_dim_1d = img_dim[1] * img_dim[2]
	for k in range(K):
		Xs.append(np.empty((0,img_dim_1d)))
		Ys.append(np.empty((0,1),int))
		for i in range(len(emotions)):
			emotion = emotions[i]
			images = np.array(k_data[emotion][k])
			images = images.reshape(images.shape[0],-1)
			Xs[k] = np.vstack((Xs[k],images))
			Ys[k] = np.vstack((Ys[k],np.zeros(images.shape[0]).reshape(images.shape[0],1)+i)) ######################################
	return Xs,Ys, np.max(e_cnts)/e_cnts

def cross_validation_data(dataset, cnt, emotions, K):
	balanced_data = balanced_sampler(dataset, cnt, emotions)
	count = len(balanced_data[emotions[0]])
	curr_size = 0
	increment = count//K
	k_data = {}
	for emotion in emotions:
		k_data[emotion] = []
	for k in range(K):
		for emotion in emotions:
			k_data[emotion].append(balanced_data[emotion][curr_size:curr_size+increment])
		curr_size = curr_size+increment
	rest_per_emotion = (count - curr_size)
	k = 0
	for emotion in emotions:
		for next_id in range(curr_size,count):
			k_data[emotion][k%K].append(balanced_data[emotion][next_id])
			k=k+1
	
	Xs = []
	Ys = []
	img_dim = np.shape(k_data[emotions[0]][0])
	img_dim_1d = img_dim[1] * img_dim[2]
	for k in range(K):
		Xs.append(np.empty((0,img_dim_1d)))
		Ys.append(np.empty((0,1),int))
		for i in range(len(emotions)):
			emotion = emotions[i]
			images = np.array(k_data[emotion][k])
			images = images.reshape(images.shape[0],-1)
			Xs[k] = np.vstack((Xs[k],images))
			Ys[k] = np.vstack((Ys[k],np.zeros(images.shape[0]).reshape(images.shape[0],1)+i)) ######################################
	return Xs,Ys
	

def PCA(X, y, n_components):
	x_shape = np.shape(X)
	mean_image = np.average(X, axis = 0)

	mean_shifted_data = X - mean_image
	msd = mean_shifted_data

	u_, si, vh = np.linalg.svd(msd, full_matrices=False)
	vh = vh[:n_components].T
	si = si[:n_components]

	return np.matmul(msd, vh)/si, mean_image, si, vh

def convert_PCA(X, mean_image, si, vh):
	mean_shifted_data = X - mean_image
	msd = mean_shifted_data
	return np.matmul(msd, vh)/si

def preprocessor(Xs,Ys, val_id, test_id, n_components, label_weights):
	K = len(Xs)
	train_set_X = np.empty((0,Xs[0].shape[1]))
	train_set_y = np.empty((0,1),'int')
	for k in range(K):
		if k!= val_id and k!=test_id:
			train_set_X = np.vstack((train_set_X, Xs[k]))
			train_set_y = np.vstack((train_set_y, Ys[k]))
	validation_set_X = Xs[val_id]
	validation_set_y = Ys[val_id]
	test_set_X = Xs[test_id]
	test_set_y = Ys[test_id]
	

	weights_y = np.ones_like(train_set_y)
	for ii in range(len(label_weights)):
		weights_y[train_set_y == ii] = label_weights[ii]


	train_PCA, mean_image, si, vh = PCA(train_set_X, train_set_y, n_components)
	valid_PCA = convert_PCA(validation_set_X, mean_image, si, vh)
	test_PCA = convert_PCA(test_set_X, mean_image, si, vh)

	return train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, vh, weights_y
		

def sigmoid(X):
	return 1./(1. + np.exp(-X))

def ce_loss(preds, y):
	return -np.mean((y*np.log(preds+1e-8) + (1-y)*np.log(1-preds + 1e-8)))
		
def logistic_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, shape_0, shape_1, vh, learn_rate, n_epochs = 200, is_stochastic = False):
	weights = np.zeros((train_PCA.shape[1]+1,1))#np.random.random((train_PCA.shape[1],1))

	train_PCA = np.hstack((train_PCA,np.ones((train_set_y.shape[0],1))))
	valid_PCA = np.hstack((valid_PCA,np.ones((valid_PCA.shape[0],1))))
	test_PCA = np.hstack((test_PCA,np.ones((test_PCA.shape[0],1))))


	best_weights = weights
	best_loss = 1e10
	train_accs = []
	train_losses =[]
	valid_accs = []
	valid_losses = []
	for epoch in range(n_epochs):
		if is_stochastic == True:
			assert(False) #Running Logistic for stochastic GD ????
			perm_ind =  np.random.permutation(train_PCA.shape[0])
			train_PCA = train_PCA[perm_ind]
			train_set_y = train_set_y[perm_ind]
			
			for idx in range(train_PCA.shape[0]):
				stochastic_x = train_PCA[idx].reshape(1,-1)
				stochastic_y = train_set_y[idx].reshape(1,-1)
				
				predictions = np.matmul(stochastic_x, weights)
				predictions = sigmoid(predictions)
				gradient =  stochastic_x *(predictions - stochastic_y).reshape(predictions.shape[0],1)
				gradient = gradient.sum(axis=0).reshape(weights.shape)
				
				weights =  weights - learn_rate * gradient
		else:
			predictions = np.matmul(train_PCA, weights)
			predictions = sigmoid(predictions)
			gradient =  train_PCA *(predictions - train_set_y).reshape(predictions.shape[0],1)
			gradient = gradient.sum(axis=0).reshape(weights.shape)
			
			weights =  weights - learn_rate * gradient
		predictions_valid = np.matmul(valid_PCA, weights)
		predictions_valid = sigmoid(predictions_valid)
		
		train_loss = ce_loss(predictions, train_set_y)
		valid_loss = ce_loss(predictions_valid, validation_set_y)
		if valid_loss<best_loss:
			best_loss = valid_loss
			best_weights = weights.copy()
		#print('Epoch {}'.format(epoch))
		#print('Train_loss {}'.format(train_loss))
		#print('Valid_loss {}'.format(valid_loss))
	
		accuracy_tr = np.sum(1*(predictions>0.5)== train_set_y)/train_set_y.shape[0]
		accuracy_valid = np.sum(1*(predictions_valid>0.5) == validation_set_y)/validation_set_y.shape[0]
		
		#print('Train_acc {}'.format(accuracy_tr))
		#print('Valid_acc {}'.format(accuracy_valid))
		train_accs.append(accuracy_tr)
		train_losses.append(train_loss)
		valid_accs.append(accuracy_valid)
		valid_losses.append(valid_loss)
	
	# if (fold == 0):
	# 	visualize_pcs(shape_0, shape_1, vh)

	predictions_test = np.matmul(test_PCA, best_weights)
	predictions_test = sigmoid(predictions_test)
	test_loss = ce_loss(predictions_test, test_set_y)
	accuracy_test = np.sum(1*(predictions_test>0.5) == test_set_y)/test_set_y.shape[0]
	return accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, np.zeros((2,2))

def softmax(X):
	temp = np.exp(X-np.max(X,axis=1,keepdims=True))
	return temp/(temp.sum(axis=1)).reshape(X.shape[0],1)

def softmax_loss(preds,y):
	return -(np.log(preds + 1e-8)*y).mean()

	 
def one_hot_f(y):
	y = y.astype(int)
	one_hot = np.zeros((y.shape[0], int(y.max()+1)))
	one_hot[np.arange(y.shape[0]),y.squeeze()] = 1
	return one_hot
	
def confusion_array(predictions, labels):
	from sklearn.metrics import confusion_matrix

	y_pred = predictions.argmax(1)
	y_actu = labels.argmax(1)

	return confusion_matrix(y_actu, y_pred)

def softmax_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, shape_0, shape_1, vh, emotions, weights_y, learn_rate = 0.1, n_epochs = 200, is_stochastic=False):
	n_emotions = int(train_set_y.max()+1)
	weights = np.zeros((train_PCA.shape[1] + 1,n_emotions))

	train_set_y = one_hot_f(train_set_y)
	validation_set_y = one_hot_f(validation_set_y)
	test_set_y = one_hot_f(test_set_y)
		
	train_PCA = np.hstack((train_PCA,np.ones((train_set_y.shape[0],1))))
	valid_PCA = np.hstack((valid_PCA,np.ones((valid_PCA.shape[0],1))))
	test_PCA = np.hstack((test_PCA,np.ones((test_PCA.shape[0],1))))
	
	best_weights = weights
	best_loss = 1e10
	train_accs = []
	train_losses =[]
	valid_accs = []
	valid_losses = []
	for epoch in range(n_epochs):
		if is_stochastic == True:
			perm_ind =  np.random.permutation(train_PCA.shape[0])
			train_PCA_shuffle = train_PCA[perm_ind]
			train_set_y_shuffle = train_set_y[perm_ind]
			
			for idx in range(train_PCA_shuffle.shape[0]):
				stochastic_x = train_PCA_shuffle[idx].reshape(1,-1)
				stochastic_y = train_set_y_shuffle[idx].reshape(1,-1)
				
				predictions = np.matmul(stochastic_x, weights)
				predictions = softmax(predictions)
				gradient = np.matmul(stochastic_x.T,(predictions - stochastic_y))
				
				weights = weights - learn_rate * gradient
				
		else:
			predictions = np.matmul(train_PCA, weights)
			predictions = softmax(predictions)
			
			gradient =  np.matmul((train_PCA*weights_y).T,(predictions - train_set_y))
			weights =  weights - learn_rate * gradient


		predictions = np.matmul(train_PCA, weights)
		predictions = softmax(predictions)

		predictions_valid = np.matmul(valid_PCA, weights)
		predictions_valid = softmax(predictions_valid)
		
		train_loss = softmax_loss(predictions, train_set_y)
		valid_loss = softmax_loss(predictions_valid, validation_set_y)
		if valid_loss<best_loss:
			best_loss = valid_loss
			best_weights = weights.copy()
		#print('Epoch {}'.format(epoch))
		#print('Train_loss {}'.format(train_loss))
		#print('Valid_loss {}'.format(valid_loss))
	
		accuracy_tr = np.sum((predictions.argmax(1)== train_set_y.argmax(1)))/train_set_y.shape[0]
		accuracy_valid = np.sum((predictions_valid.argmax(1) == validation_set_y.argmax(1)))/validation_set_y.shape[0]
		
		#print('Train_acc {}'.format(accuracy_tr))
		#print('Valid_acc {}'.format(accuracy_valid))
		train_accs.append(accuracy_tr)
		train_losses.append(train_loss)
		valid_accs.append(accuracy_valid)
		valid_losses.append(valid_loss)
	
	predictions_test = np.matmul(test_PCA, best_weights)
	predictions_test = softmax(predictions_test)
	test_loss = softmax_loss(predictions_test, test_set_y)
	accuracy_test = np.sum((predictions_test.argmax(1) == test_set_y.argmax(1)))/test_set_y.shape[0]
	
	# if (fold == 0):
	# 	visualize_weights(best_weights, shape_0, shape_1, vh, emotions)

	test_cij = confusion_array(predictions_test, test_set_y)
	return accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij


import matplotlib.pyplot as plt
def plotter(train_acc, train_loss, valid_accs, valid_loss, n_components, learn_rate):
	everyerror = 50
	train_acc_std = np.std(np.array(train_acc),axis=0).squeeze()
	valid_acc_std = np.std(np.array(valid_accs),axis=0).squeeze()
	train_loss_std = np.std(np.array(train_loss),axis=0).squeeze()
	valid_loss_std = np.std(np.array(valid_loss),axis=0).squeeze()
	
	train_acc = np.mean(np.array(train_acc),axis=0).squeeze()
	valid_accs = np.mean(np.array(valid_accs),axis=0).squeeze()
	train_loss = np.mean(np.array(train_loss),axis=0).squeeze()
	valid_loss = np.mean(np.array(valid_loss),axis=0).squeeze()
	
	xaxis = np.arange(train_acc.shape[0])+1
	fig = plt.figure(figsize=(12,8))
	plt.title('Accuracy vs Epoch, NumberPCs = {} learn_rate = {}'.format(n_components, learn_rate))
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	# plt.plot(xaxis, train_acc, label='Training')
	# plt.plot(xaxis,valid_accs, label='Validation')
	plt.errorbar(xaxis, train_acc, train_acc_std, errorevery =everyerror,elinewidth=3, ecolor='b', color='b',label='Training')
	plt.errorbar(xaxis,valid_accs, valid_acc_std,errorevery =everyerror,elinewidth=1, ecolor='g', color='r',label='Validation')
	
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()
	
	fig = plt.figure()
	plt.title('Loss vs Epoch, NumberPCs = {} learn_rate = {}'.format(n_components, learn_rate))
	plt.xlabel('Epochs')
	plt.ylabel('CE Loss')
	# plt.plot(xaxis, train_loss, label='Training')
	# plt.plot(xaxis,valid_loss, label='Validation')
	plt.errorbar(xaxis, train_loss, train_loss_std, errorevery =everyerror,elinewidth=3, ecolor='b', color='b',label='Training')
	plt.errorbar(xaxis,valid_loss, valid_loss_std,errorevery =everyerror,elinewidth=1, ecolor='g', color='r',label='Validation')
	plt.legend(loc='upper left')
	plt.grid()
	plt.show()
	

def kachra_code(data_dir, emotions, classifier=None, K=10, n_components=40, learn_rates=[0.01, 0.1, 1]):
	dataset, cnt = load_data(data_dir)
	label_weights = np.ones(len(emotions))
	Xs, Ys = cross_validation_data(dataset, cnt,emotions, K)

	train_losses_k_fold_mean = []
	train_losses_k_fold_std = []

	for lr in learn_rates:
		temp_arr = []
		for fold in range(K):
			val_id = fold
			test_id = (fold+1)%K
			train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, vh, weights_y = preprocessor(Xs, Ys, val_id, test_id, n_components, label_weights)
			# accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = logistic_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, lr)
			accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = softmax_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, emotions, weights_y, lr)

			temp_arr.append(np.array(train_losses))

		train_losses_k_fold_mean.append(np.array(temp_arr).mean(axis=0))
		train_losses_k_fold_std.append(np.array(temp_arr).std(axis=0))

	x = range(0,1000,1)
	colors = ['r','b','k']
	e_size = [5,3,1]

	plt.figure(figsize =(10,8))
	for idy,y in enumerate(train_losses_k_fold_mean):
		#ax.plot(x,y,label = learn_rates[idy])
		plt.errorbar(x,y,train_losses_k_fold_std[idy],errorevery=50,elinewidth=e_size[idy], ecolor=colors[idy], color=colors[idy],label=f'lr : {learn_rates[idy]}')

	plt.xlabel("# epoches")
	plt.ylabel("entropy loss")
	plt.ylim([0,0.5])
	plt.title(" entropy loss vs # epoches")
	plt.grid()
	plt.legend()
	plt.show()
	#plt.save('./lr_effect.png')


def kachra_code2(data_dir, emotions, classifier=None, K=10, n_components=40):
	dataset, cnt = load_data(data_dir)
	label_weights = np.ones(len(emotions))
	Xs, Ys = cross_validation_data(dataset, cnt,emotions, K)

	K=1

	temp_arr = []
	for fold in range(K):
		val_id = fold
		test_id = (fold+1)%K
		train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, vh, weights_y = preprocessor(Xs, Ys, val_id, test_id, n_components, label_weights)
		accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = softmax_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, emotions, weights_y, is_stochastic=False)

		temp_arr.append(np.array(train_losses))


	train_losses_mean_batch = np.array(temp_arr).mean(axis=0)
	train_losses_std_batch = np.array(temp_arr).std(axis=0)

	temp_arr = []
	for fold in range(K):
		val_id = fold
		test_id = (fold+1)%K
		train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, vh, weights_y = preprocessor(Xs, Ys, val_id, test_id, n_components, label_weights)
		accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = softmax_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, emotions, weights_y, is_stochastic=True)

		temp_arr.append(np.array(train_losses))


	train_losses_mean_stoch = np.array(temp_arr).mean(axis=0)
	train_losses_std_stoch = np.array(temp_arr).std(axis=0)

	x = range(0,200,1)
	colors = ['r','b']
	e_size = [3,1]

	plt.figure(figsize =(10,8))
	plt.plot(x, train_losses_mean_batch, label='BGD')
	plt.plot(x, train_losses_mean_stoch, label='SGD')
	# plt.errorbar(x,train_losses_mean_batch,train_losses_std_batch,errorevery=10,elinewidth=e_size[0], ecolor=colors[0], color=colors[0],label=f'BatchGradientDescent')
	# plt.errorbar(x,train_losses_mean_stoch,train_losses_std_stoch,errorevery=10,elinewidth=e_size[1], ecolor=colors[1], color=colors[1],label=f'SGD')

	plt.xlabel("# epoches")
	plt.ylabel("entropy loss")
	# plt.ylim([0,0.5])
	plt.title("Train entropy loss vs # epoches")
	plt.grid()
	plt.legend()
	plt.show()
	#plt.save('./lr_effect.png')
	
def k_cross_validation(data_dir, emotions, classifier=None, K=10, n_components=40, learn_rate=0.1):
	dataset, cnt = load_data(data_dir)
	label_weights = np.ones(len(emotions))
	# Xs, Ys = cross_validation_data(dataset, cnt,emotions, K)
	Xs, Ys, label_weights = unbalanced_cross_validation_data(dataset, cnt,emotions, K)
	# label_weights = np.ones(len(emotions))

	accs_kfold = []
	losses_kfold = []
	train_accs_k_fold = []
	train_losses_k_fold = []
	valid_accs_k_fold = []
	valid_losses_k_fold = []
	sum_cij = np.zeros((len(emotions), len(emotions)))

	for fold in range(K):
		val_id = fold
		test_id = (fold+1)%K
		train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, vh, weights_y = preprocessor(Xs, Ys, val_id, test_id, n_components, label_weights)
		accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = softmax_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, emotions, weights_y, learn_rate)
		# accuracy_test, test_loss, train_accs, train_losses, valid_accs, valid_losses, test_cij = logistic_classifier(fold, train_PCA, train_set_y, valid_PCA, validation_set_y, test_PCA, test_set_y, np.shape(dataset[emotions[0]])[1], np.shape(dataset[emotions[0]])[2], vh, learn_rate)

		train_accs_k_fold.append(np.array(train_accs))
		train_losses_k_fold.append(np.array(train_losses))
		
		valid_accs_k_fold.append(np.array(valid_accs))
		valid_losses_k_fold.append(np.array(valid_losses))
		
		accs_kfold.append(accuracy_test)
		losses_kfold.append(test_loss)

		sum_cij = sum_cij + test_cij
		#print('Fold {}'.format(fold))
		#print('Accuracy {}'.format(accuracy_test))
		#print('Loss {}'.format(test_loss ))
	
	# plot_confusion_matrix2(sum_cij, emotions)
	plotter(train_accs_k_fold, train_losses_k_fold, valid_accs_k_fold, valid_losses_k_fold, n_components, learn_rate)
	
	print('Test accuracy {},{}'.format(np.array(accs_kfold).mean(), np.array(accs_kfold).std()))
	print('Test loss {},{}'.format(np.array(losses_kfold).mean(), np.array(losses_kfold).std()))

def plot_confusion_matrix2(cij_arr, emotions):
	print(emotions)
	df_conf_norm = cij_arr / cij_arr.sum(axis=1)
	
	df_cm = pd.DataFrame(df_conf_norm, index = [i for i in emotions],
	                  columns = [j for j in emotions])
	fig,ax = plt.subplots(1,figsize = (10,8))
	im = ax.imshow(df_conf_norm)
	ax.figure.colorbar(im, ax = ax)
	for i in range(df_conf_norm.shape[0]):
		for j in range(df_conf_norm.shape[1]):
			ax.text(j,i, f"{df_conf_norm[i,j]}", ha = "center", va = "center")

	# ax.set_xticks(range(len(emotions)))
	# ax.set_yticks(range(len(emotions)))
	ax.set_xticklabels(emotions)
	ax.set_yticklabels(emotions)
	
	#sn.heatmap(df_cm, annot=True)
	plt.show()


def visualize_pcs(shape_0, shape_1, vh):
	for idx in range(4):
		fig = plt.figure()
		plt.title(f'Principal Component : {idx}')
		plt.imshow(vh[:,idx].reshape((shape_0, shape_1)))

def visualize_weights(weights, shape_0, shape_1, vh, emotions):
	X = weights[:-1, :]
	X = np.matmul(vh, X)

	nom = (X-X.min(axis=0))*(256)
	denom = X.max(axis=0) - X.min(axis=0)
	denom[denom==0] = 1
	w_norm = nom/denom
	images = np.reshape(w_norm, (shape_0, shape_1, -1))
	for idx in range(len(emotions)):
		fig = plt.figure()
		plt.title(emotions[idx])
		plt.imshow(images[:,:,idx])


# example on how to use it
if __name__ == '__main__':
	# The relative path to your image directory
	data_dir = "./aligned/"
	dataset, cnt = load_data(data_dir)
	
	# Part 1
	# emotions = ['fear', 'surprise']
	# k_cross_validation(data_dir, emotions)
	# kachra_code(data_dir, emotions)
	# Part 2
	emotions = ['anger','disgust','happiness','surprise','sadness','fear']
	# k_cross_validation(data_dir, emotions)
	# kachra_code(data_dir, emotions)
	kachra_code2(data_dir, emotions)
	
	
