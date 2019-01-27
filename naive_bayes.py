import os
import PIL
from PIL import Image 
import numpy as np
import sys
import math


train_file = sys.argv[1]
test_file  = sys.argv[2]
filenames = open(train_file,'r').readlines()
#print("Number of train files", len(filenames))
num_train = len(filenames)

class classData:
	""" Store data for each class"""
	def __init__(self, mean_data, covar_data, number_samples):
		self.mean = mean_data
		variance = []
		for i in range(len(mean_data)):
			variance.append(covar_data[i][i]/(number_samples))
		self.variance = np.array(variance)
		self.prob = math.log(number_samples)

	def __str__(self):
		return "\n" + "mean_data" + str(self.mean) + "variance" + str(self.variance) + "class probability" + str(self.prob)


lab = []
X = []
for i in range(len(filenames)):
	image = Image.open(filenames[i].split()[0])
	lab.append(filenames[i].split()[1])
	image_gray= image.convert('L')
	arr= np.array(list(image_gray.getdata(band=0)), dtype="int32")
	arr.shape = (image_gray.size[1], image_gray.size[0])
	X.append(arr.ravel())

X = np.array(X).astype(float)
x_t = np.transpose(X)
mean = np.zeros((len(x_t)))
## Column wise mean 
for i in range(len(x_t)):
	mean[i] = (sum(x_t[i][:])*1.0)/num_train
	x_t[i][:] -= mean[i] * 1.0
X = np.transpose(x_t)
cov_mat = np.matmul(X,x_t)
## Finding eigen vectors
eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
eigenvectors = eigenvectors
idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:,idx]

num_features = 32
eigenvectors = eigenvectors[:,:num_features]
## Transform eigen vectors
eigenvectors = np.real(np.matmul(x_t, eigenvectors))

## Normalise the eigen vectors

tmp = np.transpose(eigenvectors)
for i in range(num_features):
	tmp[i] = tmp[i]/np.linalg.norm(tmp[i])
eigenvectors = np.transpose(tmp)
# Transform data
X_new = np.matmul(X,eigenvectors)
#print(X_new)

# Separate training samples classwise
train_data = {}
for i in range(num_train):
	label = lab[i]
	if label not in train_data:
		train_data[label] = []
	train_data[label].append(X_new[i])

# Caluclate variance, mean for each class for naive bayes
class_data = {}

for l,x_data in train_data.items():
	u_class = []
	x_data = np.array(x_data)
	num_samples = len(x_data)
	for i in range(num_features):
		mean_i = sum(x_data[:,i])/num_samples
		u_class.append(mean_i)
		x_data[:,i] -= mean_i
	var_data = np.matmul(np.transpose(x_data),x_data)
	class_data[l] = classData(u_class, var_data, num_samples + 1)
	# print(l,class_data[l])
### End of training
# print(mean)

## Start of testing
filenames1 = open(test_file,'r').readlines()
Y = []
for i in range(len(filenames1)):
	image1 = Image.open(filenames1[i].split()[0])
	image_gray1 = image1.convert('L')
	arr1 = np.array(list(image_gray1.getdata(band=0)), dtype="int32")
	arr1.shape = (image_gray1.size[1], image_gray1.size[0])
	arr1 = arr1.ravel()
	arr1 = arr1 - mean
	## Transform the data
	new_arr = np.matmul(arr1, eigenvectors)
	max_pred = float('-999999999')
	label = 'default'
	for l,data in class_data.items():
		pred = 0 
		for j in range(num_features):
			pred += (-(0.5) * (new_arr[j] - data.mean[j])**2)/data.variance[j]
		pred += data.prob
		if pred > max_pred:
			max_pred = pred
			label = l
	Y.append(label)
	print(label)