
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
num_features = 32
classifiers = {}
for i in range(len(filenames)):
	image = Image.open(filenames[i].split()[0])
	lab.append(filenames[i].split()[1])
	if lab[i] not in classifiers:
		classifiers[lab[i]] = np.random.rand(1,num_features)
		classifiers[lab[i]] = classifiers[lab[i]]/np.linalg.norm(classifiers[lab[i]])
	image_gray= image.convert('L')
	arr= np.array(list(image_gray.getdata(band=0)), dtype="int32")
	arr.shape = (image_gray.size[1], image_gray.size[0])
	X.append(arr.ravel())
#print(classifiers)

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

eta = 0.01
num_iter = 100

for _ in range(num_iter):
	i = np.random.randint(num_train)
	x = X_new[i] 
	yj = lab[i]
	denom = 0.0
	vals = []
	for l in classifiers:
		vals.append(np.dot(classifiers[l],x)[0])
	
	vals = np.array(vals)
	logC = -max(vals)
	denom = 0.0
	for i in range(len(classifiers)):
		denom += np.exp(vals [i]+ logC)

	#print(_, denom)
	change = (np.exp(np.dot(classifiers[yj],x) + logC)/denom) * x
	change_yj = -1 * x + change
	for l in classifiers:
		if l == yj:
			classifiers[l] = classifiers[l] - eta * change_yj
		else:
			classifiers[l] = classifiers[l] - eta * change
	for l in classifiers:
		classifiers[l] = classifiers[l]/np.linalg.norm(classifiers[l])

##for l in classifiers:
#	print l, classifiers[l]
## Testing

filenames1 = open(test_file,'r').readlines()

for i in range(len(filenames1)):
	image1 = Image.open(filenames1[i].split()[0])
	image_gray1 = image1.convert('L')
	arr1 = np.array(list(image_gray1.getdata(band=0)), dtype="int32")
	arr1.shape = (image_gray1.size[1], image_gray1.size[0])
	arr1 = arr1.ravel()
	arr1 = arr1 - mean
	## Transform the data
	new_arr = np.matmul(arr1, eigenvectors)
	max_pred = float('-99999999999999999999')
	label = 'default'
	for l in classifiers:
		pred = np.dot(classifiers[l],new_arr)[0]
		if pred > max_pred:
			max_pred = pred
			label = l
	print(label)