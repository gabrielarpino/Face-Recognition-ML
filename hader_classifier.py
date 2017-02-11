from scipy.misc import *
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imsave

def predictions(weights, inputs):
	return np.dot(inputs, weights)

def cost_function(weights, inputs):
	return (1.0/(2.0*m))*np.sum((predictions(weights,inputs) - targets)**2)

def gradient(weights, inputs):
	return (1.0/m)*np.dot(inputs.T, (predictions(weights,inputs) - targets))

def percent_accuracy(weights, inputs, test_targets):
	accuracy = 0
	pred = predictions(weights, inputs)
	for i in range(len(pred)):
		if ((abs(pred[i] - 1) < abs(pred[i] + 1))):				# If predicted value is closer to 1 than -1
			accuracy = accuracy + (test_targets[i] == 1)
		else:
			accuracy = accuracy + (test_targets[i] == -1)

	return accuracy/float(len(test_targets))

# Training images range from 0 to 99, so for two actors, that would be 198
actors = ['hader', 'carell']
training_inputs = []
validation_inputs = []
test_inputs = []
for actor_name in actors:

	# Retrieve list of image path names
	training_files = glob.glob('training_set/male/' + actor_name + '*')

	for path in training_files:
		img = imread(path)					# Load image into numpy matrix
		#mopped = imresize(img, (32,32))		# Resize Image
		flat_img = img.flatten()			# Flatten Image into single vector
		flat_img = flat_img/255
		flat_img = np.append(flat_img, 1)				# Append Bias Unit
		#flat_img = flat_img/float((np.sum(flat_img)))
		training_inputs.append(flat_img)				# Append image to vector containing all images

	validation_files = glob.glob('validation_set/male/' + actor_name + '*')

	for path in validation_files:
		img = imread(path)					# Load image into numpy matrix
		#mopped = imresize(img, (32,32))		# Resize Image
		flat_img = img.flatten()			# Flatten Image into single vector
		flat_img = flat_img/255
		flat_img = np.append(flat_img, 1)				# Append Bias Unit
		#flat_img = flat_img/float((np.sum(flat_img)))
		validation_inputs.append(flat_img)				# Append image to vector containing all images

	test_files = glob.glob('test_set/male/' + actor_name + '*')

	for path in test_files:
		img = imread(path)					# Load image into numpy matrix
		#mopped = imresize(img, (32,32))		# Resize Image
		flat_img = img.flatten()			# Flatten Image into single vector
		flat_img = flat_img/255
		flat_img = np.append(flat_img, 1)				# Append Bias Unit
		#flat_img = flat_img/float((np.sum(flat_img)))
		test_inputs.append(flat_img)				# Append image to vector containing all images

input_mat = np.vstack(training_inputs)				# Stack the training_inputs vector so that each image is a row of matrix
validation_input_mat = np.vstack(validation_inputs)
test_input_mat = np.vstack(test_inputs)
targets = np.append(np.ones(100),-1*np.ones(100))	# Create the y vector
validation_targets = np.append(np.ones(10), -1*np.ones(10))
test_targets = np.append(np.ones(10), -1*np.ones(10))
m = float(len(targets))
alpha = 0.01							# Learning rate, originally 0.001

# Establish Initial random theta for all
theta = 1*np.ones(len(flat_img))
loss = cost_function(theta, input_mat)		# A measure of the error
i = 0

#while (percent_accuracy(theta, validation_input_mat, validation_targets)*100 < ):							# Run gradient descent in a loop	

for j in xrange(5000):
	theta -= alpha*gradient(theta, input_mat)
	loss = cost_function(theta, input_mat)

	if i%200==0:
		print "Percent Validation Accuracy:", percent_accuracy(theta, validation_input_mat, validation_targets)*100,"%"
		print "loss:", loss
		non_bias = theta[1:]
		non_bias.shape = (32,32)
		imsave(str(i) + "theta.jpg", non_bias)
	i = i+1	

non_bias = theta[1:]
non_bias.shape = (32,32)
imsave(str(i) + "theta.jpg", non_bias)

print "Percent Test Accuracy:", percent_accuracy(theta, test_input_mat, test_targets)*100,"%"










