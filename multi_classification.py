from scipy.misc import *
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imsave

def predictions(weights, inputs):
	return np.dot(inputs, weights)

def cost_function(weights, inputs, test_targets):
	return (1.0/(2.0*m))*np.sum((predictions(weights,inputs) - test_targets)**2)

def gradient(weights, inputs, test_targets):
	return (1.0/m)*np.dot(inputs.T, (predictions(weights,inputs) - test_targets))

def percent_accuracy_multiclass(weights, inputs, test_targets):
	accuracy = 0
	pred = predictions(weights, inputs)
	# See if the max value in the prediction is equal to the max value in the target, aka, it guessed the person correctly
	for i in range(len(pred)):
		if np.argmax(pred[i]) == np.argmax(test_targets[i]):
			accuracy = accuracy + 1

	return accuracy/float(len(test_targets))

def finite_differences(weights, inputs, test_targets, perturbation = 0.005):
	''' Approximates the gradient through finite differences and compares it to the gradient function'''
	print "Calculating Finite Difference and Gradient Error"
	total_error = 0
	for row in range(theta.shape[0]):
		for col in range(theta.shape[1]):
			prev_cost = cost_function(theta, input_mat, test_targets)
			deriv = gradient(theta, input_mat, test_targets)
			theta[row][col] += perturbation
			after_cost = cost_function(theta, input_mat, test_targets)
			cost_diff = (after_cost - prev_cost)/float(perturbation)
			total_error += cost_diff - deriv[row][col]

	avg_error = total_error/float(theta.shape[0]*theta.shape[1])
	return avg_error

def get_image_vectors_multiclass(actors_list, female_act, male_act, training_set_size):
	''' Prepare Training, test, and validation data '''
	training_inputs = []
	validation_inputs = []
	test_inputs = []
	for actor_name in actors_list:
		last_name = actor_name.split()[1].lower()

		if actor_name in female_act:
			# Retrieve list of image path names
			files = glob.glob('cropped/female/' + last_name + '*')
			np.random.shuffle(files)		# Randomly shuffle files to be used as training, validation, and test data
		elif actor_name in male_act:
			# Retrieve list of image path names
			files = glob.glob('cropped/male/' + last_name + '*')
			np.random.shuffle(files)		# Randomly shuffle files to be used as training, validation, and test data
		

		for i in range(len(files)):
			if i == training_set_size + 20:
				break
			img = imread(files[i])					# Load image into numpy matrix
			flat_img = img.flatten()			# Flatten Image into single vector
			flat_img = flat_img/255.0			# Approximate normalization
			flat_img = np.append(flat_img, 1)				# append Bias Unit
			if i<training_set_size:
				training_inputs.append(flat_img)				# Append image to vector containing all images
			elif (i>=training_set_size and i < (training_set_size+10)):
				validation_inputs.append(flat_img)
			elif (i>=(training_set_size+10)):
				test_inputs.append(flat_img)

	input_mat = np.vstack(training_inputs)				# Stack the training_inputs vector so that each image is a row of matrix
	validation_input_mat = np.vstack(validation_inputs)
	test_input_mat = np.vstack(test_inputs)

	# Creates Target arrays
	targets = np.vstack([0,0,0,0,0,0] for i in range(600))		# Create the y vector
	targets[:100, 0], targets[100:200, 1], targets[200:300, 2], targets[300:400, 3], targets[400:500,4], targets[500:600, 5] = 1,1,1,1,1,1
	validation_targets = np.vstack([0,0,0,0,0,0] for i in range(60))		# Create the y vector
	validation_targets[:10, 0], validation_targets[10:20, 1], validation_targets[20:30, 2], validation_targets[30:40, 3], validation_targets[40:50, 4], validation_targets[50:60, 5] = 1,1,1,1,1,1
	test_targets = np.vstack([0,0,0,0,0,0] for i in range(60))		# Create the y vector
	test_targets[:10, 0], test_targets[10:20, 1], test_targets[20:30, 2], test_targets[30:40, 3], test_targets[40:50, 4], test_targets[50:60, 5] = 1,1,1,1,1,1

	# Establish Initial theta for all
	theta = np.vstack([0.2,0.2,0.2,0.2,0.2,0.2] for i in range(len(flat_img)))

	return input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets, theta

#def classify_multiclass_multiclass(actors, male_act, female_act, learning_rate = 0.004, initial_theta = np.zeros(1025), num_iterations = 2000,training_set_size = 100, save_theta = False):

def classify_multiclass(actors, male_act, female_act, multiclass = False, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = False):

	print "Beggining Classification"

	if not multiclass:
		input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets = get_image_vectors(actors, male_act, female_act, training_set_size)
	if overfit_test:
		overfit_test_input_mat, overfit_test_targets = get_overfitting_image_vectors(actors_overfit_test, female_act_overfit_test, male_act_overfit_test)
	if multiclass:
		input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets, theta = get_image_vectors_multiclass(actors, male_act, female_act, training_set_size)

	train_acc_list = []
	valid_acc_list = []
	train_cost_list = []
	valid_cost_list = []
	iter_list = []
	i = 0
	#while (percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100 < 90):
	for j in xrange(num_iterations):
		# Run gradient descent in a loop	
		theta = theta - learning_rate*gradient(theta, input_mat, targets)

		train_acc_list.append(percent_accuracy_multiclass(theta, input_mat, targets)*100)
		valid_acc_list.append(percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100)
		train_cost_list.append(cost_function(theta, input_mat, targets))				# For analysis of cost in training and validation
		valid_cost_list.append(cost_function(theta, validation_input_mat, validation_targets))
		iter_list.append(i)

		if i%(num_iterations/4)==0:
			print "Iteration:", i
			print "Training Cost:", cost_function(theta, input_mat, targets)
			print "Validation Cost:", cost_function(theta, validation_input_mat, validation_targets)
			print "Percent Validation Accuracy:", percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100,"%"

		i = i+1	

	if save_theta:
		non_bias1=theta[1:]
		non_bias1.shape = (32,32)
		imsave(str(i) + "theta" + "Alpha:" + str(learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias1)

	print "Percent Test Accuracy:", percent_accuracy_multiclass(theta, test_input_mat, test_targets)*100,"%"
	print "Percent Validation Accuracy:", percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100,"%"
	print "Percent Train Accuracy:", percent_accuracy_multiclass(theta, input_mat, targets)*100,"%"
	if overfit_test:
		print "Percent Overfit Test Accuracy:", percent_accuracy_multiclass(theta, overfit_test_input_mat, overfit_test_targets)*100,"%"

	return iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list

if __name__ == '__main__':

	np.random.seed(0)

	# Gather initial data and set initial parameters
	actors_list = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	female_act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
	male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

	alpha = 0.005				# Learning rate
	m = float(1025)
	training_set_size = 100

	input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets, theta = get_image_vectors_multiclass_multiclass(actors_list)



	# i = 0
	# theta = theta - alpha*gradient(theta, input_mat, targets)
	# while (percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100 < 86):

	# 	# Run gradient descent in a loop	
	# 	theta = theta - alpha*gradient(theta, input_mat, targets)
	# 	loss = cost_function(theta, input_mat, targets)

	# 	if i%500==0:
	# 		print "LOSS:", cost_function(theta, input_mat, targets)
	# 		print "Percent Validation Accuracy:", percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100,"%"
	# 	i = i+1	

	# non_bias1, non_bias2 = theta[1:,0], theta[1:,1]
	# non_bias1.shape, non_bias2.shape = (32,32),(32,32)
	# imsave(str(i) + "theta1.jpg", non_bias1)
	# imsave(str(i) + "theta2.jpg", non_bias2)

	# print "Percent Test Accuracy:", percent_accuracy_multiclass(theta, test_input_mat, test_targets)*100,"%"

	print "Perturbation of 0.005:", finite_differences(theta, input_mat, targets, perturbation = 0.005)
	print "Perturbation of 0.00005", finite_differences(theta, input_mat, targets, perturbation=0.00005)

	classify_multiclass(actors_list, female_act, male_act, multiclass = True, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = False)
