from scipy.misc import *
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imsave

def predictions(weights, inputs):
	return np.dot(inputs, weights)

def cost_function(weights, inputs, targets):
	return (1.0/(2.0*float(len(targets))))*np.sum((predictions(weights,inputs) - targets)**2)

def gradient(weights, inputs, targets):
	return (1.0/(float(len(targets))))*np.dot(inputs.T, (predictions(weights,inputs) - targets))

def percent_accuracy(weights, inputs, test_targets):
	accuracy = 0
	pred = predictions(weights, inputs)
	for i in range(len(pred)):
		if ((abs(pred[i] - 1) < abs(pred[i] + 1))):				# If predicted value is closer to 1 than -1
			accuracy = accuracy + (test_targets[i] == 1)
		else:
			accuracy = accuracy + (test_targets[i] == -1)

	return accuracy/float(len(test_targets))

def get_image_vectors(actors_list, female_act, male_act, training_set_size):
	'''Prepare Training, test, and validation data '''
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
	targets = np.append(np.ones(len(actors_list)/2 * training_set_size),-1*np.ones(len(actors_list)/2 * training_set_size))	# Create the y vector
	validation_targets = np.append(np.ones(len(actors_list)/2 * 10), -1*np.ones(len(actors_list)/2 * 10))
	test_targets = np.append(np.ones(len(actors_list)/2 * 10), -1*np.ones(len(actors_list)/2 * 10))

	return input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets


def get_overfitting_image_vectors(actors_test, female_act_test, male_act_test):
	'''Prepare overfitting test data '''
	overfit_test_inputs = []
	for actor_name in actors_test:
		last_name = actor_name.split()[1].lower()

		if actor_name in female_act_test:
			# Retrieve list of image path names
			test_files = glob.glob('cropped/female/' + last_name + '*')
			np.random.shuffle(test_files)				# Randomize list of files so that there is no bias
		elif actor_name in male_act_test:
			# Retrieve list of image path names
			test_files = glob.glob('cropped/male/' + last_name + '*')
			np.random.shuffle(test_files)				# Randomize list of files so that there is no bias

		for i in range(0,10):							# Load the 10 overfit test images
			img = imread(test_files[i])					# Load image into numpy matrix
			flat_img = img.flatten()			# Flatten Image into single vector
			flat_img = flat_img/255.0					# Approximately normalize image
			flat_img = np.append(flat_img, 1)				# Append Bias Unit
			overfit_test_inputs.append(flat_img)				# Append image to vector containing all images

	overfit_test_input_mat = np.vstack(overfit_test_inputs)
	overfit_test_targets = np.append(np.ones(len(actors_test)/2 * 10), -1*np.ones(len(actors_test)/2 * 10))

	return overfit_test_input_mat, overfit_test_targets

def performance_plot(actors, male_act, female_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000, range = range(2, 140, 2), save_theta = False):
	''' Plot performance of classifiers on the training and validation sets vs the size of the training set'''

	print "Beggining Performance vs Training Size Plot"

	# Set up figure.
	fig1 = plt.figure(figsize=(12, 8), facecolor='white')
	ax = fig1.add_subplot(111, frameon=False)
	size_list = []										#Set up lists used as figure inputs
	validation_accuracies = []
	training_accuracies = []

	for training_set_size in range:
		size_list.append(training_set_size)
		print "Training Set Size:", training_set_size
		input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets = get_image_vectors(actors, male_act, female_act, training_set_size)
		
		theta = initial_theta
		i = 0
		loss = cost_function(theta, input_mat, targets)
		for x in xrange(num_iterations):							# Run gradient descent in a loop
			theta -= learning_rate*gradient(theta, input_mat, targets)
			loss = cost_function(theta, input_mat, targets)

		validation_accuracies.append(percent_accuracy(theta, validation_input_mat, validation_targets)*100)
		training_accuracies.append(percent_accuracy(theta, input_mat, targets)*100)

		if save_theta:
			non_bias1=theta[1:]
			non_bias1.shape = (32,32)
			imsave(str(i) + "theta" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + "Train_set_size:" + str(training_set_size) + ".jpg", non_bias1)

	ax.plot(size_list, validation_accuracies, 'r-', label="Validation Accuracy")
	ax.plot(size_list, training_accuracies, label="Training Accuracy")
	plt.xlabel("Training Set Size")
	plt.ylabel("Performance (% Accuracy)")
	plt.title("Training Set Size vs. Performance, Iterations:" + str(num_iterations) + ", Alpha:" + str(learning_rate))
	plt.legend(loc = 4)
	plt.show(block=False)


def classify(actors, male_act, female_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = False):

	print "Beggining Classification"

	input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets = get_image_vectors(actors, male_act, female_act, training_set_size)
	if overfit_test:
		overfit_test_input_mat, overfit_test_targets = get_overfitting_image_vectors(actors_overfit_test, female_act_overfit_test, male_act_overfit_test)

	train_acc_list = []
	valid_acc_list = []
	train_cost_list = []
	valid_cost_list = []
	iter_list = []
	theta = initial_theta
	i = 0
	#while (percent_accuracy(theta, validation_input_mat, validation_targets)*100 < 90):
	for j in xrange(num_iterations):
		# Run gradient descent in a loop	
		theta = theta - learning_rate*gradient(theta, input_mat, targets)

		train_acc_list.append(percent_accuracy(theta, input_mat, targets)*100)
		valid_acc_list.append(percent_accuracy(theta, validation_input_mat, validation_targets)*100)
		train_cost_list.append(cost_function(theta, input_mat, targets))				# For analysis of cost in training and validation
		valid_cost_list.append(cost_function(theta, validation_input_mat, validation_targets))
		iter_list.append(i)

		if i%(num_iterations/4)==0:
			print "Iteration:", i
			print "Training Cost:", cost_function(theta, input_mat, targets)
			print "Validation Cost:", cost_function(theta, validation_input_mat, validation_targets)
			print "Percent Validation Accuracy:", percent_accuracy(theta, validation_input_mat, validation_targets)*100,"%"

		i = i+1	

	if save_theta:
		non_bias1=theta[1:]
		non_bias1.shape = (32,32)
		imsave(str(i) + "theta" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias1)

	print "Percent Test Accuracy:", percent_accuracy(theta, test_input_mat, test_targets)*100,"%"
	print "Percent Validation Accuracy:", percent_accuracy(theta, validation_input_mat, validation_targets)*100,"%"
	print "Percent Train Accuracy:", percent_accuracy(theta, input_mat, targets)*100,"%"
	if overfit_test:
		print "Percent Overfit Test Accuracy:", percent_accuracy(theta, overfit_test_input_mat, overfit_test_targets)*100,"%"

	return iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list

if __name__ == '__main__':

	two_actors =['Bill Hader', 'Steve Carell']
	female_act =[]
	male_act = ['Bill Hader', 'Steve Carell']

	six_actors =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	female_act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
	male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

	actors_overfit_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
	female_act_overfit_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
	male_act_overfit_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']

	np.random.seed(0)
	alpha = 0.003		# Learning rate, originally 0.002
	num_iterations = 2000

	# ************ PART 3 ****************
	# Run classification on just steve carell and bill hader
	# iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify(two_actors, female_act, male_act, learning_rate = alpha, initial_theta = 0.5*np.zeros(1025), num_iterations = num_iterations, training_set_size = 100, overfit_test = False, save_theta=True)

	# # Plot the cost and accuracies of the training and validation on the carell and hader classification
	# fig0 = plt.figure(figsize=(12, 8), facecolor='white')
	# ax0 = fig0.add_subplot(111, frameon=False)
	# ax0.plot(iter_list, train_cost_list, 'r-', label="Training Cost")
	# ax0.plot(iter_list, valid_cost_list, label="Validation Cost")
	# plt.xlabel("Iteration")
	# plt.ylabel("Cost")
	# plt.title("Iteration vs. cost, Iterations = " + str(num_iterations) + ", Alpha = " + str(alpha))
	# plt.legend()
	# plt.show(block=False)
	# fig2 = plt.figure(figsize=(12, 8), facecolor='white')
	# ax0 = fig2.add_subplot(111, frameon=False)
	# ax0.plot(iter_list, train_acc_list, 'r-', label="Training Accuracy")
	# ax0.plot(iter_list, valid_acc_list, label="Validation Accuracy")
	# plt.xlabel("Iteration")
	# plt.ylabel("Accuracy (%)")
	# plt.title("Iteration vs. Percent Accuracy, Iterations = " + str(num_iterations) + ", Alpha = " + str(alpha))
	# plt.legend()
	# plt.show(block=False)

	# # Plot performance vs. alpha
	# alpha_list = []
	# alpha_accuracy_list1 = []
	# alpha_accuracy_list2 = []
	# for learning_rate in np.arange(0.001, 0.007, 0.001):
	# 	alpha_list.append(learning_rate)
	# 	iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify(two_actors, female_act, male_act, learning_rate = learning_rate, initial_theta = 0.5*np.zeros(1025), num_iterations = num_iterations, training_set_size = 100, overfit_test = False, save_theta=False)
	# 	iter_list, train_cost_list, valid_cost_list2, train_acc_list, valid_acc_list2 = classify(two_actors, female_act, male_act, learning_rate = learning_rate, initial_theta = 0.5*np.zeros(1025), num_iterations = num_iterations*2, training_set_size = 100, overfit_test = False, save_theta=False)
	# 	alpha_accuracy_list1.append(valid_acc_list[-1])
	# 	alpha_accuracy_list2.append(valid_acc_list2[-1])

	# fig3 = plt.figure(figsize=(12, 8), facecolor='white')
	# ax0 = fig3.add_subplot(111, frameon=False)
	# ax0.plot(alpha_list, alpha_accuracy_list1, 'r-', label=str(num_iterations))
	# ax0.plot(alpha_list, alpha_accuracy_list2, label=str(num_iterations*2))
	# plt.xlabel("Learning Rate (alpha)")
	# plt.ylabel("Accuracy (%)")
	# plt.title("Learning Rate vs. Percent Accuracy")
	# plt.legend()
	# plt.show(block=False)


	# ************ PART 4 ************
	# Visualize full theta for all training sets

	# Did that by setting previous classify function parameters to save_theta = True

	# Now, visualize (save) theta for only 2 images in the training set:

	# classify(two_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 2, overfit_test = False, save_theta = True)
	# classify(two_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 4000,training_set_size = 2, overfit_test = False, save_theta = True)

	# *** PART 5 ***
	# Run classification with overfitting test on the six actors
	#classify(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 1000, training_set_size = 100, overfit_test = True, save_theta=True)

	# *** PART 5 b ***
	# Now, run the performance plot on the six actors with the same initial model parameters
	performance_plot(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000, range = range(2, 140, 2), save_theta = True)

	



	
	
	









