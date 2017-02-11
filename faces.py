''' faces.py. Author: Gabriel Arpino

In order for this code to function, you should have run 
get_data.py which will populate an uncropped and cropped 
folder with uncropped and uncropped images. This code will 
automatically run all parts of the assignment and will save all
images and plots it produces. To run separate parts, just comment out
the part() sections you do not want to run . ALSO: facescrub_actors.txt and facescrub_actresses.txt must be in the current directory'''



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

def finite_differences(weights, inputs, test_targets, perturbation = 0.005):
	''' Approximates the gradient through finite differences and compares it to the gradient function'''
	print "Calculating Finite Difference and Gradient Error"
	total_error = 0
	for row in range(weights.shape[0]):
		for col in range(weights.shape[1]):
			prev_cost = cost_function(weights, inputs, test_targets)
			deriv = gradient(weights, inputs, test_targets)
			weights[row][col] += perturbation
			after_cost = cost_function(weights, inputs, test_targets)
			cost_diff = (after_cost - prev_cost)/float(perturbation)
			total_error += cost_diff - deriv[row][col]

	avg_error = total_error/float(weights.shape[0]*weights.shape[1])
	return avg_error

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
	plt.savefig("Performance_plot_iter:" + str(num_iterations))

def percent_accuracy_multiclass(weights, inputs, test_targets):
	accuracy = 0
	pred = predictions(weights, inputs)
	# See if the max value in the prediction is equal to the max value in the target, aka, it guessed the person correctly
	for i in range(len(pred)):
		if np.argmax(pred[i]) == np.argmax(test_targets[i]):
			accuracy = accuracy + 1

	return accuracy/float(len(test_targets))

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

		# if i%(num_iterations/4)==0:
		# 	print "Iteration:", i
		# 	print "Training Cost:", cost_function(theta, input_mat, targets)
		# 	print "Validation Cost:", cost_function(theta, validation_input_mat, validation_targets)
		# 	print "Percent Validation Accuracy:", percent_accuracy(theta, validation_input_mat, validation_targets)*100,"%"

		i = i+1	

	if save_theta:
		non_bias1=theta[1:]
		non_bias1.shape = (32,32)
		imsave(str(i) + "theta" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + str(training_set_size) + ".jpg", non_bias1)

	print "Final Percent Test Accuracy:", percent_accuracy(theta, test_input_mat, test_targets)*100,"%"
	print "Final Percent Validation Accuracy:", percent_accuracy(theta, validation_input_mat, validation_targets)*100,"%"
	print "FInal Percent Train Accuracy:", percent_accuracy(theta, input_mat, targets)*100,"%"
	if overfit_test:
		print "Final Percent Overfit Test Accuracy:", percent_accuracy(theta, overfit_test_input_mat, overfit_test_targets)*100,"%"

	return iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list

def classify_multiclass(actors, male_act, female_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = False):

	print "Beggining Classification"

	if overfit_test:
		overfit_test_input_mat, overfit_test_targets = get_overfitting_image_vectors(actors_overfit_test, female_act_overfit_test, male_act_overfit_test)
	
	input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets, theta = get_image_vectors_multiclass(actors, male_act, female_act, training_set_size)

	train_acc_list = []
	valid_acc_list = []
	train_cost_list = []
	valid_cost_list = []
	iter_list = []
	i = 0
	for j in xrange(num_iterations):
		# Run gradient descent in a loop	
		theta = theta - learning_rate*gradient(theta, input_mat, targets)

		train_acc_list.append(percent_accuracy_multiclass(theta, input_mat, targets)*100)
		valid_acc_list.append(percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100)
		train_cost_list.append(cost_function(theta, input_mat, targets))				# For analysis of cost in training and validation
		valid_cost_list.append(cost_function(theta, validation_input_mat, validation_targets))
		iter_list.append(i)

		# if i%(num_iterations/4)==0:
		# 	print "Iteration:", i
		# 	print "Training Cost:", cost_function(theta, input_mat, targets)
		# 	print "Validation Cost:", cost_function(theta, validation_input_mat, validation_targets)
		# 	print "Percent Validation Accuracy:", percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100,"%"

		i = i+1	

	if save_theta:
		# Remove the bias value
		non_bias1,non_bias2,non_bias3,non_bias4,non_bias5,non_bias6=theta[1:,0],theta[1:,1],theta[1:,2],theta[1:,3],theta[1:,4],theta[1:,5]
		# Reshape
		non_bias1.shape, non_bias2.shape, non_bias3.shape, non_bias4.shape, non_bias5.shape, non_bias6.shape = (32,32), (32,32), (32,32), (32,32), (32,32), (32,32)
		# Save
		imsave("multi_classifier_" + str(i) + "theta1" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias1)
		imsave("multi_classifier_" + str(i) + "theta2" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias2)
		imsave("multi_classifier_" + str(i) + "theta3" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias3)
		imsave("multi_classifier_" + str(i) + "theta4" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias4)
		imsave("multi_classifier_" + str(i) + "theta5" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias5)
		imsave("multi_classifier_" + str(i) + "theta6" + "1000*Alpha:" + str(1000*learning_rate) + "Iter:" + str(num_iterations) + ".jpg", non_bias6)

	print "Learning rate:", learning_rate, "Num Iterations:", num_iterations
	print "Final Percent Test Accuracy:", percent_accuracy_multiclass(theta, test_input_mat, test_targets)*100,"%"
	print "Final Percent Validation Accuracy:", percent_accuracy_multiclass(theta, validation_input_mat, validation_targets)*100,"%"
	print "Final Percent Train Accuracy:", percent_accuracy_multiclass(theta, input_mat, targets)*100,"%"
	if overfit_test:
		print "Final Percent Overfit Test Accuracy:", percent_accuracy_multiclass(theta, overfit_test_input_mat, overfit_test_targets)*100,"%"

	return iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list


def part3():
	'''************ PART 3 ****************
	Run classification on just steve carell and bill hader'''

	print "PART 3"
	iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify(two_actors, female_act_two, male_act_two, learning_rate = alpha, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000, training_set_size = 100, overfit_test = False, save_theta=True)

	# Plot the cost and accuracies of the training and validation on the carell and hader classification
	fig0 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig0.add_subplot(111, frameon=False)
	ax0.plot(iter_list, train_cost_list, 'r-', label="Training Cost")
	ax0.plot(iter_list, valid_cost_list, label="Validation Cost")
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.title("Iteration vs. cost, Iterations = " + str(2000) + ", Alpha = " + str(alpha))
	plt.legend()
	plt.savefig("Iteration vs cost, Iterations = " + str(2000))
	fig2 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig2.add_subplot(111, frameon=False)
	ax0.plot(iter_list, train_acc_list, 'r-', label="Training Accuracy")
	ax0.plot(iter_list, valid_acc_list, label="Validation Accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy (%)")
	plt.title("Iteration vs. Percent Accuracy, Iterations = " + str(2000) + ", Alpha = " + str(alpha))
	plt.legend()
	plt.savefig("Iteration vs Percent Accuracy, Iterations = " + str(2000))

	#Run classification on just steve carell and bill hader'''
	iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify(two_actors, female_act_two, male_act_two, learning_rate = 0.008, initial_theta = 0.5*np.zeros(1025), num_iterations = 100, training_set_size = 100, overfit_test = False, save_theta=True)

	# Plot the cost and accuracies of the training and validation on the carell and hader classification
	fig0 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig0.add_subplot(111, frameon=False)
	ax0.plot(iter_list, train_cost_list, 'r-', label="Training Cost")
	ax0.plot(iter_list, valid_cost_list, label="Validation Cost")
	plt.xlabel("Iteration")
	plt.ylabel("Cost")
	plt.title("Iteration vs. cost, Iterations = " + str(100) + ", Alpha = " + str(0.008))
	plt.legend()
	plt.savefig("Iteration vs cost, Iterations = " + str(100))
	fig2 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig2.add_subplot(111, frameon=False)
	ax0.plot(iter_list, train_acc_list, 'r-', label="Training Accuracy")
	ax0.plot(iter_list, valid_acc_list, label="Validation Accuracy")
	plt.xlabel("Iteration")
	plt.ylabel("Accuracy (%)")
	plt.title("Iteration vs. Percent Accuracy, Iterations = " + str(100) + ", Alpha = " + str(0.008))
	plt.legend()
	plt.savefig("Iteration vs Percent Accuracy, Iterations = " + str(100))

	# # Plot performance vs. alpha
	alpha_list = []
	alpha_accuracy_list1 = []
	alpha_accuracy_list2 = []
	for learning_rate in np.arange(-0.0001, 0.008, 0.0001):
		alpha_list.append(learning_rate)
		iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify(two_actors, female_act_two, male_act_two, learning_rate = learning_rate, initial_theta = 0.5*np.zeros(1025), num_iterations = num_iterations, training_set_size = 100, overfit_test = False, save_theta=False)
		#iter_list, train_cost_list, valid_cost_list2, train_acc_list, valid_acc_list2 = classify(two_actors, female_act, male_act, learning_rate = learning_rate, initial_theta = 0.5*np.zeros(1025), num_iterations = num_iterations*2, training_set_size = 100, overfit_test = False, save_theta=False)
		alpha_accuracy_list1.append(valid_cost_list[-1])
		#alpha_accuracy_list2.append(valid_cost_list2[-1])
		print "Learning rate", learning_rate

	fig3 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig3.add_subplot(111, frameon=False)
	ax0.plot(alpha_list, alpha_accuracy_list1, 'r-', label= str(num_iterations) + " iterations")
	#ax0.plot(alpha_list, alpha_accuracy_list2, label=str(num_iterations*2) + " iterations")
	plt.xlabel("Learning Rate (alpha)")
	plt.ylabel("Cost")
	plt.title("Learning Rate vs. Cost")
	plt.legend()
	ax0.set_ylim([0,2])
	plt.savefig("Learning Rate vs Cost_iterations" + str(num_iterations))

def part4():
	'''*********************** PART 4 ************
	Visualize full theta for all training sets'''

	#Did that by setting previous classify function parameters to save_theta = True

	#Now, visualize (save) theta for only 2 images in the training set:
	classify(two_actors, female_act_two, male_act_two, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = True)
	classify(two_actors, female_act_two, male_act_two, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 4000,training_set_size = 100, overfit_test = False, save_theta = True)
	classify(two_actors, female_act_two, male_act_two, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 2, overfit_test = False, save_theta = True)
	classify(two_actors, female_act_two, male_act_two, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 4000,training_set_size = 2, overfit_test = False, save_theta = True)

def part5():
	'''********************** PART 5 ***
	Run classification with overfitting test on the six actors'''
	classify(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 1000, training_set_size = 100, overfit_test = True, save_theta=True)
	classify(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 3000, training_set_size = 100, overfit_test = True, save_theta=True)

def part5b():
	'''******************* PART 5 b ***
	Now, run the performance plot on the six actors with the same initial model parameters'''
	performance_plot(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 1000, range = range(2, 140, 2), save_theta = False)
	performance_plot(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000, range = range(2, 140, 2), save_theta = False)

def part6():

	# Gather initial data and set initial parameters
	actors_list = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	female_act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
	male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

	alpha = 0.005				# Learning rate
	m = float(1025)
	training_set_size = 100

	input_mat, validation_input_mat, test_input_mat, targets, validation_targets, test_targets, params = get_image_vectors_multiclass(actors_list, female_act, male_act, training_set_size)


	print "Perturbation of 0.005:", finite_differences(params, input_mat, targets, perturbation = 0.005)
	print "Perturbation of 0.00005", finite_differences(params, input_mat, targets, perturbation=0.00005)


def part7():
	''' ********************PART 7 ***'''
	#Classify the actors and compare performance of learning rates to justify learning rate selection 

	# Sample multi classifier performance with different iterations
	classify_multiclass(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 1000,training_set_size = 100, overfit_test = False, save_theta = False)
	classify_multiclass(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 2000,training_set_size = 100, overfit_test = False, save_theta = False)
	classify_multiclass(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 3000,training_set_size = 100, overfit_test = False, save_theta = False)
	classify_multiclass(six_actors, female_act, male_act, learning_rate = 0.004, initial_theta = 0.5*np.zeros(1025), num_iterations = 4000,training_set_size = 100, overfit_test = False, save_theta = True)


	# Plot the performance rate versus alpha
	alpha_list = []
	alpha_accuracy_list1 = []
	alpha_accuracy_list2 = []
	alpha_accuracy_list3 = []
	for learning_rate in np.arange(-0.0001, 0.008, 0.0005):
		alpha_list.append(learning_rate)
		iter_list, train_cost_list, valid_cost_list, train_acc_list, valid_acc_list = classify_multiclass(six_actors, female_act, male_act, learning_rate = learning_rate, initial_theta = 0.5*np.zeros(1025), num_iterations = 200, training_set_size = 100, overfit_test = False, save_theta = False)
		alpha_accuracy_list1.append(valid_cost_list[-1])
		print "Learning rate", learning_rate

	# Plot these accuracies
	fig3 = plt.figure(figsize=(12, 8), facecolor='white')
	ax0 = fig3.add_subplot(111, frameon=False)
	ax0.plot(alpha_list, alpha_accuracy_list1, 'r-', label=str(2000))
	plt.xlabel("Learning Rate (alpha)")
	plt.ylabel("Validation Cost")
	plt.title("Learning Rate vs. Validation Cost")
	plt.legend()
	ax0.set_ylim([0,2])
	plt.savefig("Multiclass learning rate vs cost")


if __name__ == '__main__':

	two_actors =['Bill Hader', 'Steve Carell']
	female_act_two =[]
	male_act_two = ['Bill Hader', 'Steve Carell']

	six_actors =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
	female_act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
	male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

	actors_overfit_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
	female_act_overfit_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
	male_act_overfit_test = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']

	np.random.seed(1)
	alpha = 0.004		# Learning rate, originally 0.002
	num_iterations = 4000

	part3()
	part4()
	part5()
	part5b()
	part6()
	part7()


	
	
	









