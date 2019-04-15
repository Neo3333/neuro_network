import numpy as np
import time 
import math

def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def input_to_hidden(input_vector, weight_vector):
	input_list = input_vector.tolist()
	input_list = [1] + input_list
	bias_input_vector = np.array(input_list)
	bias_input_vector = np.matrix(bias_input_vector)
	result = bias_input_vector * weight_vector.T
	result = np.array(result).reshape(-1)
	#print(len(result))
	for i in range(len(result)):
		result[i] = sigmoid(result[i])
	return result

def softmax(input_vector):
	input_vector_array = np.asarray(input_vector).reshape(-1)
	result = []
	denominator  = 0.0
	for i in range(len(input_vector_array)):
		denominator += math.exp(input_vector_array[i])
	for i in range(len(input_vector_array)):
		result.append(math.exp(input_vector_array[i]) / denominator)
	return result

def hidden_to_final(input_vector, weight_vector):
	input_vector_array = np.asarray(input_vector).reshape(-1)
	input_vector_array = np.concatenate(([1],input_vector_array))
	#print(input_vector_array.shape)
	input_vector_matrix = np.matrix(input_vector_array)
	result = input_vector_matrix * weight_vector.T
	result = softmax(result)
	return result

def neuro_network(input_vector, W1, W2):
	hidden = input_to_hidden(input_vector, W1)
	output = hidden_to_final (hidden, W2)
	return output

def classifier(input_vector, W1, W2):
	result = neuro_network (input_vector, W1,W2)
	maximum = 0
	index = 0
	for i in range (len(result)):
		if result[i] > maximum:
			maximum = result[i]
			index = i
	return index

def err_rate (data, label, W1, W2):
	temp = []
	error = 0
	for i in range(len(data)):
		result = classifier(data[i], W1, W2)
		temp.append(result)
		if result != label[i]:
			error += 1
	print(error)
	return error / len(data), temp

if __name__ == "__main__":
	
	X = np.genfromtxt('ps5_data.csv',delimiter=',')
	y = np.genfromtxt('ps5_data-labels.csv',delimiter=',')
	y = y-1
	W1 = np.genfromtxt('ps5_theta1.csv',delimiter=',')
	W2 = np.genfromtxt('ps5_theta2.csv',delimiter=',')
	print(X.shape, y.shape, W1.shape, W2.shape)
	'''
	hidden = input_to_hidden (X[0], W1)
	output = hidden_to_final (hidden, W2)
	print(output)
	'''
	error_rate, result = err_rate(X,y,W1,W2)
	print(error_rate)





	