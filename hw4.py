import numpy as np
import time 
import math

def neuro_unit(input_vector, weight_vector):
	result = 0
	bias_input_vector = [1] + input_vector
	#print (len(bias_input_vector))
	#print(len(weight_vector))
	for i in range(len(bias_input_vector)):
		result += bias_input_vector[i] * weight_vector[i] 
	result = 1 / (1 + math.exp(-result))
	return result

def softmax(input_vector):
	result = []
	denominator  = 0.0
	for i in range(len(input_vector)):
		denominator += math.exp(input_vector[i])
	for i in range(len(input_vector)):
		result.append(math.exp(input_vector[i]) / denominator)
	return result

def hidden_to_final(input_vector, weight_matrix):
	result = []
	bias_input_vector = [1] + input_vector
	for i in range (len(weight_matrix)):
		temp = 0
		for j in range (len(bias_input_vector)):
			temp += bias_input_vector[j] * weight_matrix[i][j]
		result.append(temp)
	result = softmax(result)
	return result

def neuro_network(input_vector, W1, W2):
	hidden_layer = []
	result = []
	input_vector_list = input_vector.tolist()
	#print(type(input_vector_list))
	for i in range (len(W1)):
		hidden_layer.append(neuro_unit(input_vector_list, W1[i]))
	result = hidden_to_final(hidden_layer, W2)
	return result

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
	return error / len(data), temp

def loss(data, label, W1, W2):
	result = 0
	for i in range (len(data)):
		output = neuro_network(data[i], W1, W2)
		for j in range(len(output)):
			if j == label[i]:
				result += math.log(output[j])
			'''
			else:
				result += math.log(1 - output[j])
			'''
	result = result / (len(data)) * -1
	return result



if __name__ == "__main__":
	
	X = np.genfromtxt('ps5_data.csv',delimiter=',')
	y = np.genfromtxt('ps5_data-labels.csv',delimiter=',')
	y = y-1
	W1 = np.genfromtxt('ps5_theta1.csv',delimiter=',')
	W2 = np.genfromtxt('ps5_theta2.csv',delimiter=',')
	print(X.shape, y.shape, W1.shape, W2.shape)
	
	error_rate, result = err_rate(X,y,W1,W2)
	print(error_rate)
	print(loss(X,y,W1,W2))
	
	
	
	










