import re
from numpy import exp, array, random, dot, argmax, argmin, insert, delete, sqrt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 1 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 0.5
            # # inputs = x, # neurons = y
            # ^-- x-by-y array of random numbers from [-0.5, 0.5):


class NeuralNetwork():
    def __init__(self, layer2, layer1):
        self.layer2 = layer2
        self.layer1 = layer1
        self.mse = []
        self.min_mse = 1000     # assume that it is MAX at beginning        
        self.alpha = 0.1        # alpha in Exponential decrease (dropoff) rate
        self.eta0 = 0.1         # learning rate at time = 0
        self.learning_rate = 0.1
        self.max_learning_rate = 1
        self.starting_point = 0
        self.seed = 2
        self.save_layers_weights = {}   # by mse
        self.number_of_jumps = 0
        self.replace_bad_sample = 0
        self.bias_input = -50   # add bias to input #abc0
        

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def drop_off_rate(self,time):
        return self.eta0 * exp(-self.alpha * time)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        training_set_inputs_bias = insert(training_set_inputs, 0, self.bias_input, axis=1) # add bias to input #abc0        
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_2, output_from_layer_2_bias , output_from_layer_1 = self.think(training_set_inputs_bias)            
            
            # Calculate the error for layer 1 (The difference between the desired output
            # and the predicted output).
            layer1_error = training_set_outputs - output_from_layer_1
            column_mse = 0.5*(layer1_error**2).mean(axis = 1)        # calculate MSE for every row    
            mean_mse = column_mse.mean()
            self.mse.append(mean_mse)
            if (mean_mse < self.min_mse):
                self.min_mse = mean_mse
                # save weights of 2 layers & its MSE
                self.save_layers_weights[mean_mse] = {'layer2':self.layer2.synaptic_weights,'layer1':self.layer1.synaptic_weights}
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)                


            # Calculate the error for layer 2 (By looking at the weights in layer 2,
            # we can determine by how much layer 2 contributed to the error in layer 1).
            layer2_error = layer1_delta.dot(self.layer1.synaptic_weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2_bias)

            # Calculate how much to adjust the weights by
            layer1_adjustment = output_from_layer_2_bias.T.dot(layer1_delta)
            layer2_delta_bias = delete(layer2_delta,0,axis=1)
            layer2_adjustment = training_set_inputs_bias.T.dot(layer2_delta_bias)
            

            # randomize the weights if local minima
            if (self.learning_rate == self.max_learning_rate):
                self.learning_rate = 0.1    # restore learning rate after randomizing the weights               
                self.starting_point = 0     # restore timer after randomizng the weights
            else:
                self.starting_point += 1
                self.learning_rate = self.drop_off_rate(self.starting_point)
                if (self.starting_point > 25): #abc1
                    if (self.mse[-1] / self.mse[-2] > 0.999):    # if not decrease too much every epoch (epsilon = 0.001)                                                         
                        random.seed(self.seed)
                        self.seed += 1                                                      
                        self.layer2.synaptic_weights = 1 * random.random((11,12)) - 0.5
                        self.layer1.synaptic_weights = 1 * random.random((13,8)) - 0.5
                        self.learning_rate = self.max_learning_rate
                        self.number_of_jumps += 1
                        continue


            # Adjust the weights, eta = 10%
            self.layer1.synaptic_weights += self.learning_rate*layer1_adjustment
            self.layer2.synaptic_weights += self.learning_rate*layer2_adjustment            

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer2 = self.__sigmoid(dot(inputs, self.layer2.synaptic_weights))
        # add bias to hiden layer        
        output_from_layer_2_bias = insert(output_from_layer2, 0, 1.5, axis=1) # add bias #abc2
        
        output_from_layer1 = self.__sigmoid(dot(output_from_layer_2_bias, self.layer1.synaptic_weights))
        return output_from_layer2,output_from_layer_2_bias,output_from_layer1

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 2 (12 neurons, each with 10 inputs and 1 input for bias): ")
        print(self.layer2.synaptic_weights)
        print("    Layer 1 (8 neuron, with 12 inputs and 1 input for bias):")
        print(self.layer1.synaptic_weights)

def get_file_input(filename):    
    with open(filename) as file_object:
        contents = file_object.read()
        contents = contents[1:-1]
        inputs = re.split(r'\) \(', contents)
    t_file_inputs = {}
    for input in inputs:
        input = re.split(r' \((.+)\) ',input)
        input_list = [int(x) for x in input[1].split()]
        output_list = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        output_list[int(input[2])] += 0.6   # make an output be like [0.2 0.2 0.8 0.2..0.2], so its real output is 2
        t_file_inputs[int(input[0])] = {'input':input_list, 'output':output_list, 'output_real':int(input[2])}  
    return t_file_inputs  



if __name__ == "__main__":

    # Get data from classified_set file
    file_inputs = get_file_input('./classified_set.txt')
    
    # Count how many output for every class from classified set
    number_of_outputs = [0,0,0,0,0,0,0,0]
    for i in file_inputs.keys():
        number_of_outputs[file_inputs[i]['output_real']] += 1

    # Separate training set and holdout set from a classified set (60% training, 40% holdout)
    training_set = {}
    holdout_set = {}
    limit_training_set = [round(0.6*x) for x in number_of_outputs]
    for i in file_inputs.keys():        
        if (limit_training_set[file_inputs[i]['output_real']] > 0):
            limit_training_set[file_inputs[i]['output_real']] -= 1
            training_set[i] = file_inputs[i]
        else:
            holdout_set[i] = file_inputs[i]

    # From training set, create input, output for MLP:
    training_set_inputs = []
    training_set_outputs = []
    training_set_real_outputs = []
    for value in training_set.values():
        training_set_inputs.append(value['input'])
        training_set_outputs.append(value['output'])
        training_set_real_outputs.append(value['output_real'])
    training_set_inputs = array(training_set_inputs)
    training_set_outputs = array(training_set_outputs)


    # Seed the random number generator
    random.seed(1)

    # Create layer 2 (12 neurons, each with 10 inputs and 1 input for bias)
    layer2 = NeuronLayer(12, 11)

    # Create layer 1 (8 neurons, each with 12 inputs and 1 input for bias)
    layer1 = NeuronLayer(8, 13)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer2, layer1)

    print("1. MLP architecture")
    print(" - Initial weights:")
    neural_network.print_weights()


    # Train the neural network using the training set.
    # Do it 150,000 times and make small adjustments each time.    
    neural_network.train(training_set_inputs, training_set_outputs, 150000) #abc3
    
    # Get the best weights we have so far
    neural_network.layer2.synaptic_weights = neural_network.save_layers_weights[min(neural_network.save_layers_weights.keys())]['layer2']
    neural_network.layer1.synaptic_weights = neural_network.save_layers_weights[min(neural_network.save_layers_weights.keys())]['layer1']    
    
    print(" - Final weights:")
    neural_network.print_weights()    

    # Test the neural network with holdout set.        
    print(f'2. Test on holdout set:')
    true_all = 0        
    true_positive = [0,0,0,0,0,0,0,0]
    true_negative = [0,0,0,0,0,0,0,0]
    false_positive = [0,0,0,0,0,0,0,0]
    false_negative = [0,0,0,0,0,0,0,0]
    for key, value in holdout_set.items():
        test_case = value['input'] 
        test_case.insert(0,neural_network.bias_input)    
        makeup_test_case = []
        makeup_test_case.append(test_case)
        hidden_state, hidden_state_bias, output = neural_network.think(array(makeup_test_case))
        if (argmax(output) == value['output_real']):
            true_all += 1
            for i in range(0,8):
                if i == value['output_real']:
                    true_positive[i] +=1
                else:
                    true_negative[i] += 1
        else:
            for i in range(0,8):
                if i == value['output_real']:
                    false_positive[i] +=1
                elif i == argmax(output):
                    false_negative[i] += 1 
                else:
                    true_negative[i] += 1            
    print(f'Total vectors: {true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i]}')
    for i in range(0,8):        
        print(f'Class {i+1} (value {i}):')
        print(f'    Accuracy = {(true_positive[i] + true_negative[i])/(true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i])}')
        print(f'    Error = {(false_positive[i] + false_negative[i])/(true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i])}')        
        print(f'    Precision = {(true_positive[i])/(true_positive[i] + false_positive[i])}')
        print(f'    Recall = TPR = Sensitivity = {((true_positive[i])/(true_positive[i] + false_negative[i])) if (true_positive[i] + false_negative[i]) != 0 else 0}')
        print(f'    FPR = {(false_positive[i])/(true_negative[i] + false_positive[i])}')
        print(f'    Specificity = {(true_negative[i])/(true_negative[i] + false_positive[i])}')            

    
    # Test the neural network with validation set.   
    print(f'3. Test on validation set:')          
    true_all = 0        
    true_positive = [0,0,0,0,0,0,0,0]
    true_negative = [0,0,0,0,0,0,0,0]
    false_positive = [0,0,0,0,0,0,0,0]
    false_negative = [0,0,0,0,0,0,0,0]
    validation_set = get_file_input('validation_set.txt')   # get data from validation_set file
    for key, value in validation_set.items():       
        test_case = value['input'] 
        test_case.insert(0,neural_network.bias_input)    
        makeup_test_case = []
        makeup_test_case.append(test_case)
        hidden_state, hidden_state_bias, output = neural_network.think(array(makeup_test_case))
        if (argmax(output) == value['output_real']):
            true_all += 1
            for i in range(0,8):
                if i == value['output_real']:
                    true_positive[i] +=1
                else:
                    true_negative[i] += 1
        else:
            for i in range(0,8):
                if i == value['output_real']:
                    false_positive[i] +=1
                elif i == argmax(output):
                    false_negative[i] += 1 
                else:
                    true_negative[i] += 1            
    print(f'Total vectors: {true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i]}')
    for i in range(0,8):        
        print(f'Class {i+1} (value {i}):')
        print(f'    Accuracy = {(true_positive[i] + true_negative[i])/(true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i])}')
        print(f'    Error = {(false_positive[i] + false_negative[i])/(true_positive[i] + true_negative[i] + false_positive[i] + false_negative[i])}')        
        print(f'    Precision = {(true_positive[i])/(true_positive[i] + false_positive[i])}')
        print(f'    Recall = TPR = Sensitivity = {((true_positive[i])/(true_positive[i] + false_negative[i])) if (true_positive[i] + false_negative[i]) != 0 else 0}')
        print(f'    FPR = {(false_positive[i])/(true_negative[i] + false_positive[i])}')
        print(f'    Specificity = {(true_negative[i])/(true_negative[i] + false_positive[i])}')          
    
