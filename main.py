from read_mnist import read_mnist
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(linewidth=1000)

import datetime as dt

class MLP:

    def __init__(self, structure, activation_function, weight_initialisation="standard", weight_path=None, bias_path=None):
        self.structure = structure
        self.activation_function = activation_function
        self.weight_initialisation = weight_initialisation
        self.NUMBER_OF_LAYERS = len(structure)-1  # (excluding input)

        # the accumulator needs a zero initialisation to work on the first training example
        self.weight_accumulator = 0
        self.bias_accumulator = 0

        

        activation = []
        weight = []
        bias = []
        weighted_sum = []
        weight_gradient = []
        bias_gradient = []
        activation_gradient = []

        self.record = {
            "iteration": [],
            "class": [],
            "prediction": [],
            "error": []
        }

        for i in range(len(structure)):
            activation.append(np.zeros(shape=(structure[i], 1)))

            activation_gradient.append(np.zeros(shape=(structure[i], 1)))

            if i != 0:
                if weight_path == None:

                    # using Xavier weight initialisation in congruence with the tanh activation function
                    # weight.append(np.random.uniform(low=-(1/np.sqrt(structure[i-1])), high=(1/np.sqrt(structure[i-1])), size=(structure[i], structure[i-1])))
                    if (weight_initialisation == "standard"):
                        weight.append(np.random.uniform(low=-1, high=1, size=(structure[i], structure[i-1])))
                    elif (weight_initialisation == "xavier"):
                        weight.append(np.random.normal(loc=0, scale=1, size=(structure[i], structure[i-1])) * np.sqrt(1 / structure[i-1]))
                    else:
                        weight.append(np.random.uniform(low=-1, high=1, size=(structure[i], structure[i-1])))
                    # weight.append(np.random.normal(loc=0, scale=0, size=(structure[i], structure[i-1])) * np.sqrt(6 / structure[i-1] + structure[i]) )
                
                if bias_path == None:
                    bias.append(np.zeros(shape=(structure[i], 1)))

                weighted_sum.append(np.zeros(shape=(structure[i], 1)))

                weight_gradient.append(np.zeros(shape=(structure[i], structure[i-1])))
                bias_gradient.append(np.zeros(shape=(structure[i], 1)))


        self.activation = np.array(activation)
        self.weight = np.array(weight) if weight_path == None else np.load(weight_path)
        self.bias = np.array(bias)  if bias_path == None else np.load(bias_path)
        self.weighted_sum = np.array(weighted_sum)
        self.weight_gradient = np.array(weight_gradient)
        self.bias_gradient = np.array(bias_gradient)
        self.activation_gradient = np.array(activation_gradient)


    def save_records(self, records_filename):
        try:
            test_set_size, accuracy, mean_error = self.test(internal_call=True)
        except FileNotFoundError:
            test_set_size, accuracy, mean_error = (0, 0, 0)

        self.records_metadata = pd.DataFrame({
            "structure": str(self.structure),
            "activation_function": self.activation_function,
            "weight_initialisation": self.weight_initialisation,
            "training_iterations": [len(self.X_train)],
            "learning_rate": [self.learning_rate],
            "momentum": [self.momentum],
            "test_set_size": [test_set_size],
            "accuracy": [accuracy],
            "mean_error": [mean_error]
        })
        
        records_path = str(self.structure) + "/" + records_filename + ".csv"
        records_metadata_path = str(self.structure) + "/" + records_filename + "_metadata.csv"

        try:
            self.record.to_csv(records_path, index=False)
            self.records_metadata.to_csv(records_metadata_path, index=False)
        except FileNotFoundError:
            import os
            os.mkdir(str(self.structure))
 
            self.record.to_csv(records_path, index=False)
            self.records_metadata.to_csv(records_metadata_path, index=False)


    def train(self, learning_rate, momentum=None, verbose=True, record=True, batch_size=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.momentum = momentum


        # assumes that training data has been loaded and inputs normalised
        if batch_size == None:
            # pure stochastic gradient descent with a batch size of 1
            for x in range(len(self.X_train)):

                self._feed_forward(x)
                self._mean_sum_squared_errors(self.activation[self.NUMBER_OF_LAYERS], self.y_train[x])
                self._record(x) if record else False # records information about iteration, error, class and prediction

                self._calculate_gradients(x)
                self._adjust_network(self.learning_rate)

                self._speak(x) if verbose else False
                    
        else:
            for x in range(len(self.X_train)):

                self._feed_forward(x)
                self._mean_sum_squared_errors(self.activation[self.NUMBER_OF_LAYERS], self.y_train[x])
                self._record(x) if record else False # records information about iteration, error, class and prediction

                self._accumulate_gradients(x)

                if (x + 1) % self.batch_size == 0 and x != 0:
                    self._divide_gradients()
                    self._adjust_network(self.learning_rate)
                    self._reset_all_gradients()
                
                self._speak(x) if verbose else False
                    
            
        
        self.record = pd.DataFrame(self.record) if record else None


    def test(self, internal_call=False): 
        
        total_tests = len(self.X_test)
        correct_tests = 0
        total_error = 0

        for x in range(len(self.X_test)-1):
            self._feed_forward(x, test_set=True)
            self._mean_sum_squared_errors(self.activation[self.NUMBER_OF_LAYERS], self.y_test[x])

            total_error += self.error

            if np.argmax(self.activation[self.NUMBER_OF_LAYERS]) == np.argmax(self.y_test[x]):
                correct_tests += 1

        mean_error = total_error / total_tests
        percent_correct = (correct_tests / total_tests) * 100
 
        if (internal_call == False):
            print(f"\nTest set size: {total_tests}")
            print(f"Accuracy: {percent_correct}")
            print(f"Mean error: {mean_error}")
            

        return (total_tests, percent_correct, mean_error)


    def normalize(self, min=0, max=1, set_type='training'):
        
        # replaces the existing inputs with inputs between the range of 0 and 1 by dividing by given range
        if set_type == "training":
            set_minimum = np.min(self.X_train)
            set_range = np.ptp(self.X_train)

            self.X_train = ((self.X_train - set_minimum) / set_range) * (max - min) + min

        elif set_type == "testing":
            set_minimum = np.min(self.X_test)
            set_range = np.ptp(self.X_test)

            self.X_test = ((self.X_test - set_minimum) / set_range) * (max - min) + min


    def load_training_data(self, X_train, y_train):
        # creates instance variables containing all training data
        self.X_train, self.y_train = (X_train, y_train)


    def load_testing_data(self, X_test, y_test):
        # creates instance variables containing all training data
        self.X_test, self.y_test = (X_test, y_test)


    def save_parameters(self, weight_filename, bias_filename):
        weight_path = str(self.structure) + "/" + weight_filename
        bias_path = str(self.structure) + "/" + bias_filename

        try:
            np.save(weight_path, self.weight)
            np.save(bias_path, self.bias)
        except FileNotFoundError:
            import os
            os.mkdir(str(self.structure))

            np.save(weight_path, self.weight)
            np.save(bias_path, self.bias)


    def _feed_forward(self, index, test_set=False):
        self.activation[0] = self.X_train[index] if test_set is False else self.X_test[index]

        for i in range(0, self.NUMBER_OF_LAYERS):  # loop from the first layer to the one before the last (as I will calc activation of next layer)
            # the current weighted sums equal the matrix multiplication of the weights and activations
            # of the current layer, plus the bias
            self.weighted_sum[i] = np.dot(self.weight[i], self.activation[i]) + self.bias[i]

            self.activation[i+1] = self._activate(x=self.weighted_sum[i], type=self.activation_function)  # changes the activation of the next layer
    

    def _activate(self, x, type='tanh'):
        if type == 'sigmoid':
            try:
                return 1 / (1 + np.exp(-x))
            except RuntimeWarning:
                ipdb.set_trace()
                return 1 / (1 + np.exp(-x))
        elif type == 'tanh':
            return np.tanh(x)
        elif type == 'softsign':
            return x / (1 + np.absolute(x))


    def _activation_gradient(self, x, type='tanh'):
        if type == 'sigmoid':
            return np.exp(-x) / np.power((1 + np.exp(-x)), 2)
        elif type == 'tanh':
            return 1.0 - np.power(np.tanh(x), 2)
        elif type == 'softsign':
            return 1 / np.power((1 + np.absolute(x)), 2)


    def _mean_sum_squared_errors(self, output, desired):  # finds mean sum of the squared errors w/ outputs from feedforward and desired
        self.error = (np.sum(a=pow((output-desired), 2), axis=0) / self.structure[-1])[0]


    def _reset_activation_gradients(self):
        # as the activation gradients are calculated as a sum, they need to be set back to 0
        
        activation_gradient = []
        for i in range(len(self.structure)):
            activation_gradient.append(np.zeros(shape=(self.structure[i], 1)))
        
        self.activation_gradient = np.array(activation_gradient)


    def _reset_all_gradients(self):
        weight_gradient = []
        bias_gradient = []
        activation_gradient = []

        for i in range(len(self.structure)):
            activation_gradient.append(np.zeros(shape=(self.structure[i], 1)))

            if i != 0:
                weight_gradient.append(np.zeros(shape=(self.structure[i], self.structure[i-1])))
                bias_gradient.append(np.zeros(shape=(self.structure[i], 1)))

        self.weight_gradient = np.array(weight_gradient)
        self.bias_gradient = np.array(bias_gradient)
        self.activation_gradient = np.array(activation_gradient)


    def _calculate_gradients(self, current_training_example):
        start_time = dt.datetime.now()
        # note that "__" denotes "with respect to"

        # reset activation gradients before each calculation because they are cumulative
        self._reset_activation_gradients()

        # for calculating gradients of the first set of weights and biases
        # partial derivatives (column vector) of the cost with respect to the outputs
        gradient_cost__output = 2 * (self.activation[-1] - self.y_train[current_training_example])


        # calculate the rest of the weights and biases starting at the layer preceding the last layer
        for layer in reversed(range(len(self.activation)-1)):

            # partial derivatives of the output with respect to the weighted sum

            gradient_activation__weighted_sum = self._activation_gradient(x=self.weighted_sum[layer], type=self.activation_function)

            if layer == (len(self.activation) - 2):
                
                # calculating the derivative that gives access to weights and biases outside the loop massively speeds up calcs
                # because the calculation is vectorised and never needs to repeat
                gateway_derivative = gradient_activation__weighted_sum * gradient_cost__output

                self.bias_gradient[layer] = gateway_derivative
                self.weight_gradient[layer] = self.activation[layer].T * gateway_derivative

                for j in range(len(self.activation[layer + 1])): # iterating over all relative output layers

                    for k in range(len(self.activation[layer])): # iterating over all relative input layers
                        # calculating the gradients of the last set of weights, and previous activations with partial chain rules
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * gateway_derivative[j]
                
            elif layer != (len(self.activation) - 2) and layer > 0:
                
                gateway_derivative = gradient_activation__weighted_sum * self.activation_gradient[layer + 1]
                self.bias_gradient[layer] = gateway_derivative
                self.weight_gradient[layer] = self.activation[layer].T * gateway_derivative

                for j in range(len(self.activation[layer + 1])): # iterating over all relative output layers
                    for k in range(len(self.activation[layer])): # iterating over all relative input layers
            
                        # calculating all of the gradients of the weights and activations of the hidden layers except last layer
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * gateway_derivative[j]

            else:
                gateway_derivative = gradient_activation__weighted_sum * self.activation_gradient[layer + 1]
                self.bias_gradient[layer] = gateway_derivative
                self.weight_gradient[layer] = self.activation[layer].T * gateway_derivative
                    
        
        self.total_time = dt.datetime.now() - start_time
    

    def _accumulate_gradients(self, current_training_example):
        # used for batch gradient descent when the batch size is greater than 1,
        # therefore requiring that the gradients must accumulate to find the mean
        
        # note that "__" denotes "with respect to"

        # for calculating gradients of the first set of weights and biases
        # partial derivatives (column vector) of the cost with respect to the outputs
        gradient_cost__output = 2 * (self.activation[-1] - self.y_train[current_training_example])


        # calculate the rest of the weights and biases starting at the layer preceding the last layer
        for layer in reversed(range(len(self.activation)-1)):

            # partial derivatives of the output with respect to the weighted sums
            gradient_activation__weighted_sum = self._tanh_derivative(self.weighted_sum[layer])

            for j in range(len(self.activation[layer + 1])):
                for k in range(len(self.activation[layer])):
                    
                    if layer == (len(self.activation) - 2):
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * gradient_activation__weighted_sum[j] * gradient_cost__output[j]
                        self.bias_gradient[layer][j] += gradient_activation__weighted_sum[j] * gradient_cost__output[j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * gradient_activation__weighted_sum[j] * gradient_cost__output[j]
                    elif layer != (len(self.activation) - 2) and layer > 0:
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * gradient_activation__weighted_sum[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] += gradient_activation__weighted_sum[j] * self.activation_gradient[layer + 1][j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * gradient_activation__weighted_sum[j] * self.activation_gradient[layer + 1][j]
                    else:
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * gradient_activation__weighted_sum[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] += gradient_activation__weighted_sum[j] * self.activation_gradient[layer + 1][j]


    def _divide_gradients(self):
        # used when calculating the mean of all gradients in batch gradient descent
        
        self.weight_gradient = self.weight_gradient / self.batch_size
        self.bias_gradient = self.bias_gradient / self.batch_size


    def _adjust_network(self, learning_rate):

        if self.momentum is None:
            # using normal gradient descent to update weights

            self.weight -= learning_rate * self.weight_gradient
            self.bias -= learning_rate * self.bias_gradient
        else:
            # using momentum to update weights
            # weight accumulators are initialized to 0 in the contructor

            self.weight_accumulator = self.momentum * self.weight_accumulator + self.weight_gradient
            self.bias_accumulator = self.momentum * self.bias_accumulator + self.bias_gradient

            self.weight -= learning_rate * self.weight_accumulator
            self.bias -= learning_rate * self.bias_accumulator


    def _record(self, iteration):
        # called inside the training loop to create a dictionary of the training history
        # is converted to a dataframe at the end of the training loop
        self.record["iteration"].append(iteration)
        self.record["class"].append(np.argmax(self.y_train[iteration]))
        self.record["prediction"].append(np.argmax(self.activation[self.NUMBER_OF_LAYERS]))
        self.record["error"].append(self.error)
    

    def visualise_records(self):
        smoothing_window = len(self.X_train) // 30 if len(self.X_train) > 60 else 10

        plt.plot(self.record["iteration"], self.record["error"], label="actual")
        plt.plot(self.record["iteration"], self.record["error"].rolling(window=smoothing_window).mean(), label=f"{smoothing_window} item mean smoothing")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend()
        plt.title(f"Training network with structure {self.structure} \nwith learning rate {self.learning_rate} over {len(self.X_train)} iterations\n with {self.momentum} momentum")
        plt.show()


    def _speak(self, iteration):
        # gets the highest index in the column vector, which is also equal to digit value
        print(f"\nIteration:\t{iteration} / {(len(self.X_train) - 1)}")
        print(f"Class:\t\t{np.argmax(self.y_train[iteration])}")
        print(f"Prediction:\t{np.argmax(self.activation[self.NUMBER_OF_LAYERS])}")
        print(f"Error:\t\t{self.error}")
        
        print(f"Calc grad: \t{self.total_time}")
        print(f"ETR: \t\t{self._remaining_time(iteration, self.total_time)}")
    

    def _remaining_time(self, iteration, average_time):
        return (len(self.X_train) - iteration) * average_time


def main():
    mnist = read_mnist(100, 10000)

    #################### DATA COLLECTION ######################
    test = {
        "test1": {
            "structure":                (784, 16, 16, 10),
            "activation_function":      'tanh',
            "weight_initialisation":    'xavier',
            "weight_path":              None,
            "bias_path":                None,
            "learning_rate":            0.0002,
            "momentum":                 0.65}
    }
    ###########################################################

    for testname in test:
        network = MLP(
        structure=test[testname]["structure"], 
        activation_function=test[testname]["activation_function"],
        weight_initialisation=test[testname]["weight_initialisation"],
        weight_path=test[testname]["weight_path"], 
        bias_path=test[testname]["bias_path"])

        network.load_training_data(mnist["training"]["data"], mnist["training"]["labels"])
        network.normalize(min=0, max=1, set_type='training')

        network.load_testing_data(mnist["testing"]["data"], mnist["testing"]["labels"])
        network.normalize(min=0, max=1, set_type="testing")

        network.train(
            learning_rate=test[testname]["learning_rate"],
            momentum=test[testname]["momentum"], 
            verbose=True, 
            record=True)


        network.test()
        network.visualise_records()

        #network.save_parameters((testname + "_weights"), (testname + "_biases"))
        #network.save_records(testname + "_records")


if __name__ == "__main__":
    main()