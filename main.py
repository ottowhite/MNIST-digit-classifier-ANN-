import struct as st
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(linewidth=1000)

import datetime as dt
 
import warnings
warnings.filterwarnings("error")

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

        for i in range(len(structure)):
            activation.append(np.zeros(shape=(structure[i], 1)))

            activation_gradient.append(np.zeros(shape=(structure[i], 1)))

            if i != 0:
                if weight_path == None:

                    # using Xavier weight initialisation in congruence with the tanh activation function
                    # weight.append(np.random.uniform(low=-(1/np.sqrt(structure[i-1])), high=(1/np.sqrt(structure[i-1])), size=(structure[i], structure[i-1])))
                    if (weight_initialisation == "standard"):
                        weight.append(np.random.normal(loc=0, scale=1, size=(structure[i], structure[i-1])))
                    elif (weight_initialisation == "xavier"):
                        weight.append(np.random.normal(loc=0, scale=1, size=(structure[i], structure[i-1])) * np.sqrt(1 / structure[i-1]))
                    else:
                        weight.append(np.random.normal(loc=0, scale=1, size=(structure[i], structure[i-1])))
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

        for x in range(len(X_test)-1):
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
        self.error = (np.sum(a=pow((output-desired), 2), axis=0) / 10)[0]


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
        if iteration == 0:
            self.record = {
                "iteration": [],
                "class": [],
                "prediction": [],
                "error": []
            }
       
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


def read_mnist_train(no_items_each):
    data_locations = {
        "training": {"data": "data/train-images-idx3-ubyte", "labels": "data/train-labels-idx1-ubyte"}
    }

    all_data = [open(data_locations["training"]["data"], "rb"), open(data_locations["training"]["labels"], "rb")]

    for x in range(0, len(all_data)):
        all_data[x].seek(0)
        st.unpack(">I", all_data[x].read(4))[0]

        if x == 0:
            no_images = st.unpack(">I", all_data[x].read(4))[0]
            no_rows = st.unpack(">I", all_data[x].read(4))[0]
            no_columns = st.unpack(">I", all_data[x].read(4))[0]
            no_bytes = no_images * no_rows * no_columns

            X_train = np.asarray(
                st.unpack(">" + "B" * 784 * no_items_each, all_data[x].read(784 * no_items_each)), dtype=np.float).reshape(
                no_items_each, 784)
        elif x == 1:
            no_labels = st.unpack(">I", all_data[x].read(4))[0]
            y_train = np.asarray(st.unpack(">" + "B" * no_items_each, all_data[x].read(no_items_each)))

    formatted_data = {
        "training": {"data": X_train, "labels": y_train},
    }

    for reader in all_data:
        reader.close()

    return formatted_data


def read_mnist_test(no_items_each):
    data_locations = {
        "testing": {"data": "data/t10k-images-idx3-ubyte", "labels": "data/t10k-labels-idx1-ubyte"},
        "training": {"data": "data/train-images-idx3-ubyte", "labels": "data/train-labels-idx1-ubyte"}
    }

    all_data = [open(data_locations["testing"]["data"], "rb"), open(data_locations["testing"]["labels"], "rb")]

    for x in range(0, len(all_data)):
        all_data[x].seek(0)
        st.unpack(">I", all_data[x].read(4))[0]

        if x == 0:
            no_images = st.unpack(">I", all_data[x].read(4))[0]
            no_rows = st.unpack(">I", all_data[x].read(4))[0]
            no_columns = st.unpack(">I", all_data[x].read(4))[0]
            no_bytes = no_images * no_rows * no_columns

            X_test = np.asarray(
                st.unpack(">" + "B" * 784 * no_items_each, all_data[x].read(784 * no_items_each)), dtype=np.float).reshape(
                no_items_each, 784)
        elif x == 1:
            no_labels = st.unpack(">I", all_data[x].read(4))[0]
            y_test = np.asarray(st.unpack(">" + "B" * no_items_each, all_data[x].read(no_items_each)))

    formatted_data = {
        "testing": {"data": X_test, "labels": y_test}
    }

    for reader in all_data:
        reader.close()

    return formatted_data


def main():
    # read the first 100 items of the mnist dataset, return 2D dict
    def format_mnist_training(no_items):
        global X_train
        global y_train
        
        mnist = read_mnist_train(no_items)
        
        y_train_labels = mnist["training"]["labels"]

        # reshape into an array of column vectors
        X_train = mnist["training"]["data"].reshape(no_items, -1, 1)
        y_train = np.full(shape=(no_items, 10, 1), fill_value=0)

        # reformat y-train to give labels in the same format as the network (column vectors)
        for label, output in zip(y_train_labels, y_train):
            output[label, 0] = 1


    def format_mnist_testing(no_items):
        global X_test
        global y_test

        mnist = read_mnist_test(no_items)
        
        y_test_labels = mnist["testing"]["labels"]

        # reshape into an array of column vectors
        X_test = mnist["testing"]["data"].reshape(no_items, -1, 1)
        y_test = np.full(shape=(no_items, 10, 1), fill_value=0)
        
        for label, output in zip(y_test_labels, y_test):
            output[label, 0] = 1
    


    #################### DATA COLLECTION ######################

    test_name = "vanished_gradients"

    # TEST SIZE
    training_set_size = 20000
    testing_set_size = 10000

    # NETWORK STRUCTURE
    structure = (784, 80, 40, 10)
    activation_function = 'sigmoid'
    weight_initialisation = 'standard'

    # HYPERPARAMETERS
    learning_rate = 0.0001
    momentum = None

    ###########################################################


    format_mnist_training(training_set_size)
    format_mnist_testing(testing_set_size)
    # create network and load up the training data, normalise inputs (between 0 and 1)
    network = MLP(
        structure=structure, 
        activation_function=activation_function,
        weight_initialisation=weight_initialisation,
        weight_path=None, 
        bias_path=None)

    network.load_training_data(X_train, y_train)
    network.normalize(min=0, max=1, set_type='training')

    network.load_testing_data(X_test, y_test)
    network.normalize(min=0, max=1, set_type="testing")

    network.train(
        learning_rate=learning_rate,
        momentum=momentum, 
        verbose=True, 
        record=True)

    network.save_parameters((test_name + "_weights"), (test_name + "_biases"))
    network.save_records(test_name + "_records")

    '''
    if input("\nVisualise records? (y/n) ") == "y":
        network.visualise_records()
    
    if input("\nTest the network? (y/n) ") == "y":
        network.load_testing_data(X_test, y_test)
        network.normalize(min=0, max=1, set_type="testing")
        network.test()
    

    if input("\nEnter the weights filename: ") == "y":
        weight_filename = input("Enter the weights filename: ")
        bias_filename = input("Enter the biases filename: ")

        network.save_parameters(weight_filename, bias_filename)
   
    if input("\nSave records? (y/n) ") == "y":
        records_filename = input("Enter the records filename: ")
        network.save_records(records_filename)

    if input("\nInspect network state? (y/n) ") == "y":
        ipdb.set_trace()
    '''


if __name__ == "__main__":
    main()