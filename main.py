import struct as st
import numpy as np
import ipdb
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(linewidth=1000)

import warnings
warnings.filterwarnings("error")

class MLP:

    def __init__(self, structure, weight_path=None, bias_path=None):
        self.structure = structure
        self.NUMBER_OF_LAYERS = len(structure)-1  # (excluding input)

        # for when progress should be graphed and recorded
        self.record = {
            "iteration": [],
            "class": [],
            "prediction": [],
            "error": []
        }

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
                    weight.append(np.random.uniform(low=-0.5, high=0.5, size=(structure[i], structure[i-1])))
                
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


    def train(self, learning_rate=0.01, verbose=True, record=True, batch_size=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size


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
                    
            
            self._visualise_records(smoothing_window=10) if record else False
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
                    
            
            self._visualise_records(smoothing_window=10) if record else False
            

    def test(self):
        total_tests = len(self.X_test)
        correct_tests = 0
        total_error = 0

        
        for x in range(len(X_test)-1):
            self._feed_forward(x, test_set=self.X_test)
            self._mean_sum_squared_errors(self.activation[self.NUMBER_OF_LAYERS], self.y_test[x])

            total_error += self.error

            if np.argmax(self.activation[self.NUMBER_OF_LAYERS]) == np.argmax(self.y_test[x]):
                correct_tests += 1

        mean_error = total_error / total_tests
        percent_correct = (correct_tests / total_tests) * 100

        print(f"\nMean error: {mean_error}")
        print(f"Percent accurate: {percent_correct}")


    def normalize_inputs(self, range, set_type="training"):
        # replaces the existing inputs with inputs between the range of 0 and 1 by dividing by given range
        if set_type == "training":
            self.X_train = self.X_train / range
        elif set_type == "testing":
            self.X_test = self.X_test / range


    def load_training_data(self, X_train, y_train):
        # creates instance variables containing all training data
        self.X_train, self.y_train = (X_train, y_train)


    def load_testing_data(self, X_test, y_test):
        # creates instance variables containing all training data
        self.X_test, self.y_test = (X_test, y_test)


    def save_parameters(self, weight_path, bias_path):
        np.save(weight_path, self.weight)
        np.save(bias_path, self.bias)


    def _feed_forward(self, index, test_set=None):
        self.activation[0] = self.X_train[index] if test_set is None else test_set[index]

        for i in range(0, self.NUMBER_OF_LAYERS):  # loop from the first layer to the one before the last (as I will calc activation of next layer)
            # the current weighted sums equal the matrix multiplication of the weights and activations
            # of the current layer, plus the bias
            self.weighted_sum[i] = np.dot(self.weight[i], self.activation[i]) + self.bias[i]

            self.activation[i+1] = self._tanh(self.weighted_sum[i])  # changes the activation of the next layer
    

    def _sigmoid(self, x):
        # approximating the function because when x > 36, _sigmoid(x) rounds to 1,
        # mirrored the effect on the other side for evenness
        # performing logic in this function requires me to use vectorised version
        return 1 / (1 + np.exp(-x))


    def _tanh(self, x):
        return np.tanh(x)


    def _sigmoid_derivative(self, x):
        return np.exp(-x) / np.power((1 + np.exp(-x)), 2)
        
    
    def _tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2


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
        # note that "__" denotes "with respect to"

        # reset activation gradients before each calculation because they are cumulative
        self._reset_activation_gradients()

        # for calculating gradients of the first set of weights and biases
        # partial derivatives (column vector) of the cost with respect to the outputs
        c__o = 2 * (self.activation[-1] - self.y_train[current_training_example])


        # calculate the rest of the weights and biases starting at the layer preceding the last layer
        for layer in reversed(range(len(self.activation)-1)):

            # partial derivatives of the output with respect to the weighted sums
            o__z = self._tanh_derivative(self.weighted_sum[layer])

            # for calculations for the rest of the derivatives
            a__z = self._tanh_derivative(self.weighted_sum[layer])

            for j in range(len(self.activation[layer + 1])):
                for k in range(len(self.activation[layer])):
                    
                    if layer == (len(self.activation) - 2):
                        self.weight_gradient[layer][j][k] = self.activation[layer][k] * o__z[j] * c__o[j]
                        self.bias_gradient[layer][j] = o__z[j] * c__o[j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * o__z[j] * c__o[j]
                    elif layer != (len(self.activation) - 2) and layer > 0:
                        self.weight_gradient[layer][j][k] = self.activation[layer][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] = a__z[j] * self.activation_gradient[layer + 1][j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                    else:
                        self.weight_gradient[layer][j][k] = self.activation[layer][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] = a__z[j] * self.activation_gradient[layer + 1][j]
    

    def _accumulate_gradients(self, current_training_example):
        # used for batch gradient descent when the batch size is greater than 1,
        # therefore requiring that the gradients must accumulate to find the mean
        
        # note that "__" denotes "with respect to"

        # for calculating gradients of the first set of weights and biases
        # partial derivatives (column vector) of the cost with respect to the outputs
        c__o = 2 * (self.activation[-1] - self.y_train[current_training_example])


        # calculate the rest of the weights and biases starting at the layer preceding the last layer
        for layer in reversed(range(len(self.activation)-1)):

            # partial derivatives of the output with respect to the weighted sums
            o__z = self._tanh_derivative(self.weighted_sum[layer])

            # for calculations for the rest of the derivatives
            a__z = self._tanh_derivative(self.weighted_sum[layer])

            for j in range(len(self.activation[layer + 1])):
                for k in range(len(self.activation[layer])):
                    
                    if layer == (len(self.activation) - 2):
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * o__z[j] * c__o[j]
                        self.bias_gradient[layer][j] += o__z[j] * c__o[j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * o__z[j] * c__o[j]
                    elif layer != (len(self.activation) - 2) and layer > 0:
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] += a__z[j] * self.activation_gradient[layer + 1][j]
                        self.activation_gradient[layer][k] += self.weight[layer][j][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                    else:
                        self.weight_gradient[layer][j][k] += self.activation[layer][k] * a__z[j] * self.activation_gradient[layer + 1][j]
                        self.bias_gradient[layer][j] += a__z[j] * self.activation_gradient[layer + 1][j]


    def _divide_gradients(self):
        self.weight_gradient = self.weight_gradient / self.batch_size
        self.bias_gradient = self.bias_gradient / self.batch_size


    def _adjust_network(self, learning_rate):
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient


    def _record(self, iteration):
        self.record["iteration"].append(iteration)
        self.record["class"].append(np.argmax(self.y_train[iteration]))
        self.record["prediction"].append(np.argmax(self.activation[self.NUMBER_OF_LAYERS]))
        self.record["error"].append(self.error)
    

    def _visualise_records(self, smoothing_window=None):
        
        self.record = pd.DataFrame(self.record)

        if smoothing_window is None:
            plt.plot(self.record["iteration"], self.record["error"])
        else:
            plt.plot(self.record["iteration"], self.record["error"], label="actual")
            plt.plot(self.record["iteration"], self.record["error"].rolling(window=smoothing_window).mean(), label=f"{smoothing_window} item mean smoothing")
        plt.xlabel("Iteration")
        plt.ylabel("Error")
        plt.legend()
        plt.title(f"Training network with structure {self.structure} \nwith learning rate {self.learning_rate} over {len(self.X_train)} iterations")
        plt.show()



    def _speak(self, iteration):
        # gets the highest index in the column vector, which is also equal to digit value
        print(f"\nIteration:\t{iteration} / {(len(self.X_train) - 1)}")
        print(f"Class:\t\t{np.argmax(self.y_train[iteration])}")
        print(f"Prediction:\t{np.argmax(self.activation[self.NUMBER_OF_LAYERS])}")
        print(f"Error:\t\t{self.error}")



def read_mnist(no_items_each):
    data_locations = {
        "testing": {"data": "data/t10k-images-idx3-ubyte", "labels": "data/t10k-labels-idx1-ubyte"},
        "training": {"data": "data/train-images-idx3-ubyte", "labels": "data/train-labels-idx1-ubyte"}
    }

    all_data = [open(data_locations["training"]["data"], "rb"), open(data_locations["training"]["labels"], "rb"),
                open(data_locations["testing"]["data"], "rb"), open(data_locations["testing"]["labels"], "rb")]

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

        elif x == 2:
            no_images = st.unpack(">I", all_data[x].read(4))[0]
            no_rows = st.unpack(">I", all_data[x].read(4))[0]
            no_columns = st.unpack(">I", all_data[x].read(4))[0]
            no_bytes = no_images * no_rows * no_columns

            X_test = np.asarray(
                st.unpack(">" + "B" * 784 * no_items_each, all_data[x].read(784 * no_items_each)), dtype=np.float).reshape(
                no_items_each, 784)
        elif x == 3:
            no_labels = st.unpack(">I", all_data[x].read(4))[0]
            y_test = np.asarray(st.unpack(">" + "B" * no_items_each, all_data[x].read(no_items_each)))

    formatted_data = {
        "testing": {"data": X_test, "labels": y_test},
        "training": {"data": X_train, "labels": y_train}
    }

    for reader in all_data:
        reader.close()

    return formatted_data


def main():
    # read the first 100 items of the mnist dataset, return 2D dict
    def format_mnist_training(no_items):
        global X_train
        global y_train
        
        mnist = read_mnist(no_items)
        
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

        mnist = read_mnist(no_items)
        
        y_test_labels = mnist["testing"]["labels"]

        # reshape into an array of column vectors
        X_test = mnist["testing"]["data"].reshape(no_items, -1, 1)
        y_test = np.full(shape=(no_items, 10, 1), fill_value=0)
        
        for label, output in zip(y_test_labels, y_test):
            output[label, 0] = 1
    
    format_mnist_training(6500)
    format_mnist_testing(10000)

    # create network and load up the training data, normalise inputs (between 0 and 1)
    network = MLP(structure=(784, 200, 80, 10), weight_path='weights.npy', bias_path="biases.npy")

    #network.load_training_data(X_train, y_train)
    #network.normalize_inputs(range=255, set_type="training")
    #network.train(learning_rate=0.01, verbose=True, record=True)

    network.load_testing_data(X_test, y_test)
    network.normalize_inputs(range=255, set_type="testing")
    network.test()


    if input("\nSave weights and biases? (y/n) ") == "y":
        network.save_parameters("weights", "biases")



if __name__ == "__main__":
    main()