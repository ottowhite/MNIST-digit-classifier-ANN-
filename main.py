import struct as st
import numpy as np
import pdb
np.set_printoptions(linewidth=1000)

class MLP:

    def __init__(self):
        self.NUMBER_OF_LAYERS = 3  # excluding input
        
        self.weight = np.array([
            np.random.uniform(low=-1, high=1, size=(70, 784)),  # weights are a random number between -1 and 1
            np.random.uniform(low=-1, high=1, size=(20, 70)), 
            np.random.uniform(low=-1, high=1, size=(10, 20))
            ])

        self.bias = np.array([
            np.random.uniform(low=-1, high=1, size=(70, 1)),   # biases are a random number between -1 and 1
            np.random.uniform(low=-1, high=1, size=(20, 1)),   # implementing all column vectors as
            np.random.uniform(low=-1, high=1, size=(10, 1))    # number x 1 matrices   
            ])  
        
        self.weighted_sum = np.array([
            np.zeros(shape=(70, 1)),
            np.zeros(shape=(20, 1)),
            np.zeros(shape=(10, 1))
            ])

        self.activation = np.array([
            np.zeros(shape=(784, 1)),
            np.zeros(shape=(70, 1)),
            np.zeros(shape=(20, 1)),
            np.zeros(shape=(10, 1))
            ])

        self.weight_gradient = np.array([
            np.zeros(shape=(70, 784)),
            np.zeros(shape=(20, 70)),
            np.zeros(shape=(10, 20))
            ])
        
        self.bias_gradient = np.array([
            np.zeros(shape=(70, 1)),
            np.zeros(shape=(20, 1)),
            np.zeros(shape=(10, 1))
            ])
        
        self.activation_gradient = np.array(
            [np.zeros(shape=(784, 1)),
            np.zeros(shape=(70, 1)),
            np.zeros(shape=(20, 1)),
            np.zeros(shape=(10, 1))
            ])


    def run(self):
        # assumes that training data has been loaded and inputs normalised
        self._feed_forward(0)
        self._mean_sum_squared_errors(self.activation[3], self.y_train[0])
        self._calculate_gradients(0)


    def normalize_inputs(self, range):
        # replaces the existing inputs with inputs between the range of 0 and 1 by dividing by given range
        self.activation[0] = self.activation[0] / range


    def load_training_data(self, X_train, y_train):
        # creates instance variables containing all training data
        self.X_train, self.y_train = (X_train, y_train)


    def _feed_forward(self, index):
        self.activation[0] = self.X_train[index]

        for i in range(0, self.NUMBER_OF_LAYERS):  # loop from the first layer to the one before the last (as I will calc activation of next layer)
            # the current weighted sums equal the matrix multiplication of the weights and activations
            # of the current layer, plus the bias
            self.weighted_sum[i] = np.dot(self.weight[i], self.activation[i]) + self.bias[i]

            # the activation of the next layer equals the sigmoid( weighted sum )
            self.activation[i+1] = self._sigmoid(self.weighted_sum[i])  # changes the activation of the next layer
    

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _sigmoid_derivative(self, x):
        return np.exp(-x) / pow((1 + np.exp(-x)), 2)


    def _mean_sum_squared_errors(self, output, desired):  # finds mean sum of the squared errors w/ outputs from feedforward and desired
        self.error = (np.sum(a=pow((output-desired), 2), axis=0) / 10)[0]


    def _calculate_gradients(self, current_training_example):
        # weights
        for k in range(0, 20):
            for j in range(0, 10):
                self.weight_gradient[2][j] = np.sum(self.activation[2], axis=0)[0] * self._sigmoid_derivative(self.weighted_sum[2][j]) \
                                        * 2 * (self.activation[3][j] - self.y_train[current_training_example][j])
                self.bias_gradient[2][j] = self._sigmoid_derivative(self.weighted_sum[2][j]) * 2 * (self.activation[3][j] - self.y_train[current_training_example][j])


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
                st.unpack(">" + "B" * 784 * no_items_each, all_data[x].read(784 * no_items_each))).reshape(
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
                st.unpack(">" + "B" * 784 * no_items_each, all_data[x].read(784 * no_items_each))).reshape(
                no_items_each, 784)
        elif x == 3:
            no_labels = st.unpack(">I", all_data[x].read(4))[0]
            y_test = np.asarray(st.unpack(">" + "B" * no_items_each, all_data[x].read(no_items_each)))

    formatted_data = {
        "testing": {"data": X_test, "labels": y_test},
        "training": {"data": X_train, "labels": y_train}
    }

    return formatted_data




    # weights
    for k in range(0, 20):
        for j in range(0, 10):
            weight_gradient[2][j] = np.sum(activation[2], axis=0)[0] * sigmoid_derivative(weighted_sum[2][j]) \
                                    * 2 * (output[j] - y[j])
            bias_gradient[2][j] = sigmoid_derivative(weighted_sum[2][j]) * 2 * (activation[3][j] - y[j])


def main():
    # read the first 100 items of the mnist dataset, return 2D dict
    no_items = 100
    mnist = read_mnist(no_items)
    
    y_train_labels = mnist["training"]["labels"]

    # reshape into an array of column vectors
    X_train = mnist["training"]["data"].reshape(no_items, -1, 1)
    y_train = np.full(shape=(100, 10, 1), fill_value=0)

    # reformat y-train to give labels in the same format as the network (column vectors)
    for label, output in zip(y_train_labels, y_train):
        output[label, 0] = 1

    # create network and load up the training data, normalise inputs (between 0 and 1)
    network = MLP()
    network.load_training_data(X_train, y_train)
    network.normalize_inputs(255)

    network.run()

    # gets the highest index in the column vector, which is also equal to digit value
    print(f"Class:\t\t{np.argmax(network.y_train[0])}")
    print(f"Prediction:\t{np.argmax(network.activation[3])}")
    print(f"Error:\t\t{network.error}")


if __name__ == "__main__":
    main()