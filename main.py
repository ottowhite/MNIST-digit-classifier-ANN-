import struct as st
import numpy as np
np.set_printoptions(linewidth=1000)

class MLP:

    def __init__(self):
        self.weight = np.array((np.random.uniform(low=-1, high=1, size=54880).reshape(70, 784),  # weights are a random number
                       np.random.uniform(low=-1, high=1, size=1400).reshape(20, 70),    # between 1 and 0
                       np.random.uniform(low=-1, high=1, size=200).reshape(10, 20)))
        self.bias = np.array((np.random.uniform(low=0, high=1, size=(70, 1)),   # biases are a random number between 1 and 0
                        np.random.uniform(low=0, high=1, size=(20, 1)),   # implementing all column vectors as
                        np.random.uniform(low=0, high=1, size=(10, 1))))  # number x 1 matrices
        self.weighted_sum = np.array((np.zeros(shape=(70, 1)),
                                np.zeros(shape=(20, 1)),
                                np.zeros(shape=(10, 1))))

        self.activation = np.array((np.zeros(shape=(784, 1)),
                            np.zeros(shape=(70, 1)),
                            np.zeros(shape=(20, 1)),
                            np.zeros(shape=(10, 1))))

        self.weight_gradient = np.array((np.zeros(shape=54880).reshape(70, 784),
                                    np.zeros(shape=1400).reshape(20, 70),
                                    np.zeros(shape=200).reshape(10, 20)))
        self.bias_gradient = np.array((np.zeros(shape=(70, 1)),
                                np.zeros(shape=(20, 1)),
                                np.zeros(shape=(10, 1))))
        self.activation_gradient = np.array((np.zeros(shape=(784, 1)),
                                        np.zeros(shape=(70, 1)),
                                        np.zeros(shape=(20, 1)),
                                        np.zeros(shape=(10, 1))))


    def run(self):
        self.normalize_inputs(255)
        self._feed_forward(0)
        self._mean_sum_squared_errors(self.activation[3], self._reformat_desired_output(self.y_train))



    def normalize_inputs(self, range):
        self.activation[0] = self.activation[0] / range


    def load_training_data(self, X_train, y_train):
        self.X_train, self.y_train = (X_train, y_train)


    def _feed_forward(self, index):
        self.activation[0] = self.X_train[index].reshape(784, 1)

        for i in range(0, 2+1):  # loop from the first layer to the one before the last (as I will calc next activations)
            self.weighted_sum[i] = np.dot(self.weight[i], self.activation[i]) + self.bias[i]
            self.activation[i+1] = self._sigmoid(self.weighted_sum[i])  # changes the activation of the next layer
    

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def _sigmoid_derivative(self, x):
        return np.exp(-x) / pow((1 + np.exp(-x)), 2)


    def _reformat_desired_output(self, x):  # ex: re-formats a desired output from 3 to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for MSSE
        y = np.zeros(shape=10)
        y[x] = 1
        return y.reshape(10, 1)


    def _mean_sum_squared_errors(self, X, y):  # finds mean sum of the squared errors w/ outputs from feedforward and desired
        self.error = (np.sum(a=pow((X-y), 2), axis=0) / 10)[0]


    def calculate_gradients(self, y):

        # weights
        for k in range(0, 20):
            for j in range(0, 10):
                self.weight_gradient[2][j] = np.sum(self.activation[2], axis=0)[0] * self._sigmoid_derivative(self.weighted_sum[2][j]) \
                                        * 2 * (self.activation[3][j] - y[j])
                self.bias_gradient[2][j] = self._sigmoid_derivative(self.weighted_sum[2][j]) * 2 * (self.activation[3][j] - y[j])
        print(weight_gradient)


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


def create_network():
    global weight
    global bias
    global weighted_sum
    global activation

    global weight_gradient
    global bias_gradient
    global activation_gradient

    weight = np.array((np.random.uniform(low=-1, high=1, size=54880).reshape(70, 784),  # weights are a random number
                       np.random.uniform(low=-1, high=1, size=1400).reshape(20, 70),    # between 1 and 0
                       np.random.uniform(low=-1, high=1, size=200).reshape(10, 20)))
    bias = np.array((np.random.uniform(low=0, high=1, size=(70, 1)),   # biases are a random number between 1 and 0
                     np.random.uniform(low=0, high=1, size=(20, 1)),   # implementing all column vectors as
                     np.random.uniform(low=0, high=1, size=(10, 1))))  # number x 1 matrices
    weighted_sum = np.array((np.zeros(shape=(70, 1)),
                             np.zeros(shape=(20, 1)),
                             np.zeros(shape=(10, 1))))

    activation = np.array((np.zeros(shape=(784, 1)),
                           np.zeros(shape=(70, 1)),
                           np.zeros(shape=(20, 1)),
                           np.zeros(shape=(10, 1))))

    weight_gradient = np.array((np.zeros(shape=54880).reshape(70, 784),
                                np.zeros(shape=1400).reshape(20, 70),
                                np.zeros(shape=200).reshape(10, 20)))
    bias_gradient = np.array((np.zeros(shape=(70, 1)),
                              np.zeros(shape=(20, 1)),
                              np.zeros(shape=(10, 1))))
    activation_gradient = np.array((np.zeros(shape=(784, 1)),
                                    np.zeros(shape=(70, 1)),
                                    np.zeros(shape=(20, 1)),
                                    np.zeros(shape=(10, 1))))


def normalize_inputs(data):
    return data / 255


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x) / pow((1 + np.exp(-x)), 2)


def feed_forward(X):
    activation[0] = normalize_inputs(X).reshape(784, 1)  # squashes all initial activations between 1 and 0

    for x in range(0, 2+1):  # loop from the first layer to the one before the last (as I will calc next activations)
        weighted_sum[x] = np.dot(weight[x], activation[x]) + bias[x]
        activation[x+1] = sigmoid(weighted_sum[x])  # changes the activation of the next layer

    return activation[3]


def reformat_desired_output(x):  # ex: re-formats a desired output from 3 to [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] for MSSE
    y = np.zeros(shape=10)
    y[x] = 1
    return y.reshape(10, 1)


def mean_sum_squared_errors(X, y):  # returns mean sum of the squared errors w/ outputs from feedforward and desired
    return (np.sum(a=pow((X-y), 2), axis=0) / 10)[0]


def calculate_gradients(output, y):

    # weights
    for k in range(0, 20):
        for j in range(0, 10):
            weight_gradient[2][j] = np.sum(activation[2], axis=0)[0] * sigmoid_derivative(weighted_sum[2][j]) \
                                    * 2 * (output[j] - y[j])
            bias_gradient[2][j] = sigmoid_derivative(weighted_sum[2][j]) * 2 * (activation[3][j] - y[j])

'''
mnist = read_mnist(100)

X_train = mnist["training"]["data"]
y_train = mnist["training"]["labels"]
X_test = mnist["testing"]["data"]
y_test = mnist["testing"]["labels"]

create_network()


result = feed_forward(X_train[0])


desired = reformat_desired_output(y_train[0])

error = mean_sum_squared_errors(result, desired)

calculate_gradients(result, desired)
'''


def main():
    mnist = read_mnist(100)
    
    X_train = mnist["training"]["data"]
    y_train = mnist["training"]["labels"]

    network = MLP()
    network.load_training_data(X_train, y_train)

    network.run()

    print(network.error)



if __name__ == "__main__":
    main()