import struct as st
import numpy as np


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


mnist = read_mnist(1000)

X_train = mnist["training"]["data"]
y_train = mnist["training"]["labels"]
X_test = mnist["testing"]["data"]
y_test = mnist["testing"]["labels"]

weights = np.array(
    (np.random.random_sample(size=54880).reshape(70, 784),
     np.random.random_sample(size=1400).reshape(20, 70),
     np.random.random_sample(size=200).reshape(10, 20)))
biases = np.array(
    (np.random.random_sample(size=(70, 1)),
     np.random.random_sample(size=(20, 1)),
     np.random.random_sample(size=(10, 1))))
activations = np.array(
    (np.zeros(shape=(784, 1)),
     np.zeros(shape=(70, 1)),
     np.zeros(shape=(20, 1)),
     np.zeros(shape=(10, 1))))

def feedforward():
