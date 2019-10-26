import numpy as np
import struct as st

def read_mnist(training_items, testing_items):
    data_locations = {
        "testing": {"data": "data/t10k-images-idx3-ubyte", "labels": "data/t10k-labels-idx1-ubyte"},
        "training": {"data": "data/train-images-idx3-ubyte", "labels": "data/train-labels-idx1-ubyte"}
    }

    data = {
        "testing": {"data": open(data_locations["testing"]["data"], "rb"), "labels": open(data_locations["testing"]["labels"], "rb")},
        "training": {"data": open(data_locations["training"]["data"], "rb"), "labels": open(data_locations["training"]["labels"], "rb")}
    }

    for set_type in data:
        data[set_type]["data"].seek(0)
        data[set_type]["labels"].seek(0)

        data[set_type]["data"].read(16)
        data[set_type]["labels"].read(8)


        if (set_type == "training"):

            X_train = np.asarray(
                st.unpack(">" + "B" * 784 * training_items, data["training"]["data"].read(784 * training_items)), dtype=np.float).reshape(
                training_items, 784)
            y_train = np.asarray(st.unpack(">" + "B" * training_items, data["training"]["labels"].read(training_items)))

            data["training"]["data"].close()
            data["training"]["labels"].close()

        elif (set_type == "testing"):
            X_test = np.asarray(
                st.unpack(">" + "B" * 784 * testing_items, data["testing"]["data"].read(784 * testing_items)), dtype=np.float).reshape(
                testing_items, 784)
            y_test = np.asarray(st.unpack(">" + "B" * testing_items, data["testing"]["labels"].read(testing_items)))

            data["testing"]["data"].close()
            data["testing"]["labels"].close()
        
    mnist = {
        "training":  {"data": X_train, "labels": y_train},
        "testing": {"data": X_test, "labels": y_test}
    }

    mnist["training"]["data"] = mnist["training"]["data"].reshape(training_items, 784, 1)
    mnist["testing"]["data"] = mnist["testing"]["data"].reshape(testing_items, 784, 1)

    temp_training_labels = mnist["training"]["labels"]
    temp_test_labels = mnist["testing"]["labels"]

    mnist["training"]["labels"] = np.zeros(shape=(training_items, 10, 1))
    mnist["testing"]["labels"] = np.zeros(shape=(testing_items, 10, 1))

    
    for label_number, label in zip(temp_training_labels, mnist["training"]["labels"]):
        label[label_number, 0] = 1
    
    for label_number, label in zip(temp_test_labels, mnist["testing"]["labels"]):
        label[label_number, 0] = 1

    return mnist