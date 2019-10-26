from main import MLP
from read_mnist import read_mnist
import matplotlib.pyplot as plt
import numpy as np

class AnimatedMLP(MLP):
    def __init__(self, structure, activation_function, weight_initialisation="standard", weight_path=None, bias_path=None):
        super().__init__(structure, activation_function, weight_initialisation, weight_path, bias_path)

        self.record = {
            "iteration": [],
            "error": []
        }


    def _record(self, iteration):
        # called inside the training loop to create a dictionary of the training history
        # is converted to a dataframe at the end of the training loop
        self.record["iteration"].append(iteration)
        self.record["error"].append(self.error)
        
        plt.cla()
        plt.plot(self.record['iteration'], self.record['error'])
        plt.title("My neural network learning live")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Sum of Squared Error")
        plt.pause(0.0001)


def main():
    plt.style.use('dark_background')
    
    mnist = read_mnist(500, 1000)

    network = AnimatedMLP(structure=(784, 16, 16, 10), activation_function='softsign', weight_initialisation='xavier')

    network.load_training_data(mnist['training']['data'], mnist['training']['labels'])
    network.normalize(min=0, max=0, set_type='training')

    network.load_testing_data(mnist['testing']['data'], mnist['testing']['labels'])
    network.normalize(min=0, max=0, set_type='testing')

    network.train(learning_rate=0.000001, momentum=0.65, verbose=True, record=True)
    plt.show()


if __name__ == '__main__':
    main()
