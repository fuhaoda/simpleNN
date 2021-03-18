import load_mnist
import network
from layers import DenseLayer, ActivationLayer

training_data, test_data = load_mnist.load_data()
training_data = list(training_data)  # <1>
test_data = list(test_data)

# unzip the iterator

net = network.SequentialNetwork()

net.add(DenseLayer(784, 392))
net.add(ActivationLayer(392))
net.add(DenseLayer(392, 196))
net.add(ActivationLayer(196))
net.add(DenseLayer(196, 10))
net.add(ActivationLayer(10))

net.train(training_data, epochs=200, mini_batch_size=100,
          learning_rate=3.0, test_data=test_data)
