import numpy as np


# define a sigmoid function for activation. This function is defined on signle value but it can be applied to a numpy
# array as well. sigmoid function is connected with logistic regression in statistics. logit(p) = wx+b is the same as
# p = sigmoid(wx+b).
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# the derivative of a sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Layer:  # <1>
    def __init__(self):
        self.params = []

        self.previous = None
        self.next = None  # <2>

        self.input_data = None
        self.output_data = None  # <3>

        self.input_delta = None
        self.output_delta = None  # <4>

    # <1> Layers are stacked to build a sequential neural network.
    # <2> A layer knows its predecessor ('previous') and its successor ('next').
    # <3> Each layer can persist data flowing into and out of it in the forward pass.
    # <4> A layer holds input and output data for the backward pass.

    def connect(self, layer):  # <1>
        self.previous = layer
        layer.next = self

    # <1> This method connects a layer to its direct neighbours in the sequential network.

    def forward(self):  # <1>
        raise NotImplementedError

    def get_forward_input(self):  # <2>
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data

    def backward(self):  # <3>
        raise NotImplementedError

    def get_backward_input(self):  # <4>
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta

    def clear_deltas(self):  # <5>
        pass

    def update_params(self, learning_rate):  # <6>
        pass

    def describe(self):  # <7>
        raise NotImplementedError

    # <1> Each layer implementation has to provide a function to feed input data forward.
    # <2> input_data is reserved for the first layer, all others get their input from the previous output.
    # <3> Layers have to implement backpropagation of error terms, that is a way to feed input errors backward through the network.
    # <4> Input delta is reserved for the last layer, all other layers get their error terms from their successor.
    # <5> We compute and accumulate deltas per mini-batch, after which we need to reset these deltas.
    # <6> Update layer parameters according to current deltas, using the specified learning_rate.
    # <7> Layer implementations can print their properties.


class ActivationLayer(Layer):  # <1>
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def forward(self):  # <2>
        data = self.get_forward_input()
        self.output_data = sigmoid(data)

    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)  # <3>

    def describe(self):
        print("|--" + self.__class__.__name__)
        print("|-- dimensions: ({},{})".format(self.input_dim, self.output_dim))

    # <1> This activation layer uses the sigmoid function to activate neurons.
    # <2> The forward pass is simply applying the sigmoid to the input data.
    # <3> The backward pass is element-wise multiplication (because the activation is element by element) of the error term with the sigmoid derivative evaluated at the input to this layer.


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim): #<1>
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weight = np.random.randn(output_dim, input_dim) #<2>
        self.bias = np.random.randn(output_dim, 1)

        self.params = [self.weight, self.bias] #<3>

        self.delta_w = np.zeros(self.weight.shape) #<4>
        self.delta_b = np.zeros(self.bias.shape)

    # <1> Dense layers have input and output dimensions.
    # <2> We randomly initialize weight matrix and bias vector. We should not set initial weights as 0 since it will cause issues in computation.
    # <3> The layer parameters consist of weights and bias terms.
    # <4> Deltas for weights and biases are set to zero.


    def forward(self):
        data = self.get_forward_input()
        self.output_data = np.dot(self.weight, data) + self.bias #<1>

    # <1> The forward pass of the dense layer is the affine linear transformation on input data defined by weights and biases.
    # The weight is a matrix with output_dim x input_dim, data is a matrix with input_dim x mini_batch size. Bias is a column vector with output_dim x 1

    def backward(self):
        data = self.get_forward_input()
        delta = self.get_backward_input() # <1>

        self.delta_b += delta  # <2>
        self.delta_w += np.dot(delta, data.transpose()) # <3>
        self.output_delta = np.dot(self.weight.transpose(), delta) # <4>

    # <1> For the backward pass we first get input data and delta. delta is a p by 1 matrix. We fit each individual case each time.
    # <2> delta_b = delta. The += is calculated for each mini batch. Data are fed in case by case.
    # <3> dW = delta_{i+1} f_i. W is a p_{i+1} by p_i matrix.
    # <4> By the relationship delta_i = delta_{i+1} df_{i+1}/df_i, delta_{i+1} has dimension as p_{i+1} by 1, and
    #    df_{i+1}/df_i = W_i which is a p_{i+1} by p_i matrix. The output_delta should be a p_i by n matrix.


    def update_params(self, learning_rate): # <1>
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self): # <2>
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def describe(self): # <3>
        print("|--" + self.__class__.__name__)
        print("|-- dimensions:({},{})".format(self.input_dim, self.output_dim))

    # <1> Using weight and bias deltas we can update model parameters with gradient descent.
    # <2> After updating parameters we should reset all deltas.
    # <3> A dense layer can be described by its input and output dimensions.