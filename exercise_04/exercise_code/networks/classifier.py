
import os
import pickle
import numpy as np

from exercise_code.networks.base_networks import Network


class Classifier(Network):
    """
    Classifier of the form y = sigmoid(X * W)
    """

    def __init__(self, num_features=2):
        super().__init__("classifier")

        self.num_features = num_features 
        self.W = None
        self.cache = None
        
        self.initialize_weights()

    def initialize_weights(self, weights=None):
        """
        Initialize the weight matrix W

        :param weights: optional weights for initialization
        """
        if weights is not None:
            assert weights.shape == (self.num_features + 1, 1), \
                "weights for initialization are not in the correct shape (num_features + 1, 1)"
            self.W = weights
        else:
            self.W = 0.001 * np.random.randn(self.num_features + 1, 1)

    def forward(self, X):
        """
        Performs the forward pass of the model.

        :param X: N x D array of training data. Each row is a D-dimensional point. 
            Note that it is changed to N x (D + 1) to include the bias term.
        :return: Predicted logits for the data in X, shape N x 1
                 1-dimensional array of length N with classification scores.
                 
        Note: This simple neural-network contains TWO consecutive layers: 
        A fully-connected layer and a sigmoid layer.
        """
        assert self.W is not None, "weight matrix W is not initialized"
        # add a column of 1s to the data for the bias term
        batch_size, _ = X.shape
        X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        


        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the model. Note  #
        # that you need to implement the function self.sigmoid() for that.     #
        # Also, save in self.cache an array of all the relevant variables that #
        # you will need to use in the backward() function. E.g.: (X, ...)      #
        #                                                                      #
        
        # Explanation on the code below:       
        # batch_size, _ = X.shape
        # X = np.concatenate((X, np.ones((batch_size, 1))), axis=1)
        # np.concatenate has its first argument as a tuple of arrays to concatenate
        
        # Hint:                                                                #
        #       The function is made up of TWO functions: Affine and sigmoid.  #
        #       The sigmoid is applied to the result of the affine step.       #
        ########################################################################
        
        # Linear affine transformation
        s = np.dot(X, self.W)
        
        # Apply sigmoid activation function
        z = self.sigmoid(s)

        # Cache relevant variables for backward pass
        self.cache = (X, s, z)
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return z 

    def backward(self, dout):
        """
        Performs the backward pass of the model.

        :param dout: N x M array. Upsteam derivative. It is as the same shape of the forward() output.
                     If the output of forward() is z, then it is dL/dz, where L is the loss function.
        :return: dW --> Gradient of the weight matrix, w.r.t the upstream gradient 'dout'. (dL/dw)
        
        Note: Pay attention to the order in which we calculate the derivatives. It is the opposite of the forward pass!
        """
        assert self.cache is not None, "Run a forward pass before the backward pass. Also, don't forget to store the relevat variables\
            such as in 'self.cache = (X, y, ...)"
        dW = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t W --> dW.     #
        # Make sure you've stored ALL needed variables in self.cache. (!!!!)   #
        #                                                                      #
        # Hint 1: It is recommended to follow the TUM article (Section 3) on   #
        # calculating the chain-rule, while dealing with matrix notations:     #
        # https://bit.ly/tum-article                                           #
        #                                                                      #   
        # Hint 2: Remember that the derivative of sigmoid(x) is independent of #
        # x, and could be calculated with the result from the forward pass.    #
        #                                                                      #
        # Hint 3: The argument "dout" stands for the upstream gradeint to this #
        # layer.                                                               #
        ########################################################################
        
        # Retrieve variables stored in cache during the forward pass
        # The variables are unpacked in the exact order they were packed in the cache
        X_with_bias, s, z = self.cache

        # Calculate the gradient of the sigmoid output w.r.t. the linear layer output 's'
        # The derivative of the sigmoid function sigma(s) w.r.t 's' is sigma(s) * (1 - sigma(s))
        # This calculation uses element-wise multiplication with the upstream gradient 'dout'
        ds = z * (1 - z) * dout  # Element-wise multiplication

        # Calculate the gradient of the weights 'W'
        # This is computed by the dot product of the transposed input (including bias) and the gradient from the sigmoid derivative
        # np.dot(X_with_bias.T, ds) performs a matrix multiplication between the transpose of X_with_bias and ds
        # X_with_bias.T changes the shape from (N, D+1) to (D+1, N), and ds is (N, 1), resulting in a (D+1, 1) matrix which matches the shape of W
        dW = np.dot(X_with_bias.T, ds)
        
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return dW # (dL/dW)

    def sigmoid(self, x):
        """
        Computes the ouput of the sigmoid function.

        :param x: input of the sigmoid, np.array of any shape
        :return: output of the sigmoid with same shape as input vector x
        """
        out = None

        ########################################################################
        # TODO:                                                                #
        # Implement the sigmoid function over the input x. Return "out".       #
        out = 1 / (1 + np.exp(-x))
        # Note: The sigmoid() function operates element-wise.                  #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return out

    def save_model(self):
        directory = 'models'
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(
            model,
            open(
                directory +
                '/' +
                self.model_name +
                '.p',
                'wb'))
