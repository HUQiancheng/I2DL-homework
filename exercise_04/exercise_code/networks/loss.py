
import os
import pickle
import numpy as np
# from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth, individual_losses=False):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth, individual_losses=False):
        loss = self.forward(y_out, y_truth, individual_losses)
        return loss


class L1(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None

        result = np.abs(y_out - y_truth)
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1
    
        return gradient / len(y_out)


class MSE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None
        
        result = (y_out - y_truth) ** 2
        
        if individual_losses:
            return result
        return np.mean(result)

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for each sample of your training set.
        """
        gradient = None
        gradient = 2 * (y_out - y_truth) / len(y_out)
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth, individual_losses=False):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model (the Logits).
        :y_truth: [N, ] array ground truth value of your training set.
        :return: 
            - individual_losses=False --> A single scalar, which is the mean of the binary cross entropy loss 
                for each sample of your training set.
            - individual_losses=True  --> [N, ] array of binary cross entropy loss values for each sample of your batch.
        """
        result = None
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the BCE loss     #
        # for each imstance in the batch.                                      #
        
        # This is a way similar to the override in C++                             
        # it is called the inheritance and polymorphism in OOP
        
        # # Slow implementation
        # N = np.size(y_out)
        # if N != np.size(y_truth):
        #     raise ValueError("The size of the predicted values and the ground truth values should be the same")
        # else:
        #     for i in range(N):
        #         result += -y_truth[i] * np.log(y_out[i]) - (1 - y_truth[i]) * np.log(1 - y_out[i])
        # result = result / N
        
        # Fast implementation
        # Ensure y_out and y_truth have the same size
        if y_out.shape != y_truth.shape:
            raise ValueError("The shape of predicted values and ground truth values must match.")
        
        # Compute BCE using vectorized operations
        result = -y_truth * np.log(y_out) - (1 - y_truth) * np.log(1 - y_out)
        # Explanation:
        # 1 - y_out is a vector
        # np.log() return element-wise log of the input
        #                                                                      #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        if individual_losses:
            return result   # return a list of loss values, without taking the mean.
        result =  np.mean(result)

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
        :y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out 
                for each sample of your training set.
        """
        gradient = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient w.r.t to the input  #
        # to the loss function, y_out.                                         #
        #                                                                      #
        
        # fast implementation
        # Ensure y_out and y_truth have the same size
        if y_out.shape != y_truth.shape:
            raise ValueError("The shape of predicted values and ground truth values must match.")

        # Compute gradient using vectorized operations and divide by N to average over the batch
        N = y_out.shape[0]
        gradient = -(y_truth / y_out - (1 - y_truth) / (1 - y_out)) / N
        # Explanation:
        # In NumPy, division is element-wise. This means y_truth / y_out returns an array where each element is 
        # the division of the corresponding elements in y_truth and y_out.
        # last semester i lost 10 marks for not knowing it returns directly the array
        
        # Hint:                                                                #
        #   Don't forget to divide by N, which is the number of samples in     #
        #   the batch. It is crucial for the magnitude of the gradient.        #
        ########################################################################


        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gradient


