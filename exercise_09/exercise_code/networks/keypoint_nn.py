"""Models for facial keypoint detection"""

import torch
import torch.nn as nn

class KeypointModel(nn.Module):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        """
        super().__init__()
        self.hparams = hparams
        
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        # Define all the layers of your CNN
        self.conv1 = nn.Conv2d(1, hparams['conv1_filters'], kernel_size=hparams['conv1_kernel'], stride=1, padding=0)
        self.conv2 = nn.Conv2d(hparams['conv1_filters'], hparams['conv2_filters'], kernel_size=hparams['conv2_kernel'], stride=1, padding=0)
        self.conv3 = nn.Conv2d(hparams['conv2_filters'], hparams['conv3_filters'], kernel_size=hparams['conv3_kernel'], stride=1, padding=0)
        self.conv4 = nn.Conv2d(hparams['conv3_filters'], hparams['conv4_filters'], kernel_size=hparams['conv4_kernel'], stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.drop1 = nn.Dropout(p=hparams['dropout1_rate'])
        self.drop2 = nn.Dropout(p=hparams['dropout2_rate'])
        self.drop3 = nn.Dropout(p=hparams['dropout3_rate'])
        self.drop4 = nn.Dropout(p=hparams['dropout4_rate'])
        
        # Adjusting the input size for fully connected layers based on the reduced filters
        self.fc1 = nn.Linear(hparams['conv4_filters']*5*5, hparams['fc1_units'])
        self.fc2 = nn.Linear(hparams['fc1_units'], hparams['fc2_units'])
        self.fc3 = nn.Linear(hparams['fc2_units'], 30)
        
        if hparams['activation_function'] == 'ELU':
            self.activation = nn.ELU()
        elif hparams['activation_function'] == 'ReLU':
            self.activation = nn.ReLU()
        elif hparams['activation_function'] == 'LeakyReLU':
            self.activation = nn.LeakyReLU()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.drop2(x)
        
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = self.drop3(x)
        
        x = self.activation(self.conv4(x))
        x = self.pool(x)
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.drop4(x)
        
        x = self.activation(self.fc2(x))
        x = self.drop4(x)
        
        x = self.fc3(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(nn.Module):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
