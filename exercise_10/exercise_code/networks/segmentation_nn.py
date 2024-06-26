"""SegmentationNN"""
import torch
import torch.nn as nn

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp or {}
        num_filters = self.hp.get('num_filters', 32)
        kernel_size = self.hp.get('kernel_size', 3)
        padding = self.hp.get('padding', 1)
        stride = self.hp.get('stride', 1)

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(num_filters * 8, num_filters * 16, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters * 16, num_filters * 16, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 16, num_filters * 8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.adjust = nn.Conv2d(num_filters * 2, num_classes, kernel_size=1)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.adjust(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")