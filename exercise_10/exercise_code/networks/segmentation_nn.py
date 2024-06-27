"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        # 首先先提取特征
        self.deeplab = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT).backbone
        
        self.classifier = nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True), 
            nn.Dropout(p=0.2),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            
            
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 确保输出维度与输入维度匹配
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1), 
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),  # 确保输出维度与输入维度匹配
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        )
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
        x = self.deeplab(x)['out']
        x = self.classifier(x)
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