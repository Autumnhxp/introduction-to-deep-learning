"""SegmentationNN"""
import imp
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.models as models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        self.transforms = transforms.Compose([
            transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.num_classes = num_classes

        # transfer learning if pretrained=True
        self.feature_extractor = models.alexnet(pretrained=True).features
        
        self.classifier = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(7,7),stride=(1,1)), #6*6
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),#128*12*12
            nn.UpsamplingNearest2d(scale_factor=4),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),#64*48*48
            nn.UpsamplingNearest2d(scale_factor=5),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),#32*240*240
            nn.Conv2d(32, self.num_classes , kernel_size=3, stride=1, padding=1),
        )
                                 

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(outputs, targets)
        
    
        _, preds = torch.max(outputs, 1)
        n_correct = (targets == preds).sum()
        return {"loss":loss, "n_correct": n_correct}


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(), self.hparams["learning_rate"])
        return optim

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

        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        x = self.transforms(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

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
