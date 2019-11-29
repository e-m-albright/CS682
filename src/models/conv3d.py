"""
TODO create model with 3d conv layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.models import video

from src.defs import weights


def optimizer(model, learning_rate: float = 1e-2):
    return optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )


def criterion():
    return nn.CrossEntropyLoss()


class SingleChannelStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(SingleChannelStem, self).__init__(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


def resnet3d():
    pretrained = False
    progress = True

    return video._video_resnet('r3d_18',
         pretrained, progress,
         block=video.BasicBlock,
         conv_makers=[video.Conv3DSimple] * 4,
         layers=[2, 2, 2, 2],
         stem=SingleChannelStem)


class Net(nn.Module):
    def __init__(self, idim):
        super(Net, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # nn.Conv3d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # nn.Conv3d(128, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # nn.Conv3d(256, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        features_dim_out = 245760
        # 512 * 3 * 3

        self.classifier_layers = nn.Sequential(
            nn.Linear(features_dim_out, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        print(x.shape)
        out = self.feature_layers(x)
        print("PAST OUT", out.shape)
        out = out.view(out.size(0), -1)
        print("RESHAPE", out.shape)
        out = self.classifier_layers(out)
        print("PAST CLASSIFIER", out.shape)
        return out


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        #init.xavier_normal(self.group1.state_dict()['weight'])
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group2.state_dict()['weight'])
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group3.state_dict()['weight'])
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group4.state_dict()['weight'])
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        #init.xavier_normal(self.group5.state_dict()['weight'])

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 2048),               #
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5))
        #init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc3 = nn.Sequential(
            nn.Linear(2048, 32))           #101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4,
            self.group5
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return self.fc3(out)

    def init_weights(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
        # self.apply(weights.random_weight())
        self.apply(weights_init)

