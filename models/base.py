import torchvision.models as models
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(BaseModel, self).__init__()
        if base == 'alexnet':
            self.base = models.alexnet(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif base == 'resnet50':
            self.base = models.resnet50(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif base == 'resnet18':
            self.base = models.resnet18(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif base == 'vgg16':
            self.base = models.vgg16(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif base == 'densenet121':
            self.base = models.densenet121(pretrained=True)
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=hidden_dim)
        elif base == 'mobilenetv2':
            self.base = models.mobilenet_v2(pretrained=True)
            self.base.classifier[1] = nn.Linear(in_features=self.base.classifier[1].in_features, out_features=hidden_dim)
        elif base == 'mobilenetv3l':
            self.base = models.mobilenet_v3_large(pretrained=True)
            self.base.classifier[3] = nn.Linear(in_features=self.base.classifier[3].in_features, out_features=hidden_dim)
