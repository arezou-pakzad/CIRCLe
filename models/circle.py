import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
from models.stargan import load_stargan


class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='vgg16', use_reg=True):
        super(Model, self).__init__(hidden_dim, base)

        self.out_layer = nn.Linear(hidden_dim, config.num_classes)
        self.trans = load_stargan(
            config.gan_path + 'stargan_last_G.ckpt')
        self.trans.eval()

        self.alpha = config.alpha

        self.use_reg = use_reg

    def forward(self, x, y, d=None):
        z = F.relu(self.base(x))
        logits = self.out_layer(z)
        loss = F.cross_entropy(logits, y)
        correct = (torch.argmax(logits, 1) == y).sum().float() / x.shape[0]
        reg = loss.new_zeros([1])
        if self.training:
            if self.use_reg:
                with torch.no_grad():
                    d_new = torch.randint(0, 6, (d.size(0), )).to(d.device)
                    d_onehot = d.new_zeros([d.shape[0], 6])
                    d_onehot.scatter_(1, d[:, None], 1)
                    d_new_onehot = d.new_zeros([d.shape[0], 6])
                    d_new_onehot.scatter_(1, d_new[:, None], 1)
                    x_new = self.trans(x, d_onehot, d_new_onehot)

                z_new = F.relu(self.base(x_new))
                reg = self.alpha * F.mse_loss(z_new, z)

        return loss, reg, correct