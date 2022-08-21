import torch
import random
from torch import nn, optim
import argparse
import os, importlib
from tqdm import tqdm
import numpy as np
from torch.utils import data
from util import AverageMeter
from dataset import get_fitz_dataloaders

parser = argparse.ArgumentParser(description='DG')
parser.add_argument('--dataset', type=str, default='FitzPatrick17k')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=114)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='../data/fitz17k/images/all/')
parser.add_argument('--gan_path', type=str, default='saved/stargan/')
parser.add_argument('--model', type=str, default='circle')
parser.add_argument('--base', type=str, default='vgg16')
parser.add_argument('--model_save_dir', type=str, default='saved/model/')
parser.add_argument('--use_reg_loss', type=bool, default=True)
flags = parser.parse_args()

if flags.dataset == 'FitzPatrick17k':
    flags.num_classes = 114

# print setup
print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

device = 'cuda'
# set seed
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.cuda.manual_seed(flags.seed)
torch.cuda.manual_seed_all(flags.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True                                                                  


# Data loader.
train_loader, val_loader, _ = get_fitz_dataloaders(root='../data/fitz17k/images/all/',
                                                            holdout_mode='random_holdout',
                                                            batch_size=flags.batch_size,
                                                            shuffle=False, 
                                                            partial_skin_types=[], 
                                                            partial_ratio=1.0
                                                            )

# load models
model = importlib.import_module('models.' + flags.model) \
    .Model(flags, flags.hidden_dim, flags.base, use_reg=flags.use_reg_loss).to(device)

optim = torch.optim.SGD(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay, momentum=0.9)


def to_device(data):
    for i in range(len(data)):
        data[i] = data[i].to(device)
    return data


best_by_val = 0
best_val_acc = 0.0
best_val_loss = float('inf')
best_by_test = 0
best_test_loss = float('inf')
for epoch in range(flags.epochs):
    print('Epoch {}: Best val loss {}, Best val acc {}'.format(epoch, best_val_loss, best_val_acc))
    lossMeter = AverageMeter()
    regMeter = AverageMeter()
    correctMeter = AverageMeter()
    model.train()
    for data in tqdm(train_loader, ncols=75, leave=False):
        data = to_device(data)
        loss, reg, correct = model(*data)

        optim.zero_grad()
        if flags.use_reg_loss:
            (loss + reg).backward()
        else:
            loss.backward()
        optim.step()

        lossMeter.update(loss.detach().item(), data[0].shape[0])
        regMeter.update(reg.detach().item(), data[0].shape[0])
        correctMeter.update(correct.detach().item(), data[0].shape[0])
        del loss, reg, correct
    print('>>> Training: Loss ', lossMeter, ', Reg ', regMeter, ', Acc ', correctMeter)

    vallossMeter = AverageMeter()
    valregMeter = AverageMeter()
    valcorrectMeter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x, y, d in tqdm(val_loader, ncols=75, leave=False):
            x, y, d = x.to(device), y.to(device), d.to(device)
            loss, reg, correct = model(x, y)

            vallossMeter.update(loss.detach().item(), x.shape[0])
            valregMeter.update(reg.detach().item(), x.shape[0])
            valcorrectMeter.update(correct.detach().item(), x.shape[0])
            del loss, reg, correct
    print('>>> Val: Loss ', vallossMeter, ', Reg ', valregMeter, ', Acc ', valcorrectMeter)

    if valcorrectMeter.float() > best_val_acc:
        best_val_acc = valcorrectMeter.float()
        save_path = os.path.join(flags.model_save_dir, 'epoch{}_acc_{:.3f}.ckpt'.format(epoch, best_val_acc))
        torch.save(model.state_dict(), save_path)
        print('Saved model with highest acc ...')

    torch.cuda.empty_cache()
