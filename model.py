import torch.nn as nn
import torch
from dataloader import num_of_frames,labels_dict
import torchvision.models as models

def save_model(model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc, path):
    torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
                }, path)

def load_model(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_acc = checkpoint['train_acc']
    val_acc = checkpoint['val_acc']
    return train_loss, val_loss, train_acc, val_acc, epoch


class VideoClassifier(nn.Module):
    def __init__(self):
        super(VideoClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        out_chanels = self.model.conv1.out_channels
        in_features = self.model.fc.in_features
        self.model.conv1 = torch.nn.Conv2d(3*num_of_frames,out_chanels,kernel_size=(7,7),stride=(2,2),padding=(3,3))
        self.model.fc = torch.nn.Linear(in_features,len(labels_dict))

    def forward(self, x):
        bs,n_f,c,h,w = x.shape
        frames = x.view(bs,n_f*c,h,w)
        output = self.model(frames)
        return output