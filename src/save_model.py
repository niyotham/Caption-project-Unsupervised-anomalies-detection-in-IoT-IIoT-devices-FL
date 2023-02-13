
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def save_checkpoint_state(epoch, model, optimizer, scheduler, loss, threshold,cid_n=None):
    print('=> saving checkpoint...')
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss_min":min(loss['train']),
        "val_loss_min":min(loss['val']),
        'history': loss,
        'threshold':threshold
    }
    if cid_n != None:
        torch.save(checkpoint, f'data/partitions/{cid_n}/checkpoint1.pth.tar')
    if cid_n==None:
        torch.save(checkpoint, 'checkpoint.pth.tar')
    print('save_checkpoint_state Done')


def load_checkpoint_state(path, device, model, optimizer, scheduler):
    print('=> Loading checkpoint....')
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    epoch = checkpoint["epoch"]
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print('=> Loading checkpoint done.')
    return model, epoch, optimizer, scheduler  
