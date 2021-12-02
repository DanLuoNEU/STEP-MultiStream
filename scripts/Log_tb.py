import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Writter will output to ./runs/ directory by default
writer = SummaryWriter('runs/Cls-kinetics400-no_context-max1-i3d')

path_log = "/data/Dan/ava_v2_1/cache/Cls-kinetics400-no_context-max1-i3d-two_branch/training-20210614-033610.log"

def main():
    with open(path_log, "r") as f:
        lines = f.readlines()
    
    epoch=0
    for line in lines:
        words = line.split()
        if line.startswith('Epoch'):
            iter = int(words[3])
            lr = float(words[5])
            loss = float(words[7].split('(')[1].split(')')[0])
            print(iter, lr, loss)
            writer.add_scalar('loss', loss, iter)
            writer.add_scalar('learning rate', lr, iter)
        
        elif line.startswith('Iter'):
            epoch += 1
            MEANAP=float(words[3].split('>')[1])
            print(MEANAP)
            writer.add_scalar('MEANAP', MEANAP, epoch)
        
        elif line.startswith('(1)'):
            AP=float(words[2])
            print(AP, epoch)
            writer.add_scalar('AP_p2p', AP, epoch)
        elif line.startswith('(2)'):
            AP=float(words[2])
            print(AP, epoch)
            writer.add_scalar('AP_xfr', AP, epoch)
        elif line.startswith('(3)'):
            AP=float(words[2])
            print(AP, epoch)
            writer.add_scalar('AP_bkgd', AP, epoch)
        # Other format of AP
        elif "PascalBoxes" in line:
            AP = float(line.split(':')[1].split(',')[0])
            print(AP, epoch+1)
            if "p2p" in line:
                writer.add_scalar('AP_p2p', AP, epoch+1)
            elif "xfr" in line:
                writer.add_scalar('AP_xfr', AP, epoch+1)
            elif "bkgd" in line:
                writer.add_scalar('AP_bkgd', AP, epoch+1)
            

if __name__ == "__main__":
    main()
