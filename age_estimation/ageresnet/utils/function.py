import argparse
import os
import pprint
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
MAELoss = nn.L1Loss()

def convert_cor_loss_soft(encode): # based on correlation
    s = nn.Softmax(dim=1)
    t = s(encode)
    return (t)
def convert_soft(encode,valrange):
    arr = torch.tensor(range(0,valrange,1)).cuda()
    #encode = torch.tanh(encode)
    s = nn.Softmax(dim=1)
    t = s(encode)
    t = t* arr
    ts = torch.sum(t,dim=1)
    return (ts)

def CS5calc(a, b):
    f = np.abs(b.detach().numpy()-a.detach().numpy())
    f = f <= 5
    return np.sum(f) / len(f)

def train(args, train_loader, model, criterion, optimizer, epoch, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()
    #print((transform.params.detach().cpu())[0])
    model.train()
    print(model.code[0])    
    t_array = torch.tensor(range(0, transform.val_range, 1)).cuda()
    #print(t_array)
    weight_c = torch.tensor([[[[-1.0,1.0]]]]).cuda()
    init=0 
    foldername=("newdumpoutput"+str(args.version)+str(args.init)+"_reqgrad_"+str(args.reqgrad)+"_dsit_"+str(args.dist_weight)+"_const_"+str(args.const_weight)+"/").replace(".","o")
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    for idx, (features, targets, levels) in enumerate(train_loader): 
        targets = np.reshape(targets, (-1, 1))

        features = features.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True).flatten()
        levels = levels.cuda(non_blocking=True)

        out,outcode = model(features)
        if init==0:
             outcodel=out
             targetlist=targets
             init=1
        else:
             outcodel=torch.cat((outcodel,out),dim=0)
             targetlist=torch.cat((targetlist,targets),dim=0)
        #print(transform.params)
        
        if args.loss == "mae" or args.loss == "mse":
            outp = convert_soft(out,transform.val_range)
            loss = criterion(outp, targets.clone().detach().float())
        elif args.transform == "mc":
            loss = criterion(out, targets)
        elif args.loss == "ce":
            #outp = transform.convert_discrete(out)
            loss = criterion(out, targets)
        elif args.loss == "smooth_ce":
            #outp = transform.get_cor(out)
            ytargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(targets,(targets.size(0),1)).float().cuda()-t_array))
            #print(out.size(),ytargets.size())
            loss = criterion(out, ytargets)
        
        else:
            loss = criterion(outcode, levels)
        #distance constraint
        if True:
            temp0=torch.reshape(targets,(targets.size(0),1)).float().cuda()
            #print(out.size())
            distance = torch.cdist(outcode.contiguous(),outcode.contiguous(),p=1)
            #print(distance.size())
            target_distance = torch.cdist(temp0.contiguous(),temp0.contiguous(),p=1)
            #print(target_distance.size())
            ydistance=torch.clamp(((2*target_distance)-distance),min=0)
            #print(ydistance)
            loss+= args.dist_weight*(torch.mean(torch.sum(ydistance,dim=1)))
            #print(torch.mean(torch.sum(ydistance,dim=1)),args.dist_weight)
            #print(targets,target_distance[0])
        #bit trans constraint
        if True:
            ty = (F.conv2d(torch.reshape(model.code.cuda(),(1,1,model.code.size(0),model.code.size(1))),weight_c.cuda(),padding=(0,1)))
            tycode =  ty*ty
            loss+= args.const_weight*(torch.sum(tycode)) 

        if epoch % 1 == 0:
            oute = convert_soft(out,transform.val_range)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


        losses.update(loss.item(), features.size(0))
        if epoch % 10 == 0:
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))
        else:
            maes.update(0, features.size(0))
            cs5s.update(0, features.size(0))

        if idx % 100 == 0:
            msg =   'Epoch: [{0}][{1}/{2}]\t'\
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            epoch, idx, len(train_loader), loss=losses, mae=maes, cs5=cs5s
                        )
            print(msg)
    if (epoch+1)%10==0 or epoch==0:
       f=open(foldername+"/chech"+str(epoch)+".pkl","wb")
       pickle.dump((outcodel.detach().cpu(),targetlist.detach().cpu(),model.code.detach().cpu()),f)
       f.close()
    return maes.avg, losses.avg, cs5s.avg
    

def validate(args, valid_loader, model, criterion, epoch, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()
    t_array = torch.tensor(range(0, transform.val_range, 1)).cuda()

    model.eval()
    with torch.no_grad():
        for idx, (features, targets, levels) in enumerate(valid_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).flatten()
            levels = levels.cuda(non_blocking=True)

            out,outcode = model(features)
            if args.loss == "mae" or args.loss == "mse":
                outp = convert_soft(out,transform.val_range)
                loss = criterion(outp, targets)
            elif args.transform == "mc":
                loss = criterion(out, targets)
            elif args.loss == "ce":
                #outp = transform.convert_discrete(out)
                loss = criterion(out, targets)
            elif args.loss == "smooth_ce":
                #outp = transform.get_cor(out)
                ytargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(targets,(targets.size(0),1)).float().cuda()-t_array))
                loss = criterion(out, ytargets)
            else:
                loss = criterion(outcode, levels)

            oute = convert_soft(out,transform.val_range)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)

            losses.update(loss.item(), features.size(0))
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))

            if idx % 100 == 0:
                msg =   'Epoch: [{0}][{1}/{2}]\t'\
                        'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            epoch, idx, len(valid_loader), loss=losses, mae=maes, cs5=cs5s
                        )
                print(msg)
    
    return maes.avg, losses.avg, cs5s.avg


def test(args, test_loader, model, criterion, transform):
    losses = AverageMeter()
    maes = AverageMeter()
    cs5s = AverageMeter()

    t_array = torch.tensor(range(0, transform.val_range, 1)).cuda()

    model.eval()
    with torch.no_grad():
        for idx, (features, targets, levels) in enumerate(test_loader):
            features = features.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).flatten()
            levels = levels.cuda(non_blocking=True)

            out,outcode = model(features)
            if args.loss == "mae" or args.loss == "mse":
                outp = convert_soft(out,transform.val_range)
                loss = criterion(outp, targets)
            elif args.transform == "mc":
                loss = criterion(out, targets)
            elif args.loss == "ce":
                #outp = transform.convert_discrete(out)
                loss = criterion(out, targets)
            elif args.loss == "smooth_ce":
                #outp = transform.get_cor(out)
                ytargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(targets,(targets.size(0),1)).float().cuda()-t_array))
                loss = criterion(out, ytargets)
            else:
                loss = criterion(outcode, levels)

            oute = convert_soft(out,transform.val_range)
            oute = oute.cpu()
            targets = targets.cpu()
            mae = MAELoss(oute.flatten(), targets)
            cs5 = CS5calc(oute.flatten(), targets)

            
            losses.update(loss.item(), features.size(0))
            maes.update(mae.item(), features.size(0))
            cs5s.update(cs5.item(), features.size(0))
            if idx % 100 == 0:
                msg =   'Loss {loss.val:.5f} ({loss.avg:.5f})\t'\
                        'MAE: {mae.val:.5f} ({mae.avg:.5f})\t'\
                        'CS5: {cs5.val:.5f} ({cs5.avg:.5f})'.format(
                            loss=losses, mae=maes, cs5=cs5s
                        )
                print(msg)
            # handle mae logic

    return maes.avg, losses.avg, cs5s.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
