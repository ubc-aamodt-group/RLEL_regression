from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import sample

import time
import logging

from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from .evaluation import decode_preds, compute_nme

import seaborn as sns
import matplotlib.pylab as plt
import torch.nn.functional as F
logger = logging.getLogger(__name__)
def return_dict(complete,lcode,tpts):
    #print(lcode.size())
    tpts=torch.reshape(tpts,(lcode.size(0),lcode.size(1)))
    tpts=torch.clamp(tpts,min=0,max=255)
    #print(tpts.size())
    samples=lcode.size(0)
    codelen=lcode.size(2)

    for i in range(0,lcode.size(0)):
        for j in range(0,lcode.size(1)):
                index=(torch.round(tpts[i,j])).long()
                code=torch.reshape(lcode[i,j,:].detach().cpu(),(1,codelen))
                complete[j][index]=torch.cat((complete[j][index],code),0)
    #print(complete[0][64].size())
    return complete

def init_dict(lcode,tpts):
    #print(lcode.size())
    tpts=torch.reshape(tpts,(lcode.size(0),lcode.size(1)))
    tpts=torch.clamp(tpts,min=0,max=255)
    #print(tpts.size())
    samples=lcode.size(0)
    codelen=lcode.size(2)

    complete=[]
    for i in range(0,lcode.size(1)):
        complete.append([])
        for k in range(0,300):
            complete[i].append(torch.ones((1,codelen)))
    for i in range(0,lcode.size(0)):
        for j in range(0,lcode.size(1)):
                index=(torch.round(tpts[i,j])).long()
                code=torch.reshape(lcode[i,j,:].detach().cpu(),(1,codelen))
                complete[j][index]=torch.cat((complete[j][index],code),0)
    return complete
def dump_dict(lfullcode,final_output_dir,suffix):
    
    num_classes=len(lfullcode)
    lcode=lfullcode[8]
    quant=len(lcode)
    codelen=lcode[0].size(1)
    sum_real=torch.zeros((num_classes,quant,codelen))
    mean_real=torch.zeros((num_classes,quant,codelen))
    mean_bin=torch.zeros((num_classes,quant,codelen))
    for nn in tqdm(range(0,num_classes)):
        lcode=lfullcode[nn]
        for i in range(0,quant):
            sum_real[nn,i,:]=torch.sum(lcode[i],dim=0)
            mean_real[nn,i,:]=torch.mean(lcode[i],dim=0)
            mean_bin[nn,i,:]=torch.mean(torch.sign(lcode[i]),dim=0)
    return mean_real,mean_bin

def convert_gen_ori(encode,num_bits,di): # based on correlation
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=3)
    return (ts,ts,ts)
def convert_gen_loss_ori(encode,num_bits,di): # based on correlation
    t = torch.matmul(encode,di)
    return (t)

def convert_correlation_ori(encode,num_bits,di): # based on correlation
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=3)
    t = s(t)
    return (t)

def convert_genex_ori(encode,num_bits,di,arr):
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=3)
    t = s(t)
    t = t* arr
    ts = torch.sum(t,dim=3)
    return (ts)

def convert_gen(encode,num_bits,di): # based on correlation
    _,ts = torch.max(encode,dim=3)
    return (ts,ts,ts)
def convert_gen_loss(encode,num_bits,di): # based on correlation
    return (encode)

def convert_correlation(encode,num_bits,di): # based on correlation
    s = nn.Softmax(dim=3)
    t = s(encode)
    return (t)

def convert_genex(encode,num_bits,di,arr):
    s = nn.Softmax(dim=3)
    t = s(encode)
    t = t* arr
    ts = torch.sum(t,dim=3)
    return (ts)

def plot_print(code,pdfname):
    fig = plt.figure(figsize=(20,14))
    font = {        'size'   : 12}
    plt.rc('font', **font)
    ax = fig.add_subplot(1,1,1)
    ax = sns.heatmap(code, linewidth=0.5,cmap="coolwarm")
    plt.savefig(pdfname)
    plt.clf()
    plt.close()

def pkl_dump(code,pklname):
    f=open(pklname,"wb")
    pickle.dump(code,f)
    f.close()

def plot_dump(correlation,name):
    print(name)
    plot_print(correlation,name+".pdf")
    pkl_dump(correlation,name+".pkl")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
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
        self.avg = self.sum / self.count


def train(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict, loss_fuction,normweight,constweight,distweight,regweight,distscale,final_output_dir,disttype,drop):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #print(config)
    model.train()
    nme_count = 0
    nme_batch_sum = 0
    critCE=torch.nn.CrossEntropyLoss(reduction="sum")
    critMSE=torch.nn.MSELoss(reduction="mean")
    critL1=torch.nn.L1Loss(reduction="mean")
    critMSEsum=torch.nn.MSELoss(reduction="sum")
    critL1sum=torch.nn.L1Loss(reduction="sum")
    end = time.time()
    arr=torch.tensor(range(0,256,1)).cuda((config.GPUS)[0])
    arrs = torch.tensor(range(1,66,1)).cuda((config.GPUS)[0])
    arrs[64]=0
    finmult = torch.zeros((65)).cuda()
    finmult[64]=2
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    train_len= len(train_loader)



    numbers=torch.zeros(size=(256, 1), dtype=torch.float).cuda()
    weight_c = torch.tensor([[[[-1.0,1.0]]]])

    targets_array = torch.tensor(range(0,config.BITS,1)).cuda()
    for i, (inp, target, meta) in (enumerate(train_loader)):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output_ori,output = model(inp.cuda((config.GPUS)[0]))
        #print(output_ori.size(),output.size())
        target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
        t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
        targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()


        if loss_fuction=="bce":
            loss = normweight* criterion(output_ori, target)
        elif loss_fuction=="ce":
            t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
            loss= normweight*critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
        elif loss_fuction=="L2smooth_ce":
            t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
            targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
            targets = torch.abs(targets-targets_array)
            #print(targets[0,0,0,:])
            targets = convert_correlation(-1*targets*targets,int(config.CODE.CODE_BITS),di)
            #print(t.size(),targets.size())
            #print(targets[0,0,0,:])
            loss= normweight*critCE(t.permute(0,3,1,2),targets.permute(0,3,1,2))
        elif loss_fuction=="chernoff":
            outputexp = torch.exp(0.1*output)
            r = torch.reshape(outputexp,(output.size(0),output.size(1),output.size(2),output.size(3),1))/torch.reshape(outputexp,(output.size(0),output.size(1),output.size(2),1,output.size(3)))
            #print(output[0,0,0,:])
            #print(r[0,0,0,:,:])
            rreduce=torch.prod(r,dim=4)
            #print(rreduce[0,0,0,:])
            #print(rreduce.size())
            targets=torch.reshape(meta['tpts'].cuda(),(r.size(0),r.size(1),r.size(2),1))
            targets = torch.abs(targets-targets_array)
            #print(targets.size())
            loss =  normweight*torch.mean(rreduce*targets)

        elif loss_fuction=="smooth_ce":
            t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
            targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
            targets = torch.abs(targets-targets_array)
            #print(targets[0,0,0,:])
            targets = convert_correlation(-1*targets,int(config.CODE.CODE_BITS),di)
            #print(t.size(),targets.size())
            #print(targets[0,0,0,:])
            loss= normweight*critCE(t.permute(0,3,1,2),targets.permute(0,3,1,2))
        elif loss_fuction=="mod_ce":
            t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
            targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
            #print(targets.size(),targets_dist.size())
            targets = torch.abs(targets-targets_array)
            loss = normweight* (torch.sum(t*targets))            
        elif loss_fuction=="mse":
            t = convert_genex(output,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= normweight*critMSE(t,targets) 
        elif loss_fuction=="L1":
            t = convert_genex(output,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= normweight*critL1(t,targets) 
        elif loss_fuction=="msesum":
            t = convert_genex(output,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= normweight*critMSEsum(t,targets) 
        elif loss_fuction=="L1sum":
            t = convert_genex(output,int(config.BITS/2),di,arr) 
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
            loss= normweight*critL1sum(t,targets) 
        #corr_score
        t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
        targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
        #print(targets.size(),targets_dist.size())
        targets = torch.abs(targets-targets_array)
        corr_score = (torch.sum(t*targets))     
        output_code = torch.reshape(output_ori,(output.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),config.CODE.CODE_BITS))
        if constweight >0:
            #elif disttype=="NL1":
            output_code1=output_code.permute(1,0,2)
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1))
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.cdist(output_code1.contiguous(),output_code1.contiguous(),p=1)
            #print(targets_dist[0])
            target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=2)**2
            #print(target_distance[0])
            #print(distance[0])
            distance=distance-(2*target_distance)
            distance=torch.clamp(distance,min=0)
            distance=torch.sum(distance,dim=2)
            distance=distance.permute(1,0)
            ##loss+= constweight*torch.mean(distance)
            ##print( constweight*torch.mean(distance))
            #print(model.code[0,32,:])
            ## this is conv based constraint
            code_edge = (F.conv2d(torch.reshape(model.code,(int(model.code.size(0)),1,model.code.size(1),model.code.size(2))),weight_c.cuda()))
            #print(code_edge[0,0,32,:])
            #code_edge = (F.conv2d(torch.reshape(model.code,(1,1,model.code.size(0),model.code.size(1))),weight_c.cuda()))
            tycode =  code_edge*code_edge
            #print(tycode[0,0,32,:])
            loss  +=torch.sum(tycode)*constweight
            ## to minimize the number of decision boundaries
            #print(model.code.size())
            ##code_edge=torch.sum(model.code,dim=2)
            #print(code_edge.size())
            ##loss+= constweight * torch.norm(code_edge, 1)
            #print(constweight * torch.norm(code_edge, 1))
        ## distance loss
        
        #output_code = torch.reshape(output_ori,(output.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),config.CODE.CODE_BITS))
        if i==0:
            complete=init_dict(output_code,meta['tpts'])
            #print(sum([complete[0][i].size(0) for i in range(0,len(complete[0]))]))
        else:
            complete=return_dict(complete,output_code,meta['tpts'])
            #print(sum([complete[0][i].size(0) for i in range(0,len(complete[0]))]))

        output_code=output_code.permute(1,0,2)

        ## L2 or L1 distance based loss:
        if disttype=="NL2":
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1))
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.cdist(output_code.contiguous(),output_code.contiguous(),p=2)
            target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=2)
            distance=target_distance-distance
            distance=torch.clamp(distance,min=0)
            distance=torch.sum(distance,dim=2)
            distance=-1*distance.permute(1,0)
        elif disttype=="NL1":
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1))
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.cdist(output_code.contiguous(),output_code.contiguous(),p=1)
            target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=1)
            distance=(2*target_distance)-distance
            distance=torch.clamp(distance,min=0)
            distance=torch.sum(distance,dim=2)
            distance=-1*distance.permute(1,0)
        elif disttype=="L2":
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1))
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.cdist(output_code.contiguous(),output_code.contiguous(),p=2)
            if distscale>0.0:
                target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=2)+0.1
                scale=(torch.ones((targets_dist.size(1),targets_dist.size(1)))-torch.eye(targets_dist.size(1))).cuda()
                distance=distance*scale/target_distance
            distance=torch.sum(distance,dim=2)
            distance=distance.permute(1,0)
        elif disttype=="L1":
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1)).cuda()
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.cdist(output_code.contiguous(),output_code.contiguous(),p=1)

            if distscale>0.0:
                target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=1)+0.1
                scale=(torch.ones((targets_dist.size(1),targets_dist.size(1)))-torch.eye(targets_dist.size(1))).cuda()
                distance=distance*scale/target_distance
            distance=torch.sum(distance,dim=2)
            distance=distance.permute(1,0)
        else:
            targets_dist = torch.clamp(meta['tpts'],min=0,max=255).cuda() 
            targets_dist=torch.reshape(targets_dist,(targets_dist.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),1)).cuda()
            targets_dist=targets_dist.permute(1,0,2)
            distance = torch.matmul(output_code,output_code.permute(0,2,1))
            if distscale>0.0:
                target_distance = torch.cdist(targets_dist.contiguous(),targets_dist.contiguous(),p=2)+0.1
                scale=(torch.ones((targets_dist.size(1),targets_dist.size(1)))-torch.eye(targets_dist.size(1))).cuda()
                distance=distance*scale/target_distance

        loss +=  regweight*torch.norm(output_ori, p='fro') -  distweight* torch.mean(distance)
        #print("distance weight",(-1*torch.mean(distance)).cpu().detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        #print(loss)
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    mean_real_code=0
    mean_bin_code=0
    if epoch==0 or epoch>57:
    #if epoch==0 or (epoch+1)%10==0 or (epoch>55 and epoch <60) or epoch>75:
        mean_real_code,mean_bin_code=dump_dict(complete,final_output_dir,"train"+str(epoch))
    if False:
        nme = nme_batch_sum / nme_count
        msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
            .format(epoch, batch_time.avg, losses.avg, nme)
        logger.info(msg)
    actual_code=torch.zeros((model.code.size(0),model.code.size(1),model.code.size(2)))
    with torch.no_grad():
        actual_code.copy_(model.code)
    return 0,mean_real_code,mean_bin_code,torch.transpose(actual_code,1,2)


def validate(config, val_loader, model, criterion, epoch, writer_dict, loss_fuction,final_output_dir,mean_code):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    corr_score=0
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    critCE=torch.nn.CrossEntropyLoss(reduction="sum")
    critMSE=torch.nn.MSELoss(reduction="mean")
    critL1=torch.nn.L1Loss(reduction="mean")
    critMSEsum=torch.nn.MSELoss(reduction="sum")
    critL1sum=torch.nn.L1Loss(reduction="sum")
    model.eval()

    nme_count = 0
    nme_batch_gen = 0
    nme_batch_genex = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()
    arr=torch.tensor(range(0,256,1)).cuda((config.GPUS)[0])
    di=pickle.load(open(config.CODE.CODE_TENSOR,"rb"))
    di=torch.transpose(di,0,1).cuda()
    print(mean_code.size())
    mean_code=(torch.transpose(mean_code,1,2).cuda())[:,:,0:256]
    print(mean_code.size())
    print(model.code.size())
    #with torch.no_grad():
    #    model.code.copy_(mean_code)
    ##di  = di *2 -1
    correlation=torch.zeros(size=(256,256),dtype=torch.float).cuda()
    numbers=torch.zeros(size=(256,1),dtype=torch.float).cuda()
    targets_array = torch.tensor(range(0,config.BITS,1)).cuda()
    with torch.no_grad():
        for i, (inp, target, meta) in tqdm(enumerate(val_loader)):
            data_time.update(time.time() - end)
            output_ori,output = model(inp.cuda((config.GPUS)[0]))

            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
            targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()

            output_code = torch.reshape(output_ori,(output.size(0),int(output_ori.size(1)/int(config.CODE.CODE_BITS)),config.CODE.CODE_BITS))
            if i==0:
                complete=init_dict(output_code,meta['tpts'])
            else:
                complete=return_dict(complete,output_code,meta['tpts'])
            if loss_fuction=="bce":
                loss = criterion(output_ori, target)
            elif loss_fuction=="ce":
                t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda()
                loss= critCE(t.permute(0,3,1,2),torch.reshape(targets.long(),(t.size(0),t.size(1),t.size(2))).permute(0,1,2))
            elif loss_fuction=="smooth_ce":
                t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
                targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
                targets = torch.abs(targets-targets_array)
                targets = convert_correlation(-1*targets,int(config.CODE.CODE_BITS),di)
                #print(t.size(),targets.size())
                loss= critCE(t.permute(0,3,1,2),targets.permute(0,3,1,2))
            elif loss_fuction=="chernoff":
                outputexp = torch.exp(0.1*output)
                r = torch.reshape(outputexp,(output.size(0),output.size(1),output.size(2),output.size(3),1))/torch.reshape(outputexp,(output.size(0),output.size(1),output.size(2),1,output.size(3)))
                rreduce=torch.prod(r,dim=4)
                #print(rreduce.size())
                targets=torch.reshape(meta['tpts'].cuda(),(r.size(0),r.size(1),r.size(2),1))
                targets = torch.abs(targets-targets_array)
                #print(targets.size())
                loss =  torch.sum(rreduce*targets)
            elif loss_fuction=="L2smooth_ce":
                t = (convert_gen_loss(output,int(config.CODE.CODE_BITS),di))
                targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
                targets = torch.abs(targets-targets_array)
                targets = convert_correlation(-1*targets*targets,int(config.CODE.CODE_BITS),di)
                #print(t.size(),targets.size())
                loss= critCE(t.permute(0,3,1,2),targets.permute(0,3,1,2))
            elif loss_fuction=="mod_ce":
                t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
                targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
                #print(targets.size(),targets_dist.size())
                targets = torch.abs(targets-targets_array)
                loss =  (torch.sum(t*targets))   
            elif loss_fuction=="mse":
                t = convert_genex(output,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critMSE(t,targets) 
            elif loss_fuction=="L1":
                t = convert_genex(output,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critL1(t,targets) 
            elif loss_fuction=="msesum":
                t = convert_genex(output,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critMSEsum(t,targets) 
            elif loss_fuction=="L1sum":
                t = convert_genex(output,int(config.BITS/2),di,arr) 
                targets = torch.clamp(torch.round(meta['tpts']),min=0,max=255).cuda() 
                loss= critL1sum(t,targets)
            #corr_score  calculation 
            t = (convert_correlation(output,int(config.CODE.CODE_BITS),di))
            targets=torch.reshape(meta['tpts'].cuda(),(t.size(0),t.size(1),t.size(2),1))
            #print(targets.size(),targets_dist.size())
            targets = torch.abs(targets-targets_array)
            corr_score = (torch.sum(t*targets))  
            #output = torch.reshape(output,(output.size(0),int(output.size(1)/int(config.CODE.CODE_BITS*2)),2,config.CODE.CODE_BITS))
            _,_,preds = convert_gen(output,int(config.BITS/2),di)

            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp_gen = compute_nme(preds.detach(), meta)


            preds = convert_genex(output,int(config.BITS/2),di,arr)
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            nme_temp_genex = compute_nme(preds.detach(), meta)

            failure_008 = (nme_temp_genex > 0.08).sum()
            failure_010 = (nme_temp_genex > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_gen += np.sum(nme_temp_gen)
            nme_batch_genex += np.sum(nme_temp_genex)
            nme_count = nme_count + preds.size(0)
            for n in range(output.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    #dump_dict(complete,final_output_dir,"valid"+str(epoch))
    nme_gen = nme_batch_gen / nme_count
    nme_genex = nme_batch_genex / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count


    msg_new1 = 'Test_genex Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme_genex,
                                failure_008_rate, failure_010_rate)
    logger.info(msg_new1)

    msg_new_tanh1 = 'Test_gen Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme_gen,
                                failure_008_rate, failure_010_rate)
    logger.info(msg_new_tanh1)
    msg = 'Epoch: [0]\t corr_score {corr_score:.5f}\t'.format(epoch, corr_score=corr_score.cpu().detach().numpy())
    logger.info(msg)
    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme_genex, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme_genex, predictions, correlation/numbers


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp.cuda((config.GPUS)[0]))
            ##target = target.cuda(non_blocking=True)
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            loss = criterion(output, target)

            #t=torch.rand((loss.size(0),loss.size(1)),device=(config.GPUS)[0])
            #loss = (loss*t).sum()
            # NME
            ##score_map = output.data.cpu()
            output= output.data.cpu()
            output = torch.reshape(output,(output.size(0),int(output.size(1)/130),2,65))
            preds = convert(output,65,arr,arrs,finmult)
            ##preds=preds.data.cpu()
            ##preds=decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            preds=decode_preds(preds, meta['center'], meta['scale'], [config.BITS, config.BITS])
            ##score_map = output.data.cpu()
            ##preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            print(preds)
            #print(target)
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions

from pathlib import Path
import os
import pickle
def log_ekn(config, val_loader, model,epoch,cfg_name,title,suf):
    root_output_dir = Path(config.OUTPUT_DIR)
    dataset = config.DATASET.DATASET
    cfg_name = (os.path.basename(cfg_name).split('.')[0])+str(config.TRAIN.LR)+str(config.SUF)+str(suf)

    final_output_dir = root_output_dir / dataset / cfg_name/"accuracy/"
    if not final_output_dir.exists():
        final_output_dir.mkdir() 
    final_output_dir=str(final_output_dir)
    f=open(final_output_dir+"/"+title+"_accuracy_log.csv", "a")
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    acc =torch.zeros((num_classes*2,config.BITS,int(config.CODE.CODE_BITS)))
    count =torch.zeros((num_classes*2,config.BITS,int(config.CODE.CODE_BITS)))

    with torch.no_grad():
        for i, (inp, target, meta) in tqdm(enumerate(val_loader)):
            output = model(inp.cuda((config.GPUS)[0]))
            ##target = target.cuda(non_blocking=True)
            target=torch.flatten((meta['codetpts'].cuda((config.GPUS)[0])), start_dim=1)
            output = torch.reshape(output,(output.size(0),int(output.size(1)/(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
            if True:
                 encode= (output.sign()+1)/2
                 correcttarget = torch.reshape(target,(target.size(0),int(target.size(1)/(2*config.CODE.CODE_BITS)),2,int(config.CODE.CODE_BITS)))
                 diff  = (encode==correcttarget)
                 for v in range(0,diff.size(0)):
                     for label in range(0,num_classes*2):
                         l1=(label%num_classes)
                         l2=int(label/num_classes)
                         value=int((meta['tpts'])[v][l1][l2])
                         value=min(255,value)
                         value=max(0,value)
                         temp=diff[v:v+1,l1:l1+1,l2:l2+1,:]
                         temp=torch.flatten(temp)
                         acc[label][value]+=temp.cpu()
                         count[label][value]+=torch.ones((int(config.CODE.CODE_BITS)))
    
    if True:
        with open(final_output_dir+"/"+str(epoch)+"_"+title+".pkl", "wb") as fout:
            pickle.dump((acc,count), fout)
        print("Overall accuracy(%)")
        print(100* torch.sum(torch.flatten(acc))/torch.sum(torch.flatten(count)))
        print("label wise accuracy (%)")
        f.write("%s,%s,%s\n"%(str(epoch),str(100* torch.sum(torch.flatten(acc))/torch.sum(torch.flatten(count))),str(100* torch.sum(torch.sum(acc,dim=1),dim=1)/torch.sum(torch.sum(count,dim=1),dim=1))))
        print(100* torch.sum(torch.sum(acc,dim=1),dim=1)/torch.sum(torch.sum(count,dim=1),dim=1))
    plot_ekl(acc,count,int(config.CODE.CODE_BITS),epoch,title,final_output_dir)
    f.close()
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
def plot_ekl(error,count, bits,epoch,title,final_output_dir):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    flatten_label=torch.sum(error,dim=0)
    flatten_count=torch.sum(count,dim=0)
    labels=list(range(0,256))
    prob=flatten_label/flatten_count

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 3}

    plt.rc('font', **font)

    for i in range(1,bits+1):
        ax = fig.add_subplot(1, int(bits), i)
        ax.plot(labels, 1-prob[:,i-1:i], marker='o',linewidth=0, color="crimson", markersize=0.15, label="Empirical"+str(i))
        ax.axvline(x=(bits-i+1),c="green",linewidth=0.5,linestyle="--")
        ax.axvline(x=(bits*2-i+1),c="green",linewidth=0.5,linestyle="--")
        plt.text(100, 0.4 , "C"+str(i), rotation=0, verticalalignment='center',color="blue",fontsize="3")
        #if int(((i-1))+1)%16==1:
        ax.set_ylabel('Error probability',fontsize="3")
        if i==bits:
            ax.set_xlabel('Label',fontsize="3")
    plt.savefig(final_output_dir+"/"+str(epoch)+"_"+title+"_ekn_plots.pdf")


