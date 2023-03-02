import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import datasets, utils
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.pylab as plt
mult=1

def convert_new_tanh(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=1)
    return ts
def convert_new_tanh_loss(encode,num_bits,di): # based on correlation
    encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    return (t)

def convert_cor(encode,num_bits,di): # based on correlation
    _,ts = torch.max(encode,dim=1)
    return ts
def convert_cor_code(encode,num_bits,di): # based on correlation
    t = torch.matmul(encode,di)
    _,ts = torch.max(t,dim=1)
    return ts
def convert_cor_loss(encode,num_bits,di): # based on correlation
    return (encode)

def convert_cor_loss_soft(encode,num_bits,di): # based on correlation
    s = nn.Softmax(dim=1)
    t = s(encode)
    return (t)

def convert_soft(encode,num_bits,di):
    arr = torch.tensor(range(0,num_bits,1)).cuda()
    #encode = torch.tanh(encode)
    s = nn.Softmax(dim=1)
    t = s(encode)
    t = t* arr
    ts = torch.sum(t,dim=1)
    return (ts)
def convert_soft_code(encode,num_bits,di):
    arr = torch.tensor(range(0,num_bits,1)).cuda()
    #encode = torch.tanh(encode)
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=1)
    t = s(t)
    t = t* arr
    ts = torch.sum(t,dim=1)
    return (ts)

def measure_MAE(cont_labels,yaw,pitch,roll,gpu,func,yaw_code,pitch_code,roll_code):
    #cont_labels=cont_labels.cuda(gpu)
    #print(yaw)
    
    #yaw = (yaw.sign()+1)/2
    #pitch = (pitch.sign()+1)/2
    #roll = (roll.sign()+1)/2

    yaw_predicted=func(yaw,150,yaw_code).cpu()
    pitch_predicted= func(pitch,150,pitch_code).cpu()
    roll_predicted=func(roll,100,roll_code).cpu()

    label_yaw = cont_labels[:,0].float()
    label_pitch = cont_labels[:,1].float()
    label_roll = cont_labels[:,2].float()
    yaw_error = torch.sum(torch.abs(yaw_predicted - label_yaw))
    pitch_error = torch.sum(torch.abs(pitch_predicted - label_pitch))
    roll_error = torch.sum(torch.abs(roll_predicted - label_roll))
    return yaw_error, pitch_error,roll_error

def measure(model,i,images,labels, cont_labels, name, yaw_error, pitch_error,roll_error,gpu,total):
    images = Variable(images).cuda(gpu)
    total += cont_labels.size(0)
    yaw,pitch,roll = (model(images))
    yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu)

    # Mean absolute error
    yaw_error += yerror
    pitch_error += perror
    roll_error += rerror
    return yaw_error, pitch_error,roll_error, total


def get_ignored_params(model,arch):
    # Generator function that yields ignored params.
    if arch=="vgg16":
        b=[]
    else:
        b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model,arch):
    # Generator function that yields params that will be optimized.
    if arch=="vgg16":
        b = [model.features]
    else:
        b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model,arch):
    # Generator function that yields fc layer params.
    if arch=="vgg16":
        b = [model.classifier, model.fc_angles]
    elif arch=="resnet_stage":
        b = [model.fc1y,model.fc1p,model.fc1r,
            model.fc2y,model.fc2p,model.fc2r,
            model.fc3y,model.fc3p,model.fc3r,
            model.fc4y,model.fc4p,model.fc4r,
            model.fc5y,model.fc5p,model.fc5r,
            model.fc6y,model.fc6p,model.fc6r,
            model.fc7y,model.fc7p,model.fc7r,
            model.fc8y,model.fc8p,model.fc8r,]
    else:
        b = [model.fc_angles_yaw,model.fc_angles_pitch,model.fc_angles_roll]#,model.yawm,model.pitchm,model.rollm]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def print_statement(yaw,pitch,roll,total,f,label):
    yaw=yaw.detach().numpy()
    pitch=pitch.detach().numpy()
    roll=roll.detach().numpy()
    print(label+ ' error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw / total,
        pitch / total, roll / total))
    print("%s MAE= %.4f"% (label,((yaw / total)+(pitch / total)+(roll / total))/3))
    #f.write(label+ ' error in degrees of the model on the ' + str(total) +
    #    ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f \n' % (yaw / total,
    #    pitch / total, roll / total))
    #f.write("%s MAE= %.4f \n"% (label,((yaw / total)+(pitch / total)+(roll / total))/3))


def train(model,train_loader,test_loader, pose_dataset, output_string, gpu, arch,lr,num_epochs,batch_size,val_bound,num_bits,code_bits,code,loss_func,di,dis,args,reqgrad):
    if reqgrad>0.01:
        mu=10.0
    else:
        mu=0.0
    print(mu,"mu",reqgrad)
    if not os.path.isdir(os.environ['TMPDIR']+"/output/snapshots/" + output_string):
        os.mkdir(os.environ['TMPDIR']+"/output/snapshots/" + output_string)
    f=open(os.environ['TMPDIR']+"/output/snapshots/" + output_string+"/"+output_string,"a")
    model.cuda(gpu)
    criterion = nn.BCEWithLogitsLoss(reduction="sum").cuda(gpu)
    optimizer = torch.optim.Adam([{'params': get_ignored_params(model,arch), 'lr': 0},
                                  {'params': get_non_ignored_params(model,arch), 'lr': lr},
                                  {'params': get_fc_params(model,arch), 'lr': lr * 5},
                                  {'params':[model.yawm,model.pitchm,model.rollm],'lr':lr*mu}],
                                   lr = lr)
    critCE=torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    critMSE=torch.nn.MSELoss(reduction="mean").cuda()
    critL1=torch.nn.L1Loss(reduction="mean").cuda()
    print('Ready to train network.')
    f.write('Ready to train network.')
    #di=pickle.load(open("../bel"+code+"_150_tensor.pkl","rb"))
    #dis=pickle.load(open("../bel"+code+"_100_tensor.pkl","rb"))
    di=torch.transpose(di,0,1).cuda()
    dis=torch.transpose(dis,0,1).cuda()
    best_MAE=100
    t150_array = torch.tensor(range(0,150,1)).cuda()
    t100_array = torch.tensor(range(0,100,1)).cuda()
    weight_c = torch.tensor([[[[-1.0,1.0]]]])
    epoch_list=[1,10,25,35,num_epochs-2,num_epochs-1]
    for epoch in range(num_epochs):
        model.train()
        if epoch==30:
            lr =lr/10
            optimizer = torch.optim.Adam([{'params': get_ignored_params(model,arch), 'lr': 0},
                                  {'params': get_non_ignored_params(model,arch), 'lr': lr},
                                  {'params': get_fc_params(model,arch), 'lr': lr * 5},
                                  {'params':[model.yawm,model.pitchm,model.rollm],'lr':lr*mu}],
                                   lr = lr)
        val_loss, train_loss, test_loss=0.0, 0.0, 0.0
        ttotal,tyaw_error ,tpitch_error ,troll_error = 0,.0,.0,.0
        tstotal,tsyaw_error ,tspitch_error ,tsroll_error = 0,.0,.0,.0
        ts1total,ts1yaw_error ,ts1pitch_error ,ts1roll_error = 0,.0,.0,.0
        ts2total,ts2yaw_error ,ts2pitch_error ,ts2roll_error = 0,.0,.0,.0
        ts3total,ts3yaw_error ,ts3pitch_error ,ts3roll_error = 0,.0,.0,.0
        ts4total,ts4yaw_error ,ts4pitch_error ,ts4roll_error = 0,.0,.0,.0
        ts5total,ts5yaw_error ,ts5pitch_error ,ts5roll_error = 0,.0,.0,.0


        for i, (images, labels, cont_labels, name, tyaw,tpitch,troll,tiyaw,tipitch,tiroll) in tqdm((enumerate(train_loader))):
            tyaw= tyaw.cuda(gpu)
            tpitch= tpitch.cuda(gpu)
            troll= troll.cuda(gpu)
            images = Variable(images).cuda(gpu)
            yaw,pitch,roll,rangles,yawcode,pitchcode,rollcode = (model(images))
            angles = torch.cat((yawcode,pitchcode,rollcode),dim=1)
            bout = torch.cat((tyaw,tpitch,troll),dim=1)
            ##
            ty = (F.conv2d(torch.reshape(model.yawm,(1,1,model.yawm.size(0),model.yawm.size(1))),weight_c.cuda(),padding=(0,1)))
            tycode =  ty*ty
            tp = (F.conv2d(torch.reshape(model.pitchm,(1,1,model.pitchm.size(0),model.pitchm.size(1))),weight_c.cuda(),padding=(0,1)))
            tpcode = tp*tp
            tr = (F.conv2d(torch.reshape(model.rollm,(1,1,model.rollm.size(0),model.rollm.size(1))),weight_c.cuda(),padding=(0,1)))
            trcode =  tr*tr
            if loss_func=="bce":
                loss = args.norm_weight*criterion(angles.float(), bout.float())
            elif loss_func=="ce":
                ty = (convert_cor_loss(yaw,150,di))
                tp = (convert_cor_loss(pitch,150,di))
                tr = (convert_cor_loss(roll,100,dis))
                loss= args.norm_weight*( critCE(ty,tiyaw.cuda()) + critCE(tp,tipitch.cuda()) + critCE(tr,tiroll.cuda()))
            elif loss_func=="smooth_ce":
                ty = (convert_cor_loss(yaw,150,di))
                tp = (convert_cor_loss(pitch,150,di))
                tr = (convert_cor_loss(roll,100,dis))
                ytargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(cont_labels[:,0],(cont_labels.size(0),1)).float().cuda()-t150_array),0,0)
                ptargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(cont_labels[:,1],(cont_labels.size(0),1)).float().cuda()-t150_array),0,0)
                rtargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(cont_labels[:,2],(cont_labels.size(0),1)).float().cuda()-t100_array),0,0)
                loss= args.norm_weight*( critCE(ty,ytargets) + critCE(tp,ptargets) + critCE(tr,rtargets))
                loss += args.const_weight*(torch.sum(tycode) + torch.sum(trcode)+torch.sum(tpcode))

            elif loss_func=="mse":
                ty = (convert_soft(yaw,150,di))
                tp = (convert_soft(pitch,150,di))
                tr = (convert_soft(roll,100,dis))
                loss= 1.0*( critMSE(ty, cont_labels[:,0].float().cuda()) + critMSE(tp, cont_labels[:,1].float().cuda()) + critMSE(tr, cont_labels[:,2].float().cuda()))
            elif loss_func=="L1":
                ty = (convert_soft(yaw,150,di))
                tp = (convert_soft(pitch,150,di))
                tr = (convert_soft(roll,100,dis))
                loss= 1.0*( critL1(ty, cont_labels[:,0].float().cuda()) + critMSE(tp, cont_labels[:,1].float().cuda()) + critMSE(tr, cont_labels[:,2].float().cuda()))

            temp0 = torch.tensor(range(0,yawcode.size(0),1)).cuda()
            temp0=torch.reshape(temp0,(temp0.size(0),1)).float()
            temp2 = torch.tensor(range(0,rollcode.size(0),1)).cuda()
            temp2=torch.reshape(temp2,(temp2.size(0),1)).float()
            temp1 = torch.tensor(range(0,pitchcode.size(0),1)).cuda()
            temp1=torch.reshape(temp1,(temp1.size(0),1)).float()

            distance = torch.cdist(yawcode.contiguous(),yawcode.contiguous(),p=1)
            target_distance = torch.cdist(temp0.contiguous(),temp0.contiguous(),p=1)
            rdistance = torch.cdist(rollcode.contiguous(),rollcode.contiguous(),p=1)
            rtarget_distance = torch.cdist(temp2.contiguous(),temp2.contiguous(),p=1)
            pdistance = torch.cdist(pitchcode.contiguous(),pitchcode.contiguous(),p=1)
            ptarget_distance = torch.cdist(temp1.contiguous(),temp1.contiguous(),p=1)
            ydistance=torch.clamp(((2*target_distance)-distance),min=0)
            rdistance=torch.clamp(((2*rtarget_distance)-rdistance),min=0)
            pdistance=torch.clamp(((2*ptarget_distance)-pdistance),min=0)
            loss+= args.dist_weight*(torch.mean(torch.sum(ydistance,dim=1))+torch.mean(torch.sum(pdistance,dim=1))+torch.mean(torch.sum(rdistance,dim=1)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=(loss.data).cpu().numpy()
   
            ttotal += cont_labels.size(0)
            yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_soft,di,di,dis)
            tyaw_error += yerror; tpitch_error += perror;   troll_error += rerror
        for j, (images, labels, cont_labels, name, tyaw,tpitch,troll,tiyaw,tipitch,tiroll) in tqdm((enumerate(test_loader))):
                model.eval()
                tyaw= tyaw.cuda(gpu)
                tpitch= tpitch.cuda(gpu)
                troll= troll.cuda(gpu)
                with torch.no_grad():
                    images = Variable(images).cuda(gpu)
                    yaw,pitch,roll,rangles,yawcode,pitchcode,rollcode = (model(images))
                    angles = torch.cat((yawcode,pitchcode,rollcode),dim=1)
                    bout = torch.cat((tyaw,tpitch,troll),dim=1)
                    loss = criterion(angles.float(), bout.float())
                    test_loss+=(loss.data).cpu().numpy()
                    tstotal += cont_labels.size(0)
                    yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_soft,0,0,0)
                    ts4yaw_error += yerror;       ts4pitch_error += perror;        ts4roll_error += rerror
                    yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_cor,0,0,0)
                    ts5yaw_error += yerror;       ts5pitch_error += perror;        ts5roll_error += rerror
                    yaw = (convert_cor_loss_soft(yaw,150,di))
                    pitch = (convert_cor_loss_soft(pitch,150,di))
                    roll = (convert_cor_loss_soft(roll,100,dis))
        cur_MAE= ((ts4yaw_error+ts4pitch_error+ts4roll_error)/tstotal)
        if cur_MAE<=best_MAE:
                 print("best:")
                 best_MAE=cur_MAE
                 torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/" + output_string+"_best_model.pth")
        torch.save(model.state_dict(),os.environ['TMPDIR']+"/output/snapshots/" + output_string+str(epoch%2)+"_model.pth")
        print("epoch %d"%(epoch))
        print_statement(tyaw_error,tpitch_error,troll_error,ttotal,f,"train ")
        print_statement(ts4yaw_error,ts4pitch_error,ts4roll_error,tstotal,f,"test_genex ")
        print_statement(ts5yaw_error,ts5pitch_error,ts5roll_error,tstotal,f,"test_gen ")


    f.close()
def test(model,test_loader, pose_dataset, output_string, gpu, arch,lr,num_epochs,batch_size,val_bound,num_bits,code_bits,code,loss_func,di,dis):
    f=open(os.environ['TMPDIR']+"/"+output_string+"_test","w")
    model.cuda(gpu)
    di=pickle.load(open("code_pkl/bel"+code+"_150_tensor.pkl","rb"))
    dis=pickle.load(open("code_pkl/bel"+code+"_100_tensor.pkl","rb"))
    di=torch.transpose(di,0,1).cuda()
    dis=torch.transpose(dis,0,1).cuda()

    ttotal,tyaw_error ,tpitch_error ,troll_error = 0,.0,.0,.0
    tstotal,tsyaw_error ,tspitch_error ,tsroll_error = 0,.0,.0,.0
    ts1total,ts1yaw_error ,ts1pitch_error ,ts1roll_error = 0,.0,.0,.0
    ts2total,ts2yaw_error ,ts2pitch_error ,ts2roll_error = 0,.0,.0,.0
    ts3total,ts3yaw_error ,ts3pitch_error ,ts3roll_error = 0,.0,.0,.0
    for j, (images, labels, cont_labels, name, tyaw,tpitch,troll,_,_,_) in tqdm((enumerate(test_loader))):
            model.eval()
            tyaw= tyaw.cuda(gpu)
            tpitch= tpitch.cuda(gpu)
            troll= troll.cuda(gpu)
            with torch.no_grad():
                images = Variable(images).cuda(gpu)
                yaw,pitch,roll,rangles,yawcode,pitchcode,rollcode = (model(images))
                angles = torch.cat((yawcode,pitchcode,rollcode),dim=1)
                bout = torch.cat((tyaw,tpitch,troll),dim=1)
                tstotal += cont_labels.size(0)
                yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_soft,0,0,0)
                ts2yaw_error += yerror;       ts2pitch_error += perror;        ts2roll_error += rerror
                yerror, perror,rerror = measure_MAE(cont_labels,yaw,pitch,roll,gpu,convert_cor,0,0,0)
                ts1yaw_error += yerror;       ts1pitch_error += perror;        ts1roll_error += rerror


    print_statement(ts1yaw_error,ts1pitch_error,ts1roll_error,tstotal,f,"test_gen ")
    print_statement(ts2yaw_error,ts2pitch_error,ts2roll_error,tstotal,f,"test_genex ")


    f.close()
def finetune(model,train_loader,test_loader,pose_dataset, output_string,gpu,arch,batch_size,val_bound,num_bits):
    lr=0.0001
    model.cuda(gpu)

    num_epochs=20
    train(model,train_loader,test_loader,pose_dataset, output_string,gpu,arch,lr,num_epochs,batch_size,val_bound,num_bits,1.0,0.0)
