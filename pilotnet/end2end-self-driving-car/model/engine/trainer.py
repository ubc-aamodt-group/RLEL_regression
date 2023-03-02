import os
import torch
from model.engine.evaluation import do_evaluation
from datetime import datetime
from util.logger import setup_logger
from util.visdom_plots import VisdomLogger

import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from model import conversion_helper as conversion
from tqdm import tqdm
import pickle
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

def convert_cor(encode,valrange):
    return (torch.argmax(encode,dim=1))

mae = nn.L1Loss()

def do_train(
        cfg,
        args,
        model,
        dataloader_train,
        dataloader_evaluation,
        optimizer,
        device,
        criterion,
        transform,
        name
):
    # set mode to training for model (matters for Dropout, BatchNorm, etc.)
    model.train()

    # get the trainer logger and visdom
    visdom = VisdomLogger(cfg.LOG.PLOT.DISPLAY_PORT)
    visdom.register_keys(['loss'])
    logger = setup_logger('balad-mobile.train', False)
    logger.info("Start training")

    output_dir = os.path.join(cfg.LOG.PATH, name + '-train-run_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    os.makedirs(output_dir)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)
    t_array = torch.tensor(range(0, 700, 1)).cuda()
    weight_c = torch.tensor([[[[-1.0,1.0]]]]).cuda()
    # start the training loop
    foldername=("newdumpoutput"+str(args.version)+str(args.init)+"_reqgrad_"+str(args.reqgrad)+"_dsit_"+str(args.dist_weight)+"_const_"+str(args.const_weight)+"/").replace(".","o")
    if not os.path.isdir(foldername):
        os.mkdir(foldername)
    for epoch in range(cfg.SOLVER.EPOCHS):
        print("EPOCH: ", epoch)
        model.train()
        initcode=0
        for iteration, (images, steering_commands, steering_levels) in tqdm(enumerate(dataloader_train)):
            images = images.to(device)

            correlation,predictions = model(images)
            if initcode==0:
                outcodel=predictions
                targetlist=steering_commands
                initcode=1
            else:
                outcodel=torch.cat((outcodel,predictions),dim=0)
                targetlist=torch.cat((targetlist,steering_commands),dim=0)
            steering_levels = torch.stack(steering_levels).cuda()

            if args.loss == "mae" or args.loss == "mse":
                outp = transform.convert_continuous(predictions)
                loss = criterion(outp, steering_commands.clone().detach().float().cuda())
            elif args.transform == "mc":
                steering_int_commands = torch.tensor(steering_commands, dtype=int).cuda()
                loss = criterion(predictions, steering_int_commands)
            elif args.loss == "ce":
                outp = transform.convert_discrete(predictions)
                steering_int_commands = torch.tensor(steering_commands, dtype=int).cuda()
                loss = criterion(outp, steering_int_commands)
            elif args.loss == "smooth_ce":
                ytargets = convert_cor_loss_soft(-1*torch.abs(torch.reshape(steering_commands,(steering_commands.size(0),1)).float().cuda()-t_array))
                loss = criterion(correlation, ytargets)
            elif args.loss == "bce":
                steering_levels = torch.clamp(steering_levels, 0)
                loss = criterion(predictions, steering_levels)
            
            if True:
                temp0=torch.reshape(steering_commands,(steering_commands.size(0),1)).float().cuda()
                distance = torch.cdist(predictions.contiguous(),predictions.contiguous(),p=1)
                target_distance = torch.cdist(temp0.contiguous(),temp0.contiguous(),p=1)
                ydistance=torch.clamp(((2*target_distance)-distance),min=0)
                loss+= args.dist_weight*(torch.mean(torch.sum(ydistance,dim=1)))
            if True:
                ty = (F.conv2d(torch.reshape(model.code.cuda(),(1,1,model.code.size(0),model.code.size(1))),weight_c.cuda(),padding=(0,1)))
                tycode =  ty*ty
                loss+= args.const_weight*(torch.sum(tycode)) 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # predictions = predictions.clone().detach().cpu().numpy()
            

            if iteration % cfg.LOG.PERIOD == 0:
                visdom.update({'loss': [loss.item()]})
                logger.info("LOSS: \t{}".format(loss))
                real_predictions = transform(predictions)
                cor_predictions = convert_cor(correlation,700)
                soft_predictions = convert_soft(correlation,700)
                mae_loss = mae(steering_commands, real_predictions.cpu())
                cor_mae_loss = mae(steering_commands, cor_predictions.cpu())
                soft_mae_loss = mae(steering_commands, soft_predictions.cpu())
                logger.info("MAE_LOSS: \t{}".format(mae_loss))
                logger.info("COR_MAE_LOSS: \t{}".format(cor_mae_loss))
                logger.info("SOFT_MAE_LOSS: \t{}".format(soft_mae_loss))

            # if iteration % cfg.LOG.PLOT.ITER_PERIOD == 0:
            #     visdom.do_plotting()

            step = epoch * len(dataloader_train) + iteration
            # if step % cfg.LOG.WEIGHTS_SAVE_PERIOD == 0 and iteration:
            #     torch.save(model.state_dict(),
            #                os.path.join(output_dir, 'weights_{}.pth'.format(str(step))))
            #     do_evaluation(cfg, model, dataloader_evaluation, device, criterion, converter)
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(output_dir, 'weights_final.pth'))
