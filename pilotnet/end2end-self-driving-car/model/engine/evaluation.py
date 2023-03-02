import numpy as np
from tqdm import tqdm
from util.logger import setup_logger

from datetime import datetime

import torch.nn as nn

import torch
import numpy as np
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

def do_evaluation(
        cfg,
        model,
        dataloader,
        device,
        criterion,
        transform,
        name,
        verbose=False
):
    model.eval()
    logger = setup_logger("DRIVINGDATASET", cfg.OUTPUT.DIR,
                          name + '-eval-{0:%Y-%m-%d %H:%M:%S}_log'.format(datetime.now()))
    logger.info("Start evaluating")

    mae = nn.L1Loss()

    loss_records = []
    cor_loss_records = []
    soft_loss_records = []
    for iteration, (images, steering_commands, steering_levels) in tqdm(enumerate(dataloader)):
        images = images.to(device)

        correlation,predictions = model(images)

        # predictions = predictions.clone().detach().cpu().numpy()
        real_predictions = transform(predictions)
        mae_loss = mae(steering_commands, real_predictions.cpu())
        loss_records.append(mae_loss.item())
        cor_predictions = convert_cor(correlation,700)
        soft_predictions = convert_soft(correlation,700)
        cor_mae_loss = mae(steering_commands, cor_predictions.cpu())
        soft_mae_loss = mae(steering_commands, soft_predictions.cpu())
        cor_loss_records.append(cor_mae_loss.item())
        soft_loss_records.append(soft_mae_loss.item())
        #logger.info("EVAL_MAE_LOSS: \t{}".format(mae_loss))
        #logger.info("EVAL_COR_MAE_LOSS: \t{}".format(cor_mae_loss))
        #logger.info("EVAL_SOFT_MAE_LOSS: \t{}".format(soft_mae_loss))
        # if verbose:
        #     logger.info("LOSS: \t{}".format(loss))
        #     logger.info("MAE_LOSS: \t{}".format(mae_loss))


    logger.info('LOSS EVALUATION: {}'.format(np.mean(loss_records)))
    logger.info('COR_LOSS EVALUATION: {}'.format(np.mean(cor_loss_records)))
    logger.info('SOFT_LOSS EVALUATION: {}'.format(np.mean(soft_loss_records)))

