
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim


def create_logger(cfg, cfg_name, phase='train',suf=""):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = (os.path.basename(cfg_name).split('.')[0])+str(cfg.TRAIN.LR)+str(cfg.SUF)+str(suf)

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        #optimizer = optim.Adam(
        #    filter(lambda p: p.requires_grad, model.parameters()),
        #    lr=cfg.TRAIN.LR
        #)
        print(filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))
        optimizer = optim.Adam([
            {'params':[p[1] for p in (filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))],'lr':cfg.TRAIN.LR}
            ,
            {'params':[model.code],'lr':cfg.TRAIN.LR*10},
        ])
        #print([p[1] for p in (filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))])
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

def get_optimizer_n(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        #optimizer = optim.Adam(
        #    filter(lambda p: p.requires_grad, model.parameters()),
        #    lr=cfg.TRAIN.LR
        #)
        print(filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))
        optimizer = optim.Adam([
            {'params':[p[1] for p in (filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))],'lr':cfg.TRAIN.LR}
            ,
            {'params':[model.code],'lr':cfg.TRAIN.LR*20},
        ])
        #print([p[1] for p in (filter(lambda p: p[1].requires_grad and p[0]!="code", model.named_parameters()))])
    elif cfg.TRAIN.OPTIMIZER == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            alpha=cfg.TRAIN.RMSPROP_ALPHA,
            centered=cfg.TRAIN.RMSPROP_CENTERED
        )

    return optimizer


def save_checkpoint(states, predictions, is_best,
                    output_dir, filename='checkpoint.pth'):
    preds = predictions.cpu().data.numpy()
    torch.save(states, os.path.join(output_dir, filename))
    torch.save(preds, os.path.join(output_dir, 'current_pred.pth'))

    latest_path = os.path.join(output_dir, 'latest.pth')
    if os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(os.path.join(output_dir, filename), latest_path)
    if is_best and 'state_dict' in states.keys():
        torch.save(states['state_dict'],os.path.join(output_dir, 'model_best.pth'))
        #torch.save(states['state_dict'].module, os.path.join(output_dir, 'model_best.pth'))
