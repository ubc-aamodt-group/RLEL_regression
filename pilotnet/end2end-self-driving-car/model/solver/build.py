import torch


def make_optimizer(cfg, model,reqgrad=0.0):
    params = []
    lr = cfg.SOLVER.BASE_LR
    for key, value in model.named_parameters():
        if "code" in key:
            continue
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    print("Rqgrad",reqgrad)
    if reqgrad>0:
        print("params")
        params += [{"params": [model.code], "lr": lr*10}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    
    # optimizer = torch.optim.Adam(params, lr)

    return optimizer
