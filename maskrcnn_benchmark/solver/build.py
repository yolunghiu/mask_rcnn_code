import torch

from .lr_scheduler import WarmupMultiStepLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            # pre-trained model参数不更新
            continue

        lr = cfg.SOLVER.BASE_LR  # 0.0025
        weight_decay = cfg.SOLVER.WEIGHT_DECAY  # 0.0001
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR  # *2
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS  # 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # Optimizer文档: https://pytorch.org/docs/stable/optim.html?highlight=torch%20optim%20sgd
    # 对于指定优化参数的变量使用指定的参数进行优化, 未指定的使用函数中传入的参数进行更新
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,  # (480000, 640000)
        cfg.SOLVER.GAMMA,  # 0.1
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,  # 1.0 / 3
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,  # 500
        warmup_method=cfg.SOLVER.WARMUP_METHOD,  # "linear"
    )
