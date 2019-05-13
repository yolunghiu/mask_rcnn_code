import torch


def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    Smooth l1 loss:
        L1 = 0.5 * x**2, |x| < 1
             |x| - 0.5 , others

    这里加了个参数beta
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
