from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN  # (800,)
        max_size = cfg.INPUT.MAX_SIZE_TRAIN  # 1333
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST  # (800,)
        max_size = cfg.INPUT.MAX_SIZE_TEST  # 1333
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255  # True
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN,  # [102.9801, 115.9465, 122.7717]
        std=cfg.INPUT.PIXEL_STD,  # [1., 1., 1.]
        to_bgr255=to_bgr255
    )

    transform = T.Compose(
        [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform
