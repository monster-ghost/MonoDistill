import torch
import numpy as np
import logging
import random
import math
from collections import Iterable


def create_logger(log_path, log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_path+log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed ** 2)
    torch.manual_seed(seed ** 3)
    torch.cuda.manual_seed(seed ** 4)
    torch.cuda.manual_seed_all(seed ** 4)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def judge_nan(rgb_loss):
    #### check nan in order to visual loss
    valid_num_rgb = 0
    rgb_loss_vis = 0

    if rgb_loss.shape != torch.Size([]):
        for rgb_loss_item in rgb_loss:
            if math.isnan(rgb_loss_item.item()):
                pass
            else:
                valid_num_rgb = valid_num_rgb + 1
                rgb_loss_vis = rgb_loss_vis + rgb_loss_item.item()

        if valid_num_rgb!=0:
            rgb_loss_vis = rgb_loss_vis/valid_num_rgb
        else:
            rgb_loss_vis = 0.0

    else:
        if math.isnan(rgb_loss.item()):
            rgb_loss_vis = 0.0
        else:
            rgb_loss_vis = rgb_loss.item()

    return rgb_loss_vis