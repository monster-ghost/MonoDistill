import os
import torch
import torch.nn as nn
from collections import OrderedDict
import matplotlib.pyplot as plt


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            pretrained_dict = checkpoint['model_state']
            new_state_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k.replace("module.","")  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

            model.load_state_dict(new_state_dict)
        # if optimizer is not None and checkpoint['optimizer_state'] is not None:
        #     optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info("==> Done")
    else:
        raise FileNotFoundError

    return epoch


def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input")

def save_scalars(logger, stats, global_step, interval):
    for key in stats:
        value = float(tensor2float(stats[key])/interval)
        logger.add_scalar(key, value, global_step)

def visualize_backbone_feature_map(rgb_pyramid, depth_pyramid, save_path, global_step):
    # torch.Size([4, 16, 384, 1280])
    # torch.Size([4, 32, 192, 640])
    # torch.Size([4, 64, 96, 320])
    # torch.Size([4, 128, 48, 160])
    # torch.Size([4, 256, 24, 80])
    # torch.Size([4, 512, 12, 40])
    pyramid = ['1_1', '1_2', '1_4', '1_8', '1_16', '1_32']
    for i in range(len(rgb_pyramid)):
        rgb_feature_map = rgb_pyramid[i].cpu()
        rgb_feature_map = rgb_feature_map.data.numpy()

        depth_feature_map = depth_pyramid[i].cpu()
        depth_feature_map = depth_feature_map.data.numpy()

        num_pic = rgb_feature_map.shape[1]
        #print("num,pic", num_pic)
        # row, col = get_row_col(num_pic)
        # print(row, col)
        plt.figure()
        save_num_pic = num_pic//4
        for j in range(0, 4):
            rgb_feature_map_split = rgb_feature_map[0, j*save_num_pic, :, :]
            depth_feature_map_split = depth_feature_map[0, j*save_num_pic, :, :]

            plt.subplot(1, 2, 1)
            plt.imshow(rgb_feature_map_split)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_feature_map_split)
            plt.axis('off')  # 不显示坐标轴
            filename = save_path + str(global_step) + '_' + pyramid[i] +'.png'
            plt.savefig(filename)

def visualize_neck_feature_map(rgb, depth, save_path, global_step):
        rgb_feature_map = rgb.cpu()
        rgb_feature_map = rgb_feature_map.data.numpy()

        depth_feature_map = depth.cpu()
        depth_feature_map = depth_feature_map.data.numpy()

        num_pic = rgb_feature_map.shape[1]
                # print("num,pic", num_pic)
                # row, col = get_row_col(num_pic)
                # print(row, col)
        plt.figure()
        save_num_pic = num_pic // 4
        for j in range(0, 4):
            rgb_feature_map_split = rgb_feature_map[0, j * save_num_pic, :, :]
            depth_feature_map_split = depth_feature_map[0, j * save_num_pic, :, :]

            plt.subplot(1, 2, 1)
            plt.imshow(rgb_feature_map_split)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_feature_map_split)
            plt.axis('off')  # 不显示坐标轴
            filename = save_path + str(global_step) + '_' + 'neck' + '.png'
            plt.savefig(filename)
            plt.close()
        #axis('off')
        #filename = save_path + 'fea_map/kitti_corr/4/' + str(i + 1) + '.png'
        #plt.savefig(filename)
    # plt.show()

    # 各个特征图按1：1 叠加
    # feature_map_sum = feature_map_split[0]
    # for i in range(len(feature_map_combination) - 1):
    #     feature_map_sum = feature_map_combination[i + 1] + feature_map_sum
    # print("!!!!", feature_map_sum.shape)
    # plt.imshow(feature_map_sum)
    # plt.savefig('fea_map/kitti_corr/4/' + 'feature_map_sum.png')