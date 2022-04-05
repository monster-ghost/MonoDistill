import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import shutil

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
# from lib.helpers.save_helper import visualize_feature_map

from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from lib.helpers.utils_helper import judge_nan

import PIL
import matplotlib.pyplot as plt
import math

from progress.bar import Bar
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 model_type,
                 root_path):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_name = self.test_loader.dataset.class_name
        self.model_type = model_type
        self.root_path = root_path

        # loading pretrain/resume model
        if self.model_type == 'centernet3d':
            if cfg.get('pretrain_model'):
                assert os.path.exists(cfg['pretrain_model'])
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=cfg['pretrain_model'],
                                map_location=self.device,
                                logger=self.logger)

            if cfg.get('resume_model', None):
                assert os.path.exists(cfg['resume_model'])
                self.epoch = load_checkpoint(model=self.model,
                                             optimizer=self.optimizer,
                                             filename=cfg['resume_model'],
                                             map_location=self.device,
                                             logger=self.logger)
                self.lr_scheduler.last_epoch = self.epoch - 1

        if self.model_type == 'distill':
            if cfg.get('pretrain_model'):
                if os.path.exists(cfg['pretrain_model']['rgb']):
                    load_checkpoint(model=self.model.centernet_rgb,
                                    optimizer=None,
                                    filename=cfg['pretrain_model']['rgb'],
                                    map_location=self.device,
                                    logger=self.logger)
                else:
                    self.logger.info("no rgb pretrained model")
                    assert os.path.exists(cfg['pretrain_model']['rgb'])

                if os.path.exists(cfg['pretrain_model']['depth']):
                    load_checkpoint(model=self.model.centernet_depth,
                                    optimizer=None,
                                    filename=cfg['pretrain_model']['depth'],
                                    map_location=self.device,
                                    logger=self.logger)
                else:
                    self.logger.info("no depth pretrained model")
                    assert os.path.exists(cfg['pretrain_model']['depth'])

            if cfg.get('resume_model', None):
                assert os.path.exists(cfg['resume_model'])
                self.epoch = load_checkpoint(model=self.model,
                                             optimizer=self.optimizer,
                                             filename=cfg['resume_model'],
                                             map_location=self.device,
                                             logger=self.logger)
                self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        self.model = torch.nn.DataParallel(self.model).cuda()

    def update_lr_scheduler(self, epoch):
        if self.warmup_lr_scheduler is not None and epoch < 5:
            self.warmup_lr_scheduler.step()
        else:
            self.lr_scheduler.step()

    def save_model(self):
        if (self.epoch % self.cfg['save_frequency']) == 0:
            os.makedirs(self.cfg['model_save_path'], exist_ok=True)
            ckpt_name = os.path.join(self.cfg['model_save_path'], 'checkpoint_epoch_%d' % self.epoch)
            save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)
            #self.inference()
            #self.evaluate()

    def train(self):

        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True,
                                 desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.epoch += 1

            if self.model_type == 'centernet3d':
                self.train_one_epoch()

            elif self.model_type == 'distill':
                self.train_one_epoch_distill()
                # update learning rate
            self.update_lr_scheduler(epoch)
            self.save_model()
            progress_bar.update()


    def train_one_epoch(self):
        self.model.train()
        self.stats = {}  # reset stats dict
        self.stats['train'] = {}  # reset stats dict

        loss_stats = ['seg', 'offset2d', 'size2d', 'offset3d', 'depth', 'size3d', 'heading']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        bar = Bar('{}/{}'.format("3D", self.cfg['model_save_path']), max=num_iters)
        end = time.time()

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            # inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            _, outputs = self.model(inputs['rgb'])

            rgb_loss, rgb_stats_batch = compute_centernet3d_loss(outputs, targets)
            # depth_loss, depth_stats_batch = compute_depth_centernet3d_loss(depth_outputs, targets)
            total_loss = rgb_loss
            total_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, num_iters, phase="train",
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    rgb_stats_batch[l], inputs['rgb'].shape[0])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.next()
        bar.finish()

    def train_one_epoch_distill(self):
        self.model.train()
        self.stats = {}  # reset stats dict
        self.stats['train'] = {}  # reset stats dict

        loss_stats = ['rgb_loss', 'backbone_loss_l1', 'backbone_loss_affinity', 'head_loss']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        bar = Bar('{}/{}'.format("3D", self.cfg['model_save_path']), max=num_iters)
        end = time.time()


        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):

            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            # inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            stats_batch = {}
            #stats_batch['rgb_loss'] = 0
            self.optimizer.zero_grad()
            rgb_loss, backbone_loss_l1, backbone_loss_affinity, head_loss = self.model(inputs, targets)

            stats_batch['rgb_loss'] = judge_nan(rgb_loss)

            rgb_loss = rgb_loss.mean()
            backbone_loss_l1 = backbone_loss_l1.mean()
            backbone_loss_affinity = backbone_loss_affinity.mean()
            head_loss = head_loss.mean()

            stats_batch['backbone_loss_l1'] = backbone_loss_l1.item()
            stats_batch['backbone_loss_affinity'] = backbone_loss_affinity.item()
            stats_batch['head_loss'] = head_loss.item()

            total_loss = rgb_loss + 10*backbone_loss_l1 + backbone_loss_affinity + head_loss
            total_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, num_iters, phase="train",
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    stats_batch[l], inputs['rgb'].shape[0])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.next()
        bar.finish()




