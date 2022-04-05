from lib.models.centernet3d import CenterNet3D
from lib.models.centernet3d_distill import MonoDistill

def build_model(cfg, flag):
    if cfg['type'] == 'centernet3d':
        return CenterNet3D(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'], flag=flag, model_type=cfg['type'])

    elif cfg['type'] == 'distill':
        return MonoDistill(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'], flag=flag, model_type=cfg['type'])

    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


