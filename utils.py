from __future__ import print_function

import os
import json
import logging
import numpy as np
from tqdm import tqdm, trange
from datetime import datetime
from collections import defaultdict

import torch as t
from torch.autograd import Variable


##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = t.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
# ETC
##########################

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger

logger = get_logger()



def prepare_dirs(args):

    if args.model_name:
        args.model_name = "{}_{}".format(args.dataset, args.model_name)
        if os.path.exists(os.path.join(args.log_dir,args.model_name)):
            raise Exception(f"Model args.model_name already exits !! give a differnt name")
    else:
        if args.load_path:
            args.model_dir = args.load_path
        else:
            raise Exception("Atleast one of model name or load path should be specified")

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))
