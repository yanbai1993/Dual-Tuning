#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import pickle

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.DATALOADER.PK_SAMPLER=False
    trainer = DefaultTrainer(cfg)

    model = DefaultTrainer.build_model(cfg)
    traindata=  DefaultTrainer.build_train_loader_center(cfg)
    
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
    
    res = trainer.testcenter(cfg, model, traindata)
    print(res[:10])
    #print(res.keys())
    print(res.shape)
    if (cfg.MODEL.CENTERPATH!=""):
        print('save center results to {}'.format(cfg.MODEL.CENTERPATH))
        pickle.dump(res,open(cfg.MODEL.CENTERPATH,'wb'))
        return 



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
