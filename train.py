from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from options import LiteMonoOptions
from trainer5 import Trainer

options = LiteMonoOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
