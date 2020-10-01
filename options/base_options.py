import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', required=True, type=str,
                                 help='name of the experiment')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                                 help='checkpoint models are saved here')
        self.parser.add_argument('--output_path', type=str, default='./output',
                                 help='logs of tensorboard are saved here, and the test results are also saved here')
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')
        self.parser.add_argument('--resized_h', type=int, default=256, help='resized height')
        self.parser.add_argument('--resized_w', type=int, default=1024, help='resized width')
        self.parser.add_argument('--check_paths', action='store_true', help='return paths to check correspondence')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save options as txt file to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util().mkdirs(expr_dir)
        if self.opt.isTrain:
            file_name = os.path.join(expr_dir, 'train_opt.txt')
        else:
            file_name = os.path.join(expr_dir, 'test_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
