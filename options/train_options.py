from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True
        self.parser.add_argument('--which_epoch', type=int, default=-1,
                                 help='which epoch to load if continuing training, default -1')
        self.parser.add_argument('--niter_epoch', type=int, default=50,
                                 help='how many epochs to train')

        self.parser.add_argument('--dis_nc', type=int, default=64, help='number of channel for discriminator')
        self.parser.add_argument('--dis_nlayers', type=int, default=2, help='number of layers for discriminator')

        self.parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate for generator')
        self.parser.add_argument('--lr_dis', type=float, default=0.001, help='initial learning rate for discriminator')
        self.parser.add_argument('--step_lr_epoch', type=int, default=50, help='epoch step for optimizer scheduler')
        self.parser.add_argument('--gamma_lr', type=float, default=0.5, help='ratio of reducing learning rate per step')
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--lambda_seg', type=float, default=100,
                                 help='weight for segmentation cross entropy loss in generator')
        self.parser.add_argument('--lambda_gan_feature', type=float, default=100,
                                 help='weight for gan loss in generator')
        self.parser.add_argument('--lambda_gan_feature_dis', type=float, default=100,
                                 help='weight for gan loss in discriminator')

        self.parser.add_argument('--lambda_triplet_list', nargs='+', default=[0, 0, 25, 25, 25, 25, 0, 0],
                                 type=float, help='weight for triplet loss for each scale')
        self.parser.add_argument('--trip_layer_index', nargs='+', default=[3,4,5,6], type=int,
                                 help='which layers are used for triplet loss, counting from 1')
        self.parser.add_argument('--margin_list', nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                 type=float, help='margin for triplet loss for each scale')
        self.parser.add_argument('--num_classes', type=int, default=15, help='number of classes for segmentation')

        self.parser.add_argument('--log_epoch_freq', type=int, default=1, help='how many epoch to log on tensorboard')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--print_iter', type=int, default=100,
                                 help='frequency of printing training results on console')

        self.parser.add_argument('--data_root', type=str,
                                 default='./datasets/vkitti/vkitti_2.0.3_rgb',
                                 help='path to RGB images of Virtual KITTI dataset')
        self.parser.add_argument('--depth_root', type=str,
                                 default='./datasets/vkitti/vkitti_2.0.3_depth',
                                 help='path to depth images of Virtual KITTI dataset')
        self.parser.add_argument('--seg_root', type=str,
                                 default='./datasets/vkitti/vkitti_2.0.3_classSegmentation',
                                 help='path to segmentation images of Virtual KITTI dataset')
        self.parser.add_argument('--data_root_real', type=str, default='./datasets/kitti',
                                 help='path to RGB images of KITTI dataset')