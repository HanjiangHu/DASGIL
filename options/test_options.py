from options.base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--which_epoch', type=int, required=True, default=0,
                                 help='which epoch to load for testing')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--data_root', type=str, default='./datasets/extended_CMU/',
                                 help='path to CMU images')
        self.parser.add_argument('--retrieval_metric', type=str, default='L1',
                                 help='metric used for retrieval, L2 || cos || L1')
        self.parser.add_argument('--slice_list', nargs='+', default=[2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                 type=int, help='which slice to test')
        self.parser.add_argument('--testlayers_w', nargs='+', default=[0,0,0,0,1,1,0,0], type=float,
                                 help='the weight of image representation on each scale for retrieval')
        self.parser.add_argument('--trip_layer_index', nargs='+', default=[5,6], type=int,
                                 help='which layers are used for image retrieval, counting from 1')


