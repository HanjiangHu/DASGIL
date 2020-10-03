import torch
from networks import Generator, FeatureDiscriminator, init_weights
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import os


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        try:
            return self.loss(F.log_softmax(outputs, dim=1), targets)
        except TypeError as t:
            return self.loss(F.log_softmax(outputs), targets)


class DASGIL(nn.Module):
    def name(self):
        return 'DASGIL'

    def __init__(self, opt):
        super(DASGIL, self).__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # generator
        self.generator = Generator(opt)
        if opt.isTrain:
            self.generator.weight_init(0, 0.02)
            self.gen_parameters = list(self.generator.parameters())
            self.gen_optimizer = torch.optim.Adam(self.gen_parameters, lr=opt.lr, betas=(0.9, 0.999))
            # discriminator
            self.dis_f = FeatureDiscriminator(opt.dis_nc, opt.dis_nlayers)
            init_weights(self.dis_f, 'normal')
            dis_params = list(self.dis_f.parameters())
            self.dis_optimizer = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=opt.lr_dis,
                                                  betas=(0.5, 0.9))
            weight = torch.ones(self.opt.num_classes)
            self.criterion_corssentropy = CrossEntropyLoss2d(weight)

            # initialize inputs
            self.input_GAN_real = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)
            self.input_rgb_A = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)
            self.input_depth_A = self.Tensor(opt.batch_size, 1, opt.resized_h, opt.resized_w)
            self.input_seg_A = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)

            self.input_rgb_A_prime = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)
            self.input_depth_A_prime = self.Tensor(opt.batch_size, 1, opt.resized_h, opt.resized_w)
            self.input_seg_A_prime = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)

            self.input_rgb_B = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)
            self.input_depth_B = self.Tensor(opt.batch_size, 1, opt.resized_h, opt.resized_w)
            self.input_seg_B = self.Tensor(opt.batch_size, 3, opt.resized_h, opt.resized_w)

        # load checkpoints
        if not opt.isTrain or opt.continue_train:
            save_filename_generator = '%d_net_%s' % (opt.which_epoch, 'gen')
            save_path_generator = os.path.join(self.save_dir, save_filename_generator)
            filename_generator = save_path_generator + '.pth'
            self.generator.load_state_dict(torch.load(filename_generator))
        if opt.continue_train:
            save_filename_dis = '%d_net_%s' % (opt.which_epoch, 'dis')
            save_path_dis = os.path.join(self.save_dir, save_filename_dis)
            filename_dis = save_path_dis + '.pth'
            self.dis_f.load_state_dict(torch.load(filename_dis))


        if opt.isTrain:
            # lr scheduler update
            self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, step_size=opt.step_lr_epoch,
                                                     gamma=opt.gamma_lr)
            self.gen_scheduler.last_epoch = opt.which_epoch
            self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, step_size=opt.step_lr_epoch,
                                                 gamma=opt.gamma_lr)
            self.dis_scheduler.last_epoch = opt.which_epoch

    def set_input(self, input):
        # input three pairs of rgb, depth and segmentation map, for A, A'(A_prime) and B, A and A' are positive pairs
        # A and B are negative pairs
        self.input_rgb_A = input['rgb_img_A'].cuda() #[B,3,H,W]
        self.input_depth_A = input['depth_img_A'].cuda() #[B,1,H,W]
        self.input_seg_A = input['seg_img_A'].cuda() #[B,3,H,W]

        self.input_rgb_A_prime = input['rgb_img_A_prime'].cuda()
        self.input_depth_A_prime = input['depth_img_A_prime'].cuda()
        self.input_seg_A_prime = input['seg_img_A_prime'].cuda()

        self.input_rgb_B = input['rgb_img_B'].cuda()
        self.input_depth_B = input['depth_img_B'].cuda()
        self.input_seg_B = input['seg_img_B'].cuda()
        self.input_GAN_real = input['real_img'].cuda()

    def set_input_query(self, input_query, input_path):
        """
        Input the query images while testing
        :param input_query: image tensor from dataloader
        :param input_path: image path
        :return: image tensor on cuda
        """
        self.input_query_test = input_query.cuda()
        self.input_query_path = input_path
        return self.input_query_path

    def set_input_db(self, input_db, input_path):
        """
        Input the database images while testing
        :param input_db: image tensor from dataloader
        :param input_path: image path
        :return: image tensor on cuda
        """
        self.input_db_test = input_db.cuda()
        self.input_db_path = input_path
        return self.input_db_test, self.input_db_path

    def L1_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def L2_criterion(self, input, target):
        return torch.mean(torch.abs(input - target) ** 2)

    def scale_pyramid(self, img_gt, img_pred_diff_scale, seg=False):
        """
        Make the ground truth of image the same scale as the predicted ones
        :param img_gt: image ground truth
        :param img_pred_diff_scale: the predicted image features at different scale (if multi_resolution not specified,
                the predicted features are at the same size with ground truth already)
        :return: the rescaled ground truth image
        """
        h = img_pred_diff_scale.size()[2]
        w = img_pred_diff_scale.size()[3]
        if seg:
            rescaled_gt = F.interpolate(img_gt.float(), size=(int(h), int(w)), mode='nearest').long()
        else:
            rescaled_gt = F.interpolate(img_gt, size=(int(h), int(w)))
        return rescaled_gt

    def image_retrieval(self, query_tensor, db_tensor):
        """
        Retrieve the most similar database image
        :param query_tensor: query image tensor list
        :param db_tensor: database image tensor list
        :return: the index of retrieval in database
        """
        L2loss = torch.nn.MSELoss()
        L1loss = torch.nn.L1Loss()
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
        mean_cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        min_dist = float("inf")
        final_index = 0
        for i in range(len(db_tensor)):
            dist = 0
            for j in range(len(self.opt.trip_layer_index)):
                if self.opt.retrieval_metric == 'L2':
                    dist += L2loss(query_tensor[j], db_tensor[i][j])
                elif self.opt.retrieval_metric == 'L1':
                    dist += L1loss(query_tensor[j], db_tensor[i][j])
                elif self.opt.retrieval_metric == 'cos':
                    dist += -cos(query_tensor[j].view(-1), db_tensor[i][j].view(-1))
                else:
                    dist += -mean_cos(query_tensor[j].squeeze().view(query_tensor[j].squeeze().shape[0], -1),
                                      db_tensor[i][j].squeeze().view(db_tensor[i][j].squeeze().shape[0], -1)).mean(0)
            if dist < min_dist:
                min_dist = dist
                final_index = i
        return final_index

    def test_db(self):
        with torch.no_grad():
            depth_op_db_test, seg_op_db_test, encoded_db_test_mid, seg_mask = self.generator(self.input_db_test)
            tensor_db_test_list = []

            for i in self.opt.trip_layer_index:
                i -= 1
                tensor_db_test_list.append(encoded_db_test_mid[i] * self.opt.testlayers_w[i])

            return tensor_db_test_list

    def test_query(self):
        with torch.no_grad():
            depth_op_query_test, seg_op_query_test, encoded_query_test_mid, test_seg_mask = self.generator(self.input_query_test)
            tensor_query_test_list = []

            for i in self.opt.trip_layer_index:
                i -= 1
                tensor_query_test_list.append(encoded_query_test_mid[i] * self.opt.testlayers_w[i])
            return tensor_query_test_list

    def backward_D(self):
        _, _, mid_feature_syn_afterGen, _ = self.generator(self.input_rgb_A)
        self.depth_op_real, self.seg_op_real, mid_feature_real, seg_mask = self.generator(self.input_GAN_real)
        self.depth_output_real = self.depth_op_real[-1]
        self.seg_output_real = seg_mask

        D_syn = self.dis_f(mid_feature_syn_afterGen)
        D_real = self.dis_f(mid_feature_real)

        # distingush real and syn
        D_loss = (torch.mean((D_real - 1.0) ** 2) + torch.mean((D_syn - 0.0) ** 2)) * 0.5
        self.loss_f_D = D_loss
        self.loss_f_D.backward()

    def backward_G(self):
        # generative loss
        self.depth_op_A, self.seg_op_A, self.mid_feature_syn_A, seg_mask_A = self.generator(self.input_rgb_A)
        D_syn = self.dis_f(self.mid_feature_syn_A)
        G_loss = torch.mean((D_syn - 1.0) ** 2) * 0.5
        self.loss_f_G = G_loss * self.opt.lambda_gan_feature

        self.depth_op_A_prime, self.seg_op_A_prime, self.mid_feature_syn_A_prime, seg_mask_A_prime = self.generator(self.input_rgb_A_prime)
        self.depth_op_B, self.seg_op_B, self.mid_feature_syn_B, seg_mask_B = self.generator(self.input_rgb_B)
        self.seg_output_A = seg_mask_A
        self.seg_output_A_prime = seg_mask_A_prime
        self.seg_output_B = seg_mask_B

        self.margin_list = self.opt.margin_list
        self.triplet_weight_list = self.opt.lambda_triplet_list
        dn_list = []
        dp_list = []
        trip_list = []
        L2loss = torch.nn.MSELoss()

        # triplet loss
        self.triplet_loss_total = 0
        for i in self.opt.trip_layer_index:
            i -= 1
            mid_feature_syn_A = self.mid_feature_syn_A
            mid_feature_syn_A_prime = self.mid_feature_syn_A_prime
            mid_feature_syn_B = self.mid_feature_syn_B

            self.tensor_A = mid_feature_syn_A[i]
            self.tensor_A_prime = mid_feature_syn_A_prime[i]
            self.tensor_B = mid_feature_syn_B[i]

            dn_loss = L2loss(self.tensor_A, self.tensor_B)
            dp_loss = L2loss(self.tensor_A, self.tensor_A_prime)
            triplet_loss = torch.max(torch.cuda.FloatTensor([0.0]),
                                     1 - dn_loss /
                                     (self.margin_list[i] + dp_loss)) * self.triplet_weight_list[i]
            self.triplet_loss_total += triplet_loss
            dn_list.append(dn_loss)
            dp_list.append(dp_loss)
            trip_list.append(triplet_loss)

        # depth loss
        self.dep_output_A = self.depth_op_A[-1]
        self.depth_loss_A = self.L1_criterion(self.depth_op_A[0],
                                              self.scale_pyramid(self.input_depth_A, self.depth_op_A[0]))
        self.depth_ls_A_prime = self.L1_criterion(self.depth_op_A_prime[0], self.scale_pyramid(self.input_depth_A_prime,
                                                                                               self.depth_op_A_prime[0]))
        self.depth_ls_B = self.L1_criterion(self.depth_op_B[0],
                                            self.scale_pyramid(self.input_depth_B, self.depth_op_B[0]))
        for item in self.depth_op_A[1:]:
            self.depth_loss_A += self.L1_criterion(item, self.scale_pyramid(self.input_depth_A, item))
        self.depth_loss_A = self.depth_loss_A / len(self.depth_op_A)
        for item in self.depth_op_A_prime[1:]:
            self.depth_ls_A_prime += self.L1_criterion(item, self.scale_pyramid(self.input_depth_A_prime, item))
        self.depth_ls_A_prime = self.depth_ls_A_prime / len(self.depth_op_A_prime)
        for item in self.depth_op_B[1:]:
            self.depth_ls_B += self.L1_criterion(item, self.scale_pyramid(self.input_depth_B, item))
        self.depth_ls_B = self.depth_ls_B / len(self.depth_op_B)

        # segmentation loss
        self.seg_loss_A = self.criterion_corssentropy(self.seg_op_A[-1], self.input_seg_A[:, 0, :, :])
        self.seg_ls_A_prime = self.criterion_corssentropy(self.seg_op_A_prime[-1], self.input_seg_A_prime[:, 0, :, :])
        self.seg_ls_B = self.criterion_corssentropy(self.seg_op_B[-1], self.input_seg_B[:, 0, :, :])
        self.seg_loss_A = self.seg_loss_A * self.opt.lambda_seg
        self.seg_ls_A_prime = self.seg_ls_A_prime * self.opt.lambda_seg
        self.seg_ls_B = self.seg_ls_B * self.opt.lambda_seg

        # total loss
        self.total_loss = self.depth_loss_A + self.seg_loss_A + self.depth_ls_A_prime + self.seg_ls_A_prime + \
                          self.depth_ls_B + self.seg_ls_B + self.triplet_loss_total + \
                          self.loss_f_G

        self.total_loss.backward()

    def optimize_params(self):
        # adversarial training
        self.gen_optimizer.zero_grad()
        self.backward_G()
        self.gen_optimizer.step()

        self.dis_optimizer.zero_grad()
        self.backward_D()
        self.dis_optimizer.step()

    def save(self, epoch):
        # save the checkpoint model
        save_filename_generator = '%d_net_%s' % (epoch, 'gen')
        save_path_generator = os.path.join(self.save_dir, save_filename_generator)
        filename_generator = save_path_generator + '.pth'
        torch.save(self.generator.cpu().state_dict(), filename_generator)

        save_filename_dis = '%d_net_%s' % (epoch, 'dis')
        save_path_dis = os.path.join(self.save_dir, save_filename_dis)
        filename_dis = save_path_dis + '.pth'
        torch.save(self.dis_f.cpu().state_dict(), filename_dis)

        self.generator.cuda(self.gpu_ids[0])
        self.dis_f.cuda(self.gpu_ids[0])
