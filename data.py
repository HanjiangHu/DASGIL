from torchvision import transforms
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from PIL import Image
from util import make_dataset, IMG_EXTENSIONS, get_subdirectory


def get_data_loader(opt):
    batch_size = opt.batch_size
    num_workers = opt.num_workers
    resized_size = (opt.resized_h,opt.resized_w)
    return_paths = True if opt.check_paths else False
    isTrain = opt.isTrain
    if isTrain:
        # return training dataloader
        dataset = ImageFolderTrain(resized_size, return_paths=return_paths,train_fake_folder=opt.data_root,depth_folder=opt.depth_root, seg_folder=opt.seg_root,train_real_folder=opt.data_root_real)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=isTrain, num_workers=num_workers, drop_last=isTrain)
        return loader
    else:
        # return the dataloader list for both database and query images
        test_loader_lis_db = []
        test_loader_lis_query = []
        slice_lis = opt.slice_list
        for slice_ in slice_lis:
            transform_list = [transforms.ToTensor()]
            transform_list = [transforms.Resize(resized_size, Image.BICUBIC)] + transform_list
            transform = transforms.Compose(transform_list)

            test_data_db = ImageFolderCMUExDB(opt=opt, slice_=slice_, root=opt.data_root,
                                              transform=transform, return_paths=True)
            test_data_query = ImageFolderCMUExQ(opt=opt, slice_=slice_, root=opt.data_root,
                                                transform=transform, return_paths=True)

            test_data_loader_db = DataLoader(test_data_db, batch_size=batch_size, shuffle=isTrain, num_workers=num_workers, drop_last=isTrain)
            test_data_loader_query = DataLoader(test_data_query, batch_size=batch_size, shuffle=isTrain, num_workers=num_workers, drop_last=isTrain)

            test_loader_lis_db.append(test_data_loader_db)
            test_loader_lis_query.append(test_data_loader_query)

        return test_loader_lis_db, test_loader_lis_query


def default_loader(path):
    return Image.open(path).convert('RGB')


def syn_depth_loader(path):
    return Image.open(path)


def syn_seg_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderCMUExDB(data.Dataset):
    # database
    def __init__(self, opt, slice_, root, transform=None, return_paths=True,
                 loader=default_loader):
        self.imgname_db_list_total = []
        self.root = root
        self.transform = transform
        self.return_paths = return_paths
        self.db_loader = loader
        self.loader = loader
        self.opt = opt
        self.index0 = 0
        self.index1 = 0
        self.normalize = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.slice = slice_
        self.db_filename = root + 'slice' + str(self.slice) + '/' + \
            'pose_new_s' + str(self.slice) + '.txt'
        with open(self.db_filename) as f:
            for img_db_name in f.readlines():
                img_db_name_ = img_db_name.rstrip('\n')
                img_db_name__ = img_db_name_.split(' ')[0]
                self.imgname_db_list_total.append(img_db_name__)

        self.db_path = root + 'slice' + str(self.slice) + '/' + 'database_rect/'
        self.db_path_lis_0 = []
        self.db_path_lis_1 = []
        self.db_path_lis_total = []

        self.total_lis = []
        for i in self.imgname_db_list_total:
            path = self.db_path + i
            self.db_path_lis_total.append(path)

        self.total_lis = self.db_path_lis_total

    def __getitem__(self, index):
        path = self.total_lis[index]
        img = self.loader(path)
        if self.transform is not None:
                img = self.transform(img)
        return {'img': img, 'path': path}

    def __len__(self):
        return len(self.imgname_db_list_total)


class ImageFolderCMUExQ(data.Dataset):
    # query
    def __init__(self, opt, slice_, root, transform=None, return_paths=True,
                 loader=default_loader):
        self.imgname_query_list_0 = []
        self.imgname_query_list_1 = []
        self.imgname_query_list_total = []
        self.root = root
        self.transform = transform
        self.return_paths = return_paths
        self.query_loader = loader
        self.loader = loader
        self.opt = opt
        self.index0 = 0
        self.index1 = 0
        self.normalize = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.slice = slice_

        self.query_filename = root + 'slice' + str(self.slice) + '/' + \
            'test-images-slice' + str(self.slice) + '.txt'
        with open(self.query_filename) as f:
            for img_query_name in f.readlines():
                img_query_name_ = img_query_name.rstrip('\n')
                self.imgname_query_list_total.append(img_query_name_)

        self.query_path = root + 'slice' + str(self.slice) + '/' + 'query_rect/'
        self.query_path_lis_0 = []
        self.query_path_lis_1 = []
        self.query_path_lis_total = []
        self.total_lis = []
        for i in self.imgname_query_list_total:
            path = self.query_path + i
            self.query_path_lis_total.append(path)

        self.total_lis = self.query_path_lis_total

        for i in self.imgname_query_list_0:
            image_query_path = self.query_path + i
            self.query_path_lis_0.append(image_query_path)
        for i in self.imgname_query_list_1:
            image_query_path = self.query_path + i
            self.query_path_lis_1.append(image_query_path)

    def __getitem__(self, index):
        path = self.total_lis[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.normalize(img)
        return {'img': img, 'path': path}

    def __len__(self):
        return len(self.imgname_query_list_total)


class ImageFolderTrain(data.Dataset):
    # load data for training
    def __init__(self, resized_size, return_paths, train_fake_folder,depth_folder,seg_folder,
                 train_real_folder,loader=default_loader, depth_loader=syn_depth_loader, seg_loader=syn_seg_loader):
        self.resized_size = resized_size
        self.train_fake_root = train_fake_folder
        self.depth_root = depth_folder
        self.seg_root = seg_folder
        self.train_real_root = train_real_folder

        self.loader = loader
        self.depth_loader = depth_loader
        self.seg_loader = seg_loader

        self.return_paths = return_paths
        self.toTensor= transforms.Compose([transforms.ToTensor()])
        self.normalize = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

        self.sequence = get_subdirectory(self.train_fake_root)

        self.all_rgb_paths = []
        for i in list(range(len(self.sequence))):
            env = get_subdirectory(self.train_fake_root + '/' + self.sequence[i])
            images_seq = []
            for j in list(range(len(env))):
                images_seq.append(sorted(make_dataset(self.train_fake_root + '/' + self.sequence[i] + '/' + env[j])))
            self.all_rgb_paths.append(images_seq)

        self.all_depth_paths = []
        for i in list(range(len(self.sequence))):
            env = get_subdirectory(self.depth_root + '/' + self.sequence[i])
            images_seq = []
            for j in list(range(len(env))):
                images_seq.append(sorted(make_dataset(self.depth_root + '/' + self.sequence[i] + '/' + env[j])))
            self.all_depth_paths.append(images_seq)

        self.all_seg_paths = []
        for i in list(range(len(self.sequence))):
            env = get_subdirectory(self.seg_root + '/' + self.sequence[i])
            images_seq = []
            for j in list(range(len(env))):
                images_seq.append(sorted(make_dataset(self.seg_root + '/' + self.sequence[i] + '/' + env[j])))
            self.all_seg_paths.append(images_seq)

        transform_list = [transforms.ToTensor()]

        transform_list = [transforms.Resize(resized_size,
                                            Image.BICUBIC)] + transform_list if resized_size is not None else transform_list
        transform_list = [transforms.RandomHorizontalFlip()] + transform_list
        self.real_transform = transforms.Compose(transform_list)
        root = self.train_real_root
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs

    def __len__(self):
        return max(len(self.all_rgb_paths[0][0]), len(self.imgs))

    def __getitem__(self, index):
        # real image
        real_path = self.imgs[index]
        real_img = self.loader(real_path)
        if self.real_transform is not None:
            real_img = self.real_transform(real_img)
            real_img = self.normalize(real_img)

        # fake image, different sequences
        seq_A, seq_B = np.random.randint(0, len(self.sequence), 2)
        env_A = np.random.randint(0, len(get_subdirectory(self.train_fake_root + '/' + self.sequence[seq_A])))
        env_A_prime = np.random.randint(0, len(get_subdirectory(self.train_fake_root + '/' + self.sequence[seq_A])))
        env_B = np.random.randint(0, len(get_subdirectory(self.train_fake_root + '/' + self.sequence[seq_B])))

        self.rgb_imgs_A = self.all_rgb_paths[seq_A][env_A]
        self.depth_imgs_A = self.all_depth_paths[seq_A][env_A]
        self.seg_imgs_A = self.all_seg_paths[seq_A][env_A]

        self.rgb_imgs_A_prime = self.all_rgb_paths[seq_A][env_A_prime]
        self.depth_imgs_A_prime = self.all_depth_paths[seq_A][env_A_prime]
        self.seg_imgs_A_prime = self.all_seg_paths[seq_A][env_A_prime]

        self.rgb_imgs_B = self.all_rgb_paths[seq_B][env_B]
        self.depth_imgs_B = self.all_depth_paths[seq_B][env_B]
        self.seg_imgs_B = self.all_seg_paths[seq_B][env_B]


        if index >= 0:
            index = index % (len(self.rgb_imgs_A) - 1)

        path_rgb_A = self.rgb_imgs_A[index]
        rgb_img_A = self.loader(path_rgb_A)

        path_depth_A = self.depth_imgs_A[index]
        depth_img_A = self.depth_loader(path_depth_A)

        path_seg_A = self.seg_imgs_A[index]
        seg_img_A = self.seg_loader(path_seg_A)

        # find the positive and negative index
        if index < 5:
            ran_idx = np.random.randint(1, 6)
            index_prime = index + ran_idx
        elif (len(self.rgb_imgs_A_prime) - index) < 5 :
            ran_idx = np.random.randint(1, 6)
            index_prime = index - ran_idx
        else:
            ran_idx = np.random.randint(-5, 6)
            if ran_idx > 0:
                index_prime = min(index + ran_idx, len(self.rgb_imgs_A_prime) - 1)
            else:
                index_prime = max(index + ran_idx, 0)

        path_rgb_A_prime = self.rgb_imgs_A_prime[index_prime]
        rgb_img_A_prime = self.loader(path_rgb_A_prime)

        path_depth_A_prime = self.depth_imgs_A_prime[index_prime]
        depth_img_A_prime = self.depth_loader(path_depth_A_prime)

        path_seg_A_prime = self.seg_imgs_A_prime[index_prime]
        seg_img_A_prime = self.seg_loader(path_seg_A_prime)

        index_B = np.random.randint(0, len(self.rgb_imgs_B))

        path_rgb_B = self.rgb_imgs_B[index_B]
        rgb_img_B = self.loader(path_rgb_B)

        path_depth_B = self.depth_imgs_B[index_B]
        depth_img_B = self.depth_loader(path_depth_B)

        path_seg_B = self.seg_imgs_B[index_B]
        seg_img_B = self.seg_loader(path_seg_B)
        A_A_prime_flip_prob = random.random()
        B_flip_prob = random.random()

        # transform the rgb, depth, segmentation simultaneously for A, A' (A_prime) and B
        rgb_img_resized_A, depth_img_resized_A, depth_img_A, seg_img_resized_A = self.transform_triplet(A_A_prime_flip_prob, rgb_img_A, depth_img_A, seg_img_A)
        rgb_img_resized_A_prime, depth_img_resized_A_prime, depth_img_A_prime, seg_img_resized_A_prime = self.transform_triplet(A_A_prime_flip_prob, rgb_img_A_prime, depth_img_A_prime, seg_img_A_prime)
        rgb_img_resized_B, depth_img_resized_B, depth_img_B, seg_img_resized_B = self.transform_triplet(B_flip_prob, rgb_img_B, depth_img_B, seg_img_B)

        if self.return_paths:
            return {'rgb_img_A':rgb_img_resized_A, 'depth_img_A':depth_img_resized_A, 'seg_img_A':seg_img_resized_A, 'path_rgb_A':path_rgb_A, 'path_depth_A':path_depth_A, 'path_seg_A':path_seg_A, \
                'noresize_depth_img_A':depth_img_A, \
                    'rgb_img_A_prime':rgb_img_resized_A_prime, 'depth_img_A_prime':depth_img_resized_A_prime, 'seg_img_A_prime':seg_img_resized_A_prime, 'path_rgb_A_prime':path_rgb_A_prime, \
                        'path_depth_A_prime':path_depth_A_prime, 'path_seg_A_prime':path_seg_A_prime, \
                'noresize_depth_img_A_prime':depth_img_A_prime, \
                    'rgb_img_B':rgb_img_resized_B, 'depth_img_B':depth_img_resized_B, 'seg_img_B':seg_img_resized_B, 'path_rgb_B':path_rgb_B, 'path_depth_B':path_depth_B, 'path_seg_B':path_seg_B, \
                'noresize_depth_img_B':depth_img_B,'real_img':real_img,'real_path':real_path}
        else:
            return {'rgb_img_A':rgb_img_resized_A, 'depth_img_A':depth_img_resized_A, 'seg_img_A':seg_img_resized_A, 'noresize_depth_img_A':depth_img_A, \
                'rgb_img_A_prime':rgb_img_resized_A_prime, 'depth_img_A_prime':depth_img_resized_A_prime, 'seg_img_A_prime':seg_img_resized_A_prime, \
                    'noresize_depth_img_A_prime':depth_img_A_prime, \
                        'rgb_img_B':rgb_img_resized_B, 'depth_img_B':depth_img_resized_B, 'seg_img_B':seg_img_resized_B, 'noresize_depth_img_B':depth_img_B,'real_img':real_img}


    def transform_triplet(self, flip_prob, rgb_img, depth_img, seg_img):
        """
        Image pair processing before feeding into networks.
        :param flip_prob: probability for flipping the images
        :param rgb_img: synthetic rgb image
        :param depth_img: corresponding synthetic depth image
        :param seg_img: corresponding synthetic segmentation image
        :return: identically transformed image pairs
        """
        flip_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=flip_prob)])
        rgb_img = flip_transform(rgb_img)
        depth_img = flip_transform(depth_img)
        seg_img = flip_transform(seg_img)
        if self.resized_size is not None:
            scale_transform = transforms.Compose([transforms.Resize(self.resized_size, Image.BICUBIC)])
            scale_transform_seg = transforms.Compose([transforms.Resize(self.resized_size, Image.NEAREST)])

            rgb_img_resized = scale_transform(rgb_img)
            depth_img_resized = scale_transform(depth_img)
            seg_img_resized = scale_transform_seg(seg_img)

        else:
            rgb_img_resized = rgb_img
            depth_img_resized = depth_img
            seg_img_resized = seg_img

        rgb_img_resized = self.toTensor(rgb_img_resized)
        rgb_img_resized = self.normalize(rgb_img_resized)

        # depth map
        arr_depth_resized = np.array(depth_img_resized, dtype=np.float32)
        arr_depth_resized /= 100
        arr_depth_resized[arr_depth_resized < 0.0] = 0.0
        dep_tensor_resized = torch.from_numpy(arr_depth_resized).unsqueeze(0)
        arr_depth = np.array(depth_img, dtype=np.float32)
        arr_depth /= 100
        arr_depth[arr_depth < 0.0] = 0.0
        dep_tensor = torch.from_numpy(arr_depth).unsqueeze(0)

        # segmentation map
        seg_img_resized = torch.from_numpy(np.array(seg_img_resized)).long()
        seg_img_resized = seg_img_resized.transpose(1,2)
        seg_img_resized = seg_img_resized.transpose(0,1)

        return rgb_img_resized, dep_tensor_resized, dep_tensor, seg_img_resized


