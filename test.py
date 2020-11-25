from options.test_options import TestOptions
from model import DASGIL
from data import get_data_loader


def test(opt):
    test_loader_lis_db, test_loader_lis_query = get_data_loader(opt)
    if opt.gpu_ids >= 0:
        model = DASGIL(opt).cuda(opt.gpu_ids)
    else:
        model = DASGIL(opt).cpu()

    for i in range(len(test_loader_lis_db)):
        tensor_db_list_0 = []
        tensor_db_list_1 = []
        path_db_list_0 = []
        path_db_list_1 = []

        # build the feature database for each slice i
        for _, loader in enumerate(test_loader_lis_db[i]):
            print('testing CMU slice:', opt.slice_list[i], 'database:', loader['path'][0])
            imgname_ = loader['path'][0].split('/')
            imgname = imgname_[-1]
            assert '1303' == imgname[13:17]
            if '_c0_' in loader['path'][0]:
                img_c0, img_db_path_0 = model.set_input_db(loader['img'], loader['path'][0])
                tensor_db_0 = model.test_db()
                tensor_db_list_0.append(tensor_db_0)
                path_db_list_0.append(img_db_path_0)

            elif '_c1_' in loader['path'][0]:
                img_c1, img_db_path_1 = model.set_input_db(loader['img'], loader['path'][0])
                tensor_db_1 = model.test_db()
                tensor_db_list_1.append(tensor_db_1)
                path_db_list_1.append(img_db_path_1)

            else:
                print('there is something wrong with database loading')
                break

        last_env_name_c0 = 'database'
        last_env_name_c1 = 'database'
        iteration_c0 = 0
        iteration_c1 = 0
        layer_index = ''
        for layer in opt.trip_layer_index:
            layer_index += str(layer)
        # encode the query feature
        for _, loader in enumerate(test_loader_lis_query[i]):
            print('testing CMU slice:', opt.slice_list[i], 'query:', loader['path'][0])
            imgname_ = loader['path'][0].split('/')
            imgname = imgname_[-1]
            assert '1303' != imgname[13:17]
            if '_c0_' in loader['path'][0]:
                if last_env_name_c0 != imgname[13:17]:
                    iteration_c0 = 0
                query_path_0 = model.set_input_query(loader['img'], loader['path'][0])
                query_path_0 = query_path_0.split('/')[-1]
                tensor_query_0 = model.test_query()
                final_index_0 = model.image_retrieval(tensor_query_0, tensor_db_list_0)
                final_path_0 = path_db_list_0[final_index_0]
                final_imgname_0 = final_path_0.split('/')[-1]
                root = opt.data_root
                slice_ = opt.slice_list[i]
                pos_0 = find_pos(root, slice_, final_imgname_0)
                # save slice result for camera 0
                fname = opt.output_path + '/' + opt.name + '/' + opt.name + '_slice' + str(slice_) + '_epoch' + str(
                    opt.which_epoch)  + '_layer' + layer_index + '.txt'

                with open(fname, 'a')as f:
                    f.write(query_path_0)
                    f.write(' ')
                    f.write(pos_0)
                    f.write('\n')
                iteration_c0 += 1
                last_env_name_c0 = imgname[13:17]
            elif '_c1_' in loader['path'][0]:
                if last_env_name_c1 != imgname[13:17]:
                    iteration_c1 = 0
                query_path_1 = model.set_input_query(loader['img'], loader['path'][0])
                query_path_1 = query_path_1.split('/')[-1]
                tensor_query_1 = model.test_query()
                final_index_1 = model.image_retrieval(tensor_query_1, tensor_db_list_1)
                final_path_1 = path_db_list_1[final_index_1]
                final_imgname_1 = final_path_1.split('/')[-1]
                root = opt.data_root
                slice_ = opt.slice_list[i]
                pos_1 = find_pos(root, slice_, final_imgname_1)
                # save slice result for camera 1
                fname = opt.output_path + '/' + opt.name + '/' + opt.name + '_slice' + str(slice_) + '_epoch' + str(
                    opt.which_epoch) + '_layer' + layer_index + '.txt'

                with open(fname, 'a') as f:
                    f.write(query_path_1)
                    f.write(' ')
                    f.write(pos_1)
                    f.write('\n')
                iteration_c1 += 1
                last_env_name_c1 = imgname[13:17]
            else:
                break
    # the merged txt result
    merge_txt = opt.output_path + '/' + opt.name + '/' + opt.name + '_epoch' + str(opt.which_epoch) + '_layer' + layer_index + '.txt'

    for sli in opt.slice_list:
        fname = opt.output_path + '/' + opt.name + '/' + opt.name + '_slice' + str(sli) + '_epoch' + str(
            opt.which_epoch)  + '_layer' + layer_index + '.txt'
        with open(fname) as f:
            for data in f.readlines():
                data = data.rstrip('\n')
                with open(merge_txt, 'a') as m_f:
                    m_f.write(data)
                    m_f.write('\n')


# find the poses of database images
def find_pos(root, slice_, imgname):
    path = root + 'slice' + str(slice_) + '/' + 'pose_new_s' + str(slice_) + '.txt'
    result = ''
    with open(path) as f:
        for pos in f.readlines():
            pos = pos.rstrip('\n')
            pos_sp = pos.split(' ')
            if pos_sp[0] == imgname:
                for i in range(len(pos_sp)):
                    if i == 0:
                        continue
                    result = result + pos_sp[i] + ' '

                break
    len_ = len(result)
    return result[0: len_ - 1]


if __name__ == "__main__":
    opt = TestOptions().parse()
    test(opt)