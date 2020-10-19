from options.train_options import TrainOptions
from util import write_loss_image
from model import DASGIL
from data import get_data_loader
import tensorboardX
import os


def train(opt):
    train_loader = get_data_loader(opt)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opt.output_path, opt.name))
    model = DASGIL(opt).cuda()
    total_steps = 0
    for epoch in range(opt.which_epoch, opt.niter_epoch + 1):
        for i, data in enumerate(train_loader):
            total_steps += opt.batch_size
            print('total_steps:', total_steps, ' epoch_iter:', epoch)
            model.set_input(data)
            model.optimize_params()
            if (total_steps) % opt.print_iter == 0:
                print('Dis loss:',model.loss_f_D, 'Total gen loss:', model.total_loss, 'Gen loss:', model.loss_f_G)
        if (epoch + 1) % opt.log_epoch_freq == 0:
            write_loss_image(epoch, model, train_writer)
        if (epoch + 1) % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch + 1, total_steps))
            model.save(epoch + 1)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    train(opt)
