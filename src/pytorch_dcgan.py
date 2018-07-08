# based on
# https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/blob/master/pytorch_MNIST_cDCGAN.py

import datetime
import os
import time
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F

try:
    from .dcgan_discriminator import Discriminator
    from .dcgan_generator import Generator
except (ModuleNotFoundError, ImportError):
    # noinspection PyUnresolvedReferences
    from dcgan_discriminator import Discriminator
    # noinspection PyUnresolvedReferences
    from dcgan_generator import Generator


def train(
    name, train_loader, channels, net_resolution, train_epoch, batch_size, lr, save_nth_net=0, load_old=True,
    mbl_size=0, use_feature_match=False, feature_match_weight=10, train_classes=False, decay=None, clean=False, gen_min_quality=0):
    """
    Trains a network on given dataset

    :param name: Name of dataset (Prefix for saved networks, pictures)
    :param train_loader: DataLoader for training samples
    :param channels: Input image channels
    :param net_resolution: Multiplier for depth of convolution / transposed convolution
    :param train_epoch: Number of epochs
    :param batch_size: Batchsize for training
    :param lr: Learning rate
    :param save_nth_net: Save snapshot of every nth network state for later examination
    :param load_old: Should old networks be loaded
    :param mbl_size: If >0, The amount of additional minibatch-discrimination parameters
    :param train_classes: If True, learning is supervised instead of unsupervised
    :param use_feature_match: Enable Feature Match loss
    :param feature_match_weight: Weight of Feature Match loss in comparison to BCE
    :param decay: decay = (n, m): every nth epoch, multiply lr by m
    :param clean: If True, will delete old images and models before training
    :param gen_min_quality: If > 0, the generator will be trained at the end of each epoch to reach this loss
    """

    decay = [5, .3] if decay is None else decay  # default: every 5 epochs multiply by 3/10

    # precomputed one hot matrix to lower compute time
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
    fixed_label = Variable(onehot[torch.LongTensor(5 * 5).random_(0, 10)].cuda())
    # precomputed one hot image matrix to lower compute time
    fill = torch.zeros([10, 10, 64, 64])
    for i in range(10):
        fill[i, i, :, :] = 1

    *_, second_last, last_batch = train_loader
    last_batch_size = last_batch[0].size()[0]
    classes = torch.max(second_last[1]) - torch.min(second_last[1]) + 1

    # network
    G = Generator(channels, net_resolution, train_classes)
    D = Discriminator(channels, classes, net_resolution, mbl_size, train_classes)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    G.cuda()
    D.cuda()

    fixed_z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)  # fixed noise
    fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)

    # Binary Cross Entropy loss
    BCE_loss = nn.BCELoss()

    if clean:
        import shutil
        if os.path.isdir('%s_models' % name):
            shutil.rmtree('%s_models' % name)
        if os.path.isdir('%s_imgs' % name):
            shutil.rmtree('%s_imgs' % name)

    # results save folder
    if not os.path.isdir('%s_imgs' % name):
        os.mkdir('%s_imgs' % name)
    if not os.path.isdir('%s_imgs/random_results' % name):
        os.mkdir('%s_imgs/random_results' % name)
    if not os.path.isdir('%s_imgs/fixed_results' % name):
        os.mkdir('%s_imgs/fixed_results' % name)
    if not os.path.isdir('%s_models' % name):
        os.mkdir('%s_models' % name)

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    num_iter = 0

    if load_old:
        try:
            G = torch.load('%s_models/g.model' % name)
            D = torch.load('%s_models/d.model' % name)
            print("! Old models loaded !")
        except FileNotFoundError:
            pass

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    print('Training start! (%d Epochs)' % train_epoch)
    start_time = time.time()
    n_steps = len(train_loader)
    n_total = second_last[0].size()[0] * (n_steps - 1) + last_batch_size
    for epoch in range(train_epoch):
        if epoch > 0 and epoch % decay[0] == 0:
            G_optimizer.param_groups[0]['lr'] *= decay[1]
            D_optimizer.param_groups[0]['lr'] *= decay[1]

        D_losses = []
        G_losses = []
        epoch_start_time = time.time()
        i_step = -1
        for x_, y_ in train_loader:
            i_step += 1

            # train discriminator D
            D.zero_grad()

            mini_batch = x_.size()[0]
            show_progress(epoch, train_epoch, i_step, n_steps, mini_batch, n_total, batch_size)

            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)

            y_fill = Variable(fill[y_].cuda()) if train_classes else None
            # y_onehot = Variable(onehot[y_].cuda()) if train_classes else None

            x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())

            D_res = D(x_, y_fill)
            D_result = D_res[0].squeeze()
            D_real_map = D_res[1]
            D_real_loss = BCE_loss(D_result, y_real_)

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            label_fake = torch.LongTensor(mini_batch).random_(0, 10)
            y_fill = Variable(fill[label_fake].cuda()) if train_classes else None
            y_onehot = Variable(onehot[label_fake].cuda()) if train_classes else None

            G_result = G(z_, y_onehot)

            D_res = D(G_result, y_fill)
            D_result = D_res[0].squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            if use_feature_match:
                D_train_loss.backward(retain_graph=True)  # Retain feature maps for feature match
            else:
                D_train_loss.backward()
            D_optimizer.step()

            # D_losses.append(D_train_loss.data[0])
            D_losses.append(D_train_loss.data[0])

            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            label_fake = torch.LongTensor(mini_batch).random_(0, 10)
            y_fill = Variable(fill[label_fake].cuda())
            y_onehot = Variable(onehot[label_fake].cuda())

            G_result = G(z_, y_onehot)

            D_res = D(G_result, y_fill)
            D_result = D_res[0].squeeze()
            D_fake_map = D_res[1]
            G_train_loss = BCE_loss(D_result, y_real_)

            # Feature match losses
            if use_feature_match:
                D_real_map = torch.mean(D_real_map, 0)
                D_fake_map = torch.mean(D_fake_map, 0)
                G_train_loss += feature_match_weight * F.pairwise_distance(D_real_map.view(1, -1), D_fake_map.view(1, -1), p=2)
            G_train_loss.backward()
            G_optimizer.step()

            G_losses.append(G_train_loss.data[0])

            num_iter += 1

        stop = gen_min_quality == 0
        post_train_loss = []
        while not stop:
            # train generator G
            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
            z_ = Variable(z_.cuda())

            label_fake = torch.LongTensor(mini_batch).random_(0, 10)
            y_fill = Variable(fill[label_fake].cuda())
            y_onehot = Variable(onehot[label_fake].cuda())

            G_result = G(z_, y_onehot)

            D_res = D(G_result, y_fill)
            D_result = D_res[0].squeeze()
            D_fake_map = D_res[1]
            G_train_loss = BCE_loss(D_result, y_real_)

            # Feature match losses
            if use_feature_match:
                size = torch.Tensor([D_real_map.shape[1] * D_real_map.shape[2] * D_real_map.shape[3]])
                size = Variable(size.cuda())
                D_real_map = torch.mean(D_real_map, 0)
                D_fake_map = torch.mean(D_fake_map, 0)
                G_train_loss += feature_match_weight * 1 / size * F.pairwise_distance(D_real_map.view(1, -1), D_fake_map.view(1, -1), p=2)
            G_train_loss.backward()
            G_optimizer.step()
            
            g_loss = G_train_loss.data[0]
            post_train_loss.append(g_loss)
            if len(post_train_loss) > 10:
                post_train_loss.pop(0)
            avg_loss = sum(post_train_loss)/len(post_train_loss)
            stop = avg_loss < gen_min_quality
            G_losses.append(g_loss)
            

        # Save generator and discriminator after each epoch
        torch.save(G, '%s_models/g.model' % name)
        torch.save(D, '%s_models/d.model' % name)

        if save_nth_net and epoch % save_nth_net == 0:
            torch.save(G, '{}_models/g_{}.model'.format(name, epoch))
            torch.save(D, '{}_models/d_{}.model'.format(name, epoch))

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        show_progress_end(torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))
        # print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
        #    (epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
        #    torch.mean(torch.FloatTensor(G_losses))))
        p = '%s_imgs/random_results/%s_dcgan_%d.png' % (name, name, epoch + 1)
        fixed_p = '%s_imgs/fixed_results/%s_dcgan_%d.png' % (name, name, epoch + 1)
        show_result((epoch + 1), G, fixed_z_, fixed_label, onehot, False, True, p, False, is_rgb=channels > 1)
        show_result((epoch + 1), G, fixed_z_, fixed_label, onehot, False, True, fixed_p, True, is_rgb=channels > 1)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!")
    torch.save(G.state_dict(), "%s_imgs/generator_param.pkl" % name)
    torch.save(D.state_dict(), "%s_imgs/discriminator_param.pkl" % name)
    with open('%s_imgs/train_hist.pkl' % name, 'wb') as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save=True, path='%s_imgs/%s_dcgan_train_hist.png' % (name, name))

    images = []
    for e in range(train_epoch):
        img_name = '%s_imgs/fixed_results/%s_dcgan_%d.png' % (name, name, e + 1)
        images.append(imageio.imread(img_name))
    imageio.mimsave('%s_imgs/generation_animation.gif' % name, images, fps=5)


_total_time = -1
_mini_batch = -1
_is_final_batch = False


def sec2time(sec, n_msec=1):
    """
    Convert seconds to 'HH:MM:SS(.F ...)'

    :param sec: time in seconds
    :param n_msec: amount of decimal places for the seconds (0 -> N; 1 -> N.N; ...)
    :return: formatted time
    """
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    return pattern % (h, m, s)


def _print_progress(e, e_t, p, total, remaining):
    """
    Print the progressbar (only used by show_progress)

    :param e: current epoch
    :param p: progress in percent
    :param total: elapsed time
    :param remaining: remaining time
    """

    total = sec2time(total)
    if remaining < 0:
        remaining = 0
    remaining = sec2time(remaining)

    s = "["
    p_ = int(p * 20)
    p = '%5.1f%%' % (p * 100)

    if p_ >= 14:
        s += p + ' '
    for _ in range(p_ if p_ < 14 else p_ - 7):
        s += '#'
    for _ in range((20 - p_) if p_ >= 14 else (13 - p_)):
        s += ' '
    if p_ < 14:
        s += ' ' + p

    s += ']'

    s = ('\r%%%dd: %%s %%s -%%s' % len(str(e_t))) % (e + 1, total, s, remaining)
    print(s, end="")


def show_progress(e, e_t, i_step, n_steps, mini_batch, n_total, batch_size):
    """
    Displays a progressbar for the learning progress

    :param e: Current epoch
    :param e_t: Total epochs
    :param i_step: index of current minibatch
    :param n_steps: amount of minibatches
    :param mini_batch: size of current batch
    :param n_total: total amount of datapoints
    :param batch_size: size of normal minibatch (not the last one)
    """

    global _total_time, _mini_batch, _is_final_batch
    _is_final_batch = i_step >= n_steps - 1
    t = time.time()

    fix = i_step / (n_steps - 1)

    if _total_time < 0:
        _total_time = t
    tt = t - _total_time

    p = i_step * batch_size + mini_batch * fix + .00001
    p /= n_total

    remaining = tt / p - tt

    _print_progress(e, e_t, p, tt, remaining)

    _mini_batch = mini_batch


def show_progress_end(loss_d, loss_g):
    """
    Helper-function that terminates the progressbar when the process is finished.

    :param loss_d: Loss to be printed at the end
    :param loss_g: Loss to be printed at the end
    """
    global _total_time, _mini_batch, _is_final_batch

    if not _is_final_batch:
        return

    _total_time = -1
    _mini_batch = -1
    _is_final_batch = False
    print(', loss_d: %.3f, loss_g: %.3f' % (loss_d, loss_g))


def show_result(
    num_epoch, G, fixed_z_, fixed_label, onehot, show=False, save=False, path='result.png', is_fix=False, is_rgb=True):
    """
    Shows some generated images for a generator and saves the image to a file.

    :param num_epoch: index of current epoch for labelling
    :param G: Generator
    :param fixed_z_: vector of fixed noise for consistent plots
    :param fixed_label: vector of fixed labels for consistent plots (if classgan)
    :param onehot: onehot-vector for classgan
    :param show: if false, the generated image will not be shown
    :param save: if false, the generated image will not be saved
    :param path: path to the file to which the image is saved
    :param is_fix: if true, use the fixed noise vector
    :param is_rgb: if true, the image is plotted in RGB, else greyscale
    """

    z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)
    to_img = transforms.ToPILImage()
    normalize = transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2))

    G.eval()
    if is_fix:
        test_images = G(fixed_z_, fixed_label)
    else:
        label = torch.LongTensor(5 * 5).random_(0, 10)
        l = Variable(onehot[label].cuda())
        test_images = G(z_, l)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(size_figure_grid, size_figure_grid))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid * size_figure_grid):
        if is_rgb:
            img = test_images[k].cpu().data  # .normal_() (dont do this. You just get noise.)
            img = to_img(normalize(img))

        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()  # clear

        if is_rgb:
            ax[i, j].imshow(img)
        else:
            ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
