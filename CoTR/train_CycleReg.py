from __future__ import print_function
import argparse
import os
import itertools
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_RegMorph, define_G_cyclegan, define_D, GANLoss, get_scheduler, update_learning_rate, GradientLoss
from data import get_train_set, get_test_set
from utils import ReplayBuffer, Transformer_2D

# Training settings
parser = argparse.ArgumentParser(description='cyclegan-pytorch-implementation')
parser.add_argument('--name', type=str, default='CycleReg', help='facades')
parser.add_argument('--dataset', type=str, default='CoTR', help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_train_set(root_path + opt.dataset)
test_set = get_test_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

print('===> Building models')
net_g_a2b = define_G_cyclegan(opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02).cuda()
net_g_b2a = define_G_cyclegan(opt.output_nc, opt.input_nc, opt.ngf, 'instance', False, 'normal', 0.02).cuda()
net_r_a2b = define_RegMorph(opt.input_nc).cuda()
net_r_b2a = define_RegMorph(opt.output_nc).cuda()
net_d_a = define_D(opt.input_nc, opt.ndf, 'basic').cuda()
net_d_b = define_D(opt.output_nc, opt.ndf, 'basic').cuda()

criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionGrad = GradientLoss().cuda()
criterionGAN = GANLoss().cuda()

# setup optimizer
optimizer_g = optim.Adam(itertools.chain(net_g_a2b.parameters(), net_g_b2a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_r = optim.Adam(itertools.chain(net_r_a2b.parameters(), net_r_b2a.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d_a = optim.Adam(net_d_a.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d_b = optim.Adam(net_d_b.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_r_scheduler = get_scheduler(optimizer_r, opt)
net_d_a_scheduler = get_scheduler(optimizer_d_a, opt)
net_d_b_scheduler = get_scheduler(optimizer_d_b, opt)

fake_a_buffer = ReplayBuffer()
fake_b_buffer = ReplayBuffer()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].cuda(), batch[1].cuda()

        ###### Generator ######
        optimizer_r.zero_grad()
        optimizer_g.zero_grad()
        # Gen GAN loss
        fake_b = net_g_a2b(real_a)
        pred_fake = net_d_b(fake_b)
        loss_gen_a2b = criterionGAN(pred_fake, True)

        fake_a = net_g_b2a(real_b)
        pred_fake = net_d_a(fake_a)
        loss_gen_b2a = criterionGAN(pred_fake, True)

        # Reg loss
        warped_real_b, flow_b2a = net_r_b2a(real_b, fake_b)
        loss_reg_mse_b2a = criterionMSE(warped_real_b, fake_b)
        loss_reg_flow_b2a = criterionGrad(flow_b2a)
        loss_reg_b2a = loss_reg_mse_b2a * 20 + loss_reg_flow_b2a * 10

        warped_real_a, flow_a2b = net_r_a2b(real_a, fake_a)
        loss_reg_mse_a2b = criterionMSE(warped_real_a, fake_a)
        loss_reg_flow_a2b = criterionGrad(flow_a2b)
        loss_reg_a2b = loss_reg_mse_a2b * 20 + loss_reg_flow_a2b * 10

        # Gen Cycle loss
        warped_fake_a = net_g_b2a(warped_real_b)
        loss_gen_cycle_aba = criterionL1(warped_fake_a, real_b) * 10

        warped_fake_b = net_g_a2b(warped_real_a)
        loss_gen_cycle_bab = criterionL1(warped_fake_b, real_a) * 10

        # Reg Cycle loss
        rec_real_a, back_flow_a2b = net_r_a2b(warped_real_a, warped_fake_a)
        loss_reg_cycle_aba = criterionL1(rec_real_a, real_a) * 0.1

        rec_real_b, back_flow_b2a = net_r_b2a(warped_real_b, warped_fake_b)
        loss_reg_cycle_bab = criterionL1(rec_real_b, real_b) * 0.1

        # Reg Total loss
        loss_reg = loss_reg_a2b + loss_reg_b2a + loss_reg_cycle_aba + loss_reg_cycle_bab

        # Gen Total loss
        loss_gen = loss_gen_a2b + loss_gen_b2a + loss_gen_cycle_aba + loss_gen_cycle_bab

        loss_g = loss_gen + loss_reg
        loss_g.backward()
        optimizer_g.step()
        optimizer_r.step()

        ###### Discriminator a ######
        optimizer_d_a.zero_grad()
        # Real loss
        pred_real = net_d_a(real_a)
        loss_d_real = criterionGAN(pred_real, True)
        # Fake loss
        #pred_fake = net_d_a(fake_a.detach())
        fake_a_ = fake_a_buffer.push_and_pop(fake_a)
        pred_fake = net_d_a(fake_a_.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # Total loss
        loss_d_a = (loss_d_real + loss_d_fake) * 0.5
        loss_d_a.backward()

        optimizer_d_a.step()

        ###### Discriminator b ######
        optimizer_d_b.zero_grad()
        # Real loss
        pred_real = net_d_b(real_b)
        loss_d_real = criterionGAN(pred_real, True)
        # Fake loss
        #pred_fake = net_d_b(fake_b.detach())
        fake_b_ = fake_b_buffer.push_and_pop(fake_b)
        pred_fake = net_d_b(fake_b_.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # Total loss
        loss_d_b = (loss_d_real + loss_d_fake) * 0.5
        loss_d_b.backward()

        optimizer_d_b.step()

        print("===> Epoch[{}]({}/{}): Loss_G: {:.4f} Loss_Gen: {:.4f} Loss_Reg: {:.4f} Loss_D_a: {:.4f}  Loss_D_b: {:.4f} ".format(
            epoch, iteration, len(training_data_loader), loss_g.item(), loss_gen.item(), loss_reg.item(), loss_d_a.item(), loss_d_b.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_r_scheduler, optimizer_r)
    update_learning_rate(net_d_a_scheduler, optimizer_d_a)
    update_learning_rate(net_d_b_scheduler, optimizer_d_b)

    #checkpoint
    if epoch % 5 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.name)):
            os.mkdir(os.path.join("checkpoint", opt.name))
        net_g_a2b_model_out_path = "checkpoint/{}/netG_a2b_model_epoch_{}.pth".format(opt.name, epoch)
        net_g_b2a_model_out_path = "checkpoint/{}/netG_b2a_model_epoch_{}.pth".format(opt.name, epoch)
        net_r_a2b_model_out_path = "checkpoint/{}/netR_a2b_model_epoch_{}.pth".format(opt.name, epoch)
        net_r_b2a_model_out_path = "checkpoint/{}/netR_b2a_model_epoch_{}.pth".format(opt.name, epoch)
        net_d_a_model_out_path = "checkpoint/{}/netD_a_model_epoch_{}.pth".format(opt.name, epoch)
        net_d_b_model_out_path = "checkpoint/{}/netD_b_model_epoch_{}.pth".format(opt.name, epoch)
        torch.save(net_g_a2b, net_g_a2b_model_out_path)
        torch.save(net_g_b2a, net_g_b2a_model_out_path)
        torch.save(net_r_a2b, net_r_a2b_model_out_path)
        torch.save(net_r_b2a, net_r_b2a_model_out_path)
        torch.save(net_d_a, net_d_a_model_out_path)
        torch.save(net_d_b, net_d_b_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.name))

