import os
import torch
import numpy as np
import random
import warnings
from tqdm import trange, tqdm
import torch.utils.data as data
import torchvision.transforms as transforms
from dataloader import TrainData, TestData
from args_setting import args
from natsort import natsorted
import glob
from network import UNet_full
from loss import Fusionloss

warnings.filterwarnings("ignore")

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def Mytrain():
    setup_seed()
    model_path = './modelsave/' + args.model + '/' + args.task + '/'
    os.makedirs(model_path, exist_ok=True)

    lr = args.lr

    temp_dir = './temp/' + args.model + '/' + args.task
    os.makedirs(temp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_set = TrainData(transform=transform)
    train_loader = data.DataLoader(train_set,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=1,
                                   pin_memory=True)

    model = UNet_full()
    model.cuda()
    model.train()

    cont_training = False
    epoch_start = 0
    if cont_training:
        epoch_start = 600
        model_dir = './modelsave/' + args.model + '/' + args.task + '/'
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)

    for epoch in range(epoch_start, args.epoch):
        loss_mean = []
        for idx, datas in enumerate(tqdm(train_loader, desc='[Epoch--%d]' % (epoch + 1))):
            # for idx, datas in tqdm(train_loader):

            # print(len(data))
            img1, img2 = datas

            model, img_fusion, loss_per_img = train(model, img1, img2, lr)
            loss_mean.append(loss_per_img)

        # print loss
        sum_list = 0
        for item in loss_mean:
            sum_list += item
        sum_per_epoch = sum_list / len(loss_mean)
        print('\tLoss:%.5f' % sum_per_epoch)

        # save info to txt file
        strain_path = temp_dir + '/temp_loss.txt'
        Loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(sum_per_epoch.detach().cpu().numpy())
        with open(strain_path, 'a') as f:
            f.write(Loss_file + '\r\n')

        max_model_num = 20
        # save model
        if (epoch + 1) % 1 == 0 or epoch + 1 == args.epoch:
            torch.save(model.state_dict(), model_path + str(epoch + 1) + '_' + '{}.pth'.format(args.model))
            print('model save in %s' % './modelsave/' + args.model + '/' +  args.task)

            model_lists = natsorted(glob.glob('./modelsave/' + args.model + '/' +  args.task + '/*'))
            while len(model_lists) > max_model_num:
                os.remove(model_lists[0])
                model_lists = natsorted(glob.glob('./modelsave/' + args.model + '/' +  args.task + '/*'))

def train(model, img1, img2, lr):
    model.cuda()
    model.train()

    img1 = img1.cuda()
    img2 = img2.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr)

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    img_fusion = model(img1, img2)
    img_fusion = img_fusion.cuda()
    loss = Fusionloss().cuda()

    loss_total = loss(img_fusion, img1, img2)

    opt.zero_grad()
    loss_total.backward()
    opt.step()

    return model, img_fusion, loss_total


if __name__ == '__main__':
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
    Mytrain()
