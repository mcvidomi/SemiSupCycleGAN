import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import pdb
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import ToyDataset,ToyDataset_paired
from models import Generator, Discriminator,weights_init_normal

"""
To Do: print learning rate lambda in each step

"""



makeVideo = False
batchSize = 20
size = 2
input_nc = 2
output_nc = 2 # size 8
D_lr = 0.01  # 0.0001
G_lr = 0.01  # 0.0002

cyc_lambda = 3
cyc_alpha = 2
cyc_v = 2

n_epochs = 50
epoch = 1
decay_epoch = n_epochs/2
acc_D_A = list()
acc_r18_apt = list()
val_epoch_loss = list()
best_model = False
pairs = 6

seed = 4 #np.random.randint(0,100,1)[0]



# Generate 2D source data [X,Y]
R = np.random.normal(1, 0.01, 10000)
T = np.random.uniform(0, 2*np.pi, 10000)
X = R * np.cos(T)
Y = R * np.sin(T)
source = np.array([X, Y]).T


# Generate 2D target data, by shifting and rotating source
desired_map = np.array([[0.2, 0], [0, 4]])
desired_bias = np.array([1, 3])
target = np.matmul(source, desired_map) + desired_bias

# train test split and paired unpaired

source_train = source[:8000]+2  #source[:8000]
target_train = target[:8000]+2

source_test = source[8000:]
target_test = target[8000:]


index = list(range(len(source_train)))
if seed is not None:
    np.random.seed(seed)
    random.seed(seed)
    np.random.set_state(np.random.RandomState(seed).get_state())

np.random.shuffle(index)


target_paired = target_train[index[:pairs]]
source_paired = source_train[index[:pairs]]
target_unpaired = target_train[index[pairs:]]
source_unpaired = source_train[index[pairs:]]


def columns(d): return (d.iloc[:, 0], d.iloc[:, 1]) if type(d) is pd.DataFrame else (d[:, 0], d[:, 1])

def plot_data_distr(source_unpaired_, target_unpaired_, source_paired_=None, target_paired_=None, savepath="data",legend_= {'Source': 'Source', 'Target': 'Target', 'Source Paired': 'Source Paired', 'Target Paired': 'Target Paired'},seed=99):
    plt.figure()
    plt.cla()
    handles, labels = [], []
    handles.append(plt.scatter(*columns(source_unpaired_), color='r'))
    labels.append(legend_['Source'])
    handles.append(plt.scatter(*columns(target_unpaired_), color='b'))
    labels.append(legend_['Target'])
    if source_paired_ is not None:
        handles.append(plt.scatter(*columns(source_paired_), color='g'))
        labels.append(legend_['Source Paired'])
    if target_paired_ is not None:
        handles.append(plt.scatter(*columns(target_paired_), color='y'))
        labels.append(legend_['Target Paired'])
    plt.legend(handles, labels)
    plt.legend(handles, labels)
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.show()
    plt.savefig(savepath + '.png')
    plt.close()



import os

if not os.path.exists("./results"):
    os.mkdir("./results")

import cv2

if makeVideo:
    ini = False
    for root, dirs, files in os.walk("./results/"):

        for filename in files:
            img = cv2.imread("./results/" + filename)
            if ini == False:
                height , width , layers =  img.shape
                #video = cv2.VideoWriter('swissrole.avi',-1,1,(width,height))
                video = cv2.VideoWriter("results.avi", cv2.VideoWriter_fourcc(*"XVID"), 2,(width,height))

                ini = True

            video.write(img)


    cv2.destroyAllWindows()
    video.release()
    pdb.set_trace()

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def _gradient_penalty( real_data, generated_data, netD):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data)

    if use_gpu:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if use_gpu:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()).cuda() if use_gpu else torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()



plot_data_distr(source_unpaired,target_unpaired,source_paired,target_paired, savepath="results/data",seed=1)



dataloader = DataLoader(ToyDataset(source_unpaired,target_unpaired,source_paired,target_paired), batch_size=batchSize, shuffle=True)
dataloader_paired = DataLoader(ToyDataset_paired(source_paired,target_paired), batch_size=batchSize, shuffle=False)

netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(input_nc, output_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(input_nc)
use_gpu = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_gpu else torch.Tensor

if use_gpu:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_gan = torch.nn.MSELoss()
criterion_cyc = torch.nn.L1Loss()
criterion_cls = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers

#optimizer_G_BA = torch.optim.RMSprop(netG_B2A.parameters(), lr=G_lr)
#optimizer_G_AB = torch.optim.RMSprop(netG_A2B.parameters(), lr=G_lr)
#optimizer_D_A = torch.optim.RMSprop(netD_A.parameters(), lr=D_lr)
#optimizer_D_B = torch.optim.RMSprop(netD_B.parameters(), lr=D_lr)

optimizer_G_BA = torch.optim.Adam(netG_B2A.parameters(), lr=G_lr, betas=(0.5, 0.999))
optimizer_G_AB = torch.optim.Adam(netG_A2B.parameters(), lr=G_lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=D_lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=D_lr, betas=(0.5, 0.999))

lr_scheduler_G_AB = torch.optim.lr_scheduler.StepLR(optimizer_G_AB, step_size=20, gamma=0.1)

                                                      #lr_lambda=LambdaLR(n_epochs, epoch,
                                                       #                  decay_epoch).step)
lr_scheduler_G_BA = torch.optim.lr_scheduler.StepLR(optimizer_G_BA, step_size=20, gamma=0.1)
                                                      #lr_lambda=LambdaLR(n_epochs, epoch,
                                                       #                  decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A, step_size=20, gamma=0.1)
lr_scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B, step_size=20, gamma=0.1)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
###### Training ######dat=next(iter(dataloader))
loss_store = {'real_lossB': [], 'cyc_loss_B': [], 'cyc_loss_A': [],
              'w_dist_B': [], 'w_dist_A': [], 'real_lossA': [], 'cyc_paired_A2B': [], 'cyc_paired_B2A': []}



for epoch in range(n_epochs):
    print(epoch)
    for i, batch in enumerate(dataloader):
        critic = i
        real_A = batch['A']
        real_B = batch['B']


        input_A = Tensor(len(real_A),size) # maybe change here Tensor(opt.batchSize, 1, 1, 8)
        input_B = Tensor(len(real_B),size)

        # Set model input
        real_A = Variable(input_A.copy_(real_A)).cuda()
        real_B = Variable(input_B.copy_(real_B)).cuda()
        real_A_y = 0
        real_B_y = 0

        target_real = Variable(Tensor(len(real_A)).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(len(real_A)).fill_(0.0), requires_grad=False)



        """
        1. Train D_A 
        """
        real_score = netD_A(real_A)
        real_score = real_score.mean()

        fake_A = netG_B2A(real_B)
        fake_score = netD_A(fake_A)
        fake_score = fake_score.mean()

        gp = _gradient_penalty(real_A, fake_A, netD_A)

        w_dist_A = fake_score - real_score

        D_lossA = fake_score - real_score + cyc_lambda * gp


        # Update
        optimizer_G_BA.zero_grad()
        optimizer_G_AB.zero_grad()
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        D_lossA.backward()
        optimizer_D_A.step()

        loss_store["w_dist_A"].append(w_dist_A.data.cpu().numpy())
        loss_store["real_lossA"].append(real_score.data.cpu().numpy())


        """
        2. Train D_B
        """

        real_score = netD_B(real_B)
        real_score = real_score.mean()

        fake_B = netG_A2B(real_A)
        fake_score = netD_B(fake_B)
        fake_score = fake_score.mean()

        gp = _gradient_penalty(real_B, fake_B, netD_B)

        w_dist_B = fake_score - real_score

        D_lossB = fake_score - real_score + cyc_lambda * gp

        # Update
        optimizer_G_BA.zero_grad()
        optimizer_G_AB.zero_grad()
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        D_lossB.backward()
        optimizer_D_B.step()

        loss_store["w_dist_B"].append(w_dist_B.data.cpu().numpy())
        loss_store["real_lossB"].append(real_score.data.cpu().numpy())

        netG_B2A.train()
        netG_A2B.train()
        if np.mod(critic, 5) == 0 and critic > 0:

            # 3. train G_AB

            real_score = netD_A(real_A)
            real_score = real_score.mean()

            fake_A = netG_B2A(real_B)
            fake_score = netD_A(fake_A)
            fake_score = fake_score.mean()

            gp = _gradient_penalty(real_A, fake_A, netD_A)

            D_lossA = fake_score - real_score + cyc_lambda * gp

            fake_B = netG_A2B(real_A)
            cycle_A = netG_B2A(fake_B)

            cyc_loss_A = cyc_v * criterion_cyc(cycle_A, real_A)


            if torch.sum(batch['P_A']) > 10000:

                real_A_paired = torch.stack([batch['A'][x] for x in range(len(batch['P_A'])) if batch['P_A'][x]>0])
                real_B_paired = torch.stack([batch['B'][x] for x in range(len(batch['P_B'])) if batch['P_B'][x]>0])

                input_A_paired = Tensor(len(real_A_paired), size) # maybe change here Tensor(opt.batchSize, 1, 1, 8)
                input_B_paired = Tensor(len(real_B_paired), size)

                # Set model input
                real_A_paired = Variable(input_A_paired.copy_(real_A_paired)).cuda()
                real_B_paired = Variable(input_B_paired.copy_(real_B_paired)).cuda()

                cyc_paired_B2A = criterion_gan(netG_B2A(real_B_paired),real_A_paired)#torch.dist(netG_B2A(real_B_paired), real_A_paired) #
                cyc_paired_B2A = cyc_alpha * cyc_paired_B2A

                cyc_paired_A2B = criterion_gan(netG_B2A(real_B_paired),real_A_paired)#torch.dist(netG_A2B(real_A_paired), real_B_paired) #
                cyc_paired_A2B = cyc_alpha * cyc_paired_A2B

                loss_store["cyc_paired_B2A"].append(cyc_paired_B2A.data.cpu().numpy())
                loss_store["cyc_paired_A2B"].append(cyc_paired_A2B.data.cpu().numpy())


            else:
                cyc_paired_B2A = 0
                cyc_paired_A2B = 0


            G_loss = -D_lossA + cyc_loss_A + cyc_paired_A2B

            # Update
            optimizer_G_BA.zero_grad()
            optimizer_G_AB.zero_grad()
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            G_loss.backward()
            optimizer_G_AB.step()
            optimizer_G_BA.step()

            loss_store["cyc_loss_A"].append(cyc_loss_A.data.cpu().numpy())

            # 4. Traing G_AB, G_BA

            real_score = netD_B(real_B)
            real_score = real_score.mean()

            fake_B = netG_A2B(real_A)
            fake_score = netD_B(fake_B)
            fake_score = fake_score.mean()

            gp = _gradient_penalty(real_B, fake_B, netD_B)

            D_loss_B = fake_score - real_score + cyc_lambda * gp

            fake_A = netG_B2A(real_B)
            cycle_B = netG_A2B(fake_A)
            cyc_loss_B = cyc_v * criterion_cyc(cycle_B, real_B)

            G_loss = -D_loss_B + cyc_loss_B + cyc_paired_B2A

            # Update
            optimizer_G_BA.zero_grad()
            optimizer_G_AB.zero_grad()
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            G_loss.backward()

            optimizer_G_AB.step()
            optimizer_G_BA.step()

            loss_store["cyc_loss_B"].append(cyc_loss_B.data.cpu().numpy())

            for j, batch_paired in enumerate(dataloader_paired):

                real_A_paired = batch_paired['A']
                real_B_paired = batch_paired['B']

                input_A_paired = Tensor(len(real_A_paired), size) # maybe change here Tensor(opt.batchSize, 1, 1, 8)
                input_B_paired = Tensor(len(real_B_paired), size)

                # Set model input
                real_A_paired = Variable(input_A_paired.copy_(real_A_paired)).cuda()
                real_B_paired = Variable(input_B_paired.copy_(real_B_paired)).cuda()

                cyc_paired_B2A = torch.dist(netG_B2A(real_B_paired), real_A_paired) #criterion_gan(netG_B2A(real_B_paired),real_A_paired)
                cyc_paired_B2A = cyc_alpha * cyc_paired_B2A

                cyc_paired_A2B = torch.dist(netG_A2B(real_A_paired), real_B_paired)
                #criterion_gan(netG_B2A(real_B_paired),real_A_paired)
                cyc_paired_A2B = cyc_alpha * cyc_paired_A2B

                loss_store["cyc_paired_B2A"].append(cyc_paired_B2A.data.cpu().numpy())
                loss_store["cyc_paired_A2B"].append(cyc_paired_A2B.data.cpu().numpy())
                G_loss = cyc_paired_A2B + cyc_paired_B2A

                # Update
                optimizer_G_BA.zero_grad()
                optimizer_G_AB.zero_grad()
                optimizer_D_A.zero_grad()
                optimizer_D_B.zero_grad()

                G_loss.backward()

                optimizer_G_AB.step()
                optimizer_G_BA.step()



    lr_scheduler_G_AB.step()
    #print(lr_scheduler_G_AB.get_lr())
    lr_scheduler_G_BA.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    netG_B2A.eval()
    netG_A2B.eval()




    fake_source = netG_B2A(torch.Tensor(target_unpaired).cuda()).data.cpu().numpy()
    fake_target = netG_A2B(torch.Tensor(source_unpaired).cuda()).data.cpu().numpy()
    plot_data_distr(source_unpaired,target_unpaired, source_paired_=fake_source,target_paired_=fake_target, savepath="results/data" + str(epoch),legend_= {'Source': 'Source', 'Target': 'Target', 'Source Paired': 'Fake Source', 'Target Paired': 'Fake Target'},seed=99)


def PL(loss_store,savepath="loss.png"):
        plt.figure('loss')
        plt.cla()

        for loss in loss_store.keys():

            plt.plot(np.arange(len(loss_store[loss])), np.stack(loss_store[loss]), label=loss)

        plt.legend()
        plt.savefig(savepath)


PL(loss_store)


