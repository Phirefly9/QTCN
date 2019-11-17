import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torch import nn
from torchvision import transforms as transforms
import numpy as np

import argparse

from tqdm import trange

from QTCN.models.LSTM import LSTM
from QTCN.models.QTCN import QTCN
from QTCN.models.TCN import TCN
from apex.fp16_utils import *

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=False, action='store_true', help='whether cuda is in use')
    parser.add_argument('--device', default=0, help="what device to use")
    parser.add_argument('--net', default="QTCN", choices=["QTCN", "TCN", "LSTM"])
    parser.add_argument('--fp16', default=False, action='store_true')
    parser.add_argument('--dropout', default=0, type=float, help="dropout to use")
    parser.add_argument('--dataset', default="CIFAR10", type=str, choices=["CIFAR10", "CIFAR100"], help="What dataset to use")
    parser.add_argument('--parallel', default=False, action='store_true', help="activate data parallel")
    parser.add_argument('--dropconnect', default=0, type=float, help="dropconnect to use, not really tested")
    args = parser.parse_args()

    solver = Solver(args)
    solver.run()

def add_grayscale(image):
    gray_image = transforms.functional.to_grayscale(image, num_output_channels=1)
    gray_tensor = transforms.functional.to_tensor(gray_image)
    image_tensor = transforms.functional.to_tensor(image)
    data = torch.cat([gray_tensor, image_tensor])
    return data

def flatten_lstm(image):
    tmp = image.flatten(start_dim=1)
    return tmp.transpose(0,1)

def schedule(epoch):
    if   epoch >=   0 and epoch <  10:
        lrate = 0.001
    elif epoch >=  10 and epoch < 100:
        lrate = 0.01
    elif epoch >= 100 and epoch < 120:
        lrate = 0.01
    elif epoch >= 120 and epoch < 150:
        lrate = 0.001
    elif epoch >= 150:
        lrate = 0.0001
    return lrate

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.device = config.device
        self.train_loader = None
        self.test_loader = None
        self.net = config.net
        self.fp16 = config.fp16
        self.dropout = config.dropout
        self.dropconnect = config.dropconnect
        self.dataset = config.dataset
        self.parallel = config.parallel

        self.outputclasses = 10 if self.dataset == "CIFAR10" else 100

    def load_data(self):
        if self.dataset == "CIFAR10":
            mean = [0.53129727, 0.5259391, 0.52069134, 0.51609874]
            std = [0.28938246, 0.28505746, 0.27971658, 0.27499184]
        else:
            mean = [0.5423671, 0.53410053, 0.5282784, 0.5234337]
            std = [0.3012955, 0.29579896, 0.2906593, 0.286055]

        if self.net in ["QTCN", "TCN"]:
            print("Adding greyscale layer")
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), #transforms.RandomAffine(0, translate=(.125, .0)), 
                                                  add_grayscale,
                                                  transforms.Normalize(mean=mean, std=std),
                                                  transforms.Lambda(lambda img: torch.flatten(img, start_dim=1))])
            test_transform = transforms.Compose([add_grayscale, 
                                                transforms.Normalize(mean=mean, std=std),
                                                transforms.Lambda(lambda img: torch.flatten(img, start_dim=1))])
        elif self.net == "LSTM":
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), #transforms.RandomAffine(0, translate=(.125, .125)),
                                                  add_grayscale,
                                                  transforms.Normalize(mean=mean, std=std), 
                                                  flatten_lstm])
            test_transform = transforms.Compose([add_grayscale, 
                                                transforms.Normalize(mean=mean, std=std),
                                                flatten_lstm])
        else:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), #transforms.RandomAffine(0, translate=(.125, .125)), 
                                                transforms.ToTensor(), transforms.Normalize(mean=mean[1:], std=std[1:]),
                                                transforms.Lambda(lambda img: torch.flatten(img, start_dim=1))])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean =mean[1:], std=std[1:]),
                                                transforms.Lambda(lambda img: torch.flatten(img, start_dim=1))])
        if self.dataset == "CIFAR10":
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        else:
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def load_model(self):
        if self.cuda:
            self.device = torch.device(f'cuda:{self.device}')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        channel_sizes = [64] * 8
        if self.net == "QTCN":
            self.model = QTCN(4, self.outputclasses, channel_sizes, kernel_size=10, dropout=self.dropout).to(self.device)
        elif self.net == "LSTM":
            self.model = LSTM(4, self.outputclasses, dropout=0).to(self.device)
        else:
            self.model = TCN(4, self.outputclasses, channel_sizes, kernel_size=8, dropout=self.dropout).to(self.device)
        if self.fp16:
            self.model = network_to_half(self.model)
        if self.parallel:
            self.model =  nn.DataParallel(self.model)

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=.0001)# ,momentum=.9)
        #self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=.0001)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=.9, nesterov=True, weight_decay=.0001)
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=schedule)

        # QTCN with a starting TCN layer must have a milestone at 20 or it diverges
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[ 75, 150], gamma=0.1)
        if self.fp16:
            self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        print("number of params in model is: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def train(self):
        print("train:")
        self.model.train()
        train_loss = 0
        train_correct = 0
        total = 0

        num_batches = len(self.train_loader)
        with trange(num_batches) as pbar:
            for batch_num, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                if self.fp16:
                    data = data.half()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                if self.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                train_loss += loss.item()
                prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
                total += target.size(0)

                # train_correct incremented by one if predicted right
                train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                pbar.set_postfix(Loss=(train_loss / (batch_num + 1)), Acc=(100. * train_correct / total))
                pbar.update(1)

        return train_loss, train_correct / total

    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        
        num_batches = len(self.test_loader)
        with trange(num_batches) as pbar:
            with torch.no_grad():
                for batch_num, (data, target) in enumerate(self.test_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    if self.fp16:
                        data = data.half()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    test_loss += loss.item()
                    prediction = torch.max(output, 1)
                    total += target.size(0)
                    test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                    pbar.set_postfix(Loss=(test_loss / (batch_num + 1)), Acc=(100. * test_correct / total))
                    pbar.update(1)

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        self.load_data()
        self.load_model()
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            
            print("\n===> epoch: %d/200" % epoch)
            train_result = self.train()
            self.scheduler.step(epoch)
            print(train_result)
            test_result = self.test()
            accuracy = max(accuracy, test_result[1])
            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()


if __name__ == '__main__':
    main()
