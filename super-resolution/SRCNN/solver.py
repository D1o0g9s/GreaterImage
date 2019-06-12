from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from progress_bar import progress_bar
from matplotlib import pyplot as plt


print1 = False
print2 = False
inputChannels = 1 # number of channels to input into CNN
baseFilter = 64 # number of channels to output in the first Conv layer in CNN
theEpoch = 0
class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader, allLayers):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        print("device: ", self.device)
        self.models = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizers = None
        self.schedulers = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.numModels = 3 if allLayers else 1
        self.outputFilepath = config.outputPath

    def build_model(self):
        self.models = dict() 
        self.optimizers = dict() 
        self.schedulers = dict()
        for i in range(self.numModels): 
            self.models[i] = Net(num_channels=inputChannels, base_filter=baseFilter, upscale_factor=self.upscale_factor).to(self.device)
            self.models[i].weight_init(mean=0.0, std=0.01)
            self.optimizers[i] = torch.optim.Adam(self.models[i].parameters(), lr=self.lr)
            self.schedulers[i] = torch.optim.lr_scheduler.MultiStepLR(self.optimizers[i], milestones=[50, 75, 100], gamma=0.5)

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        

    def save_model(self):
        model_path = self.outputFilepath.split(".")
        model_out_name = model_path[0]
        model_out_extender = ""
        model_out_extension = "." + model_path[1]
        for i in range(self.numModels) :
            torch.save(self.models[i], model_out_name + model_out_extender + model_out_extension)
            for param_tensor in self.models[i].state_dict():
                print(param_tensor, "\t", self.models[i].state_dict()[param_tensor].size())
            model_out_extender = model_out_extender + "_"


        print("Checkpoint saved to {}".format(model_out_name))

    def train(self):
        for i in range(self.numModels): 
            self.models[i].train()
        train_loss = 0
        for batch_num, (datas, targets) in enumerate(self.training_loader):
            for i in range(len(datas)) : 
                data, target = datas[i].to(self.device), targets[i].to(self.device)
                global print2
                if print2 and batch_num == 0 and theEpoch == 1: 
                    print("data shape ", data.shape)
                    plt.imshow(data.data[0, 0, :, :].numpy()),plt.title('data')
                    plt.xticks([]), plt.yticks([])
                    plt.show()
                    if i == 2 : 
                        print2 = False
                self.optimizers[i].zero_grad()
                loss = self.criterion(self.models[i](data), target) / len(datas)
                train_loss += loss.item()
                loss.backward()
                self.optimizers[i].step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    
    def test(self):
        for i in range(self.numModels): 
            self.models[i].eval()
        avg_psnr = 0
        
        with torch.no_grad():
            for batch_num, (datas, targets) in enumerate(self.testing_loader):
                for i in range(len(datas)) : 
                    data, target = datas[i].to(self.device), targets[i].to(self.device)
                    prediction = self.models[i](data)
                    global print1
                    if print1 and batch_num == 0 and theEpoch == self.nEpochs: 
                        print("data shape ", data.shape)
                        plt.imshow(data.data[0, 0, :, :].numpy()),plt.title('data')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

                        print("prediction shape ", prediction.shape)
                        plt.imshow(prediction[0, 0]),plt.title('prediction')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

                        if i == 2 : 
                            print1 = False
                    mse = self.criterion(prediction, target) / len(datas)
                    psnr = 10 * log10(1 / mse.item())
                    avg_psnr += psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            global theEpoch
            theEpoch = epoch
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            for i in range(self.numModels) : 
                self.schedulers[i].step(epoch)
            if epoch == self.nEpochs:
                self.save_model()
