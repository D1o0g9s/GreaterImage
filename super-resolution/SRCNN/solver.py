from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from progress_bar import progress_bar
from matplotlib import pyplot as plt
import numpy as np
import math 


print1 = False 
print2 = False
inputChannels = 1 # number of channels to input into CNN
baseFilter = 64 # number of channels to output in the first Conv layer in CNN
theEpoch = 0

def baseline(arr) :
    tensorArr = arr.clone()
    if(torch.max(tensorArr) - torch.min(tensorArr) != 0) :
        toReturn = (tensorArr - torch.min(tensorArr)) / (torch.max(tensorArr) - torch.min(tensorArr)) 
    else: 
        toReturn = tensorArr
    #print("min " + str( torch.min(toReturn)) +  " max " + str(torch.max(toReturn)))
    if torch.min(toReturn) < 0 : 
        print( "BASELINE MIN IS NEGATIVE AHHHHHHHHHH")

    return toReturn

def original(arr, arr_original=None):
    return arr.clone()

def unbaseline(arr, arr_unbaselined) :
    tensorArr = arr.clone()
    tensorUnbaselined = arr_unbaselined.clone()
    #print(tensorArr.data.numpy())
    toReturn = ((torch.max(tensorUnbaselined) - torch.min(tensorUnbaselined)) * tensorArr) + torch.min(tensorUnbaselined)
    #print("min " + str( torch.min(toReturn)) +  " max " + str(torch.max(toReturn)))
    if torch.min(toReturn) < 0: 
        print("UNBASELINE MIN IS NEGATIVE AHHHH")
    return toReturn

class SRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        print("device: ", self.device)
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.allColors = True if config.allColors.strip().lower() == 'true' else False
        self.allLayers = True if config.allLayers.strip().lower() == 'true' else False 

        self.numModels = 3 if self.allLayers else 1
        self.outputFilepath = config.outputPath
        self.predictColors = True if config.predictColors.strip().lower() == 'true' else False
        self.prepArr = baseline if self.predictColors else original
        self.unprepArr = unbaseline if self.predictColors else original

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
                data_unbaselined = datas[i if not self.predictColors else 0]
                data = self.prepArr(data_unbaselined).to(self.device)

                target_unbaselined = targets[i]
                target = self.prepArr(target_unbaselined).to(self.device)
                

                prediction = self.models[i](data)
                prediction_squeezed = self.prepArr(prediction)
                prediction_unbaselined = self.unprepArr(prediction_squeezed, data_unbaselined)

                self.optimizers[i].zero_grad()
                loss = self.criterion(prediction_squeezed, target) 


                train_loss += loss.item()
                loss.backward()
                self.optimizers[i].step()


                global print2
                if print2 and math.isnan(loss.item()) : #batch_num == 3 and theEpoch == 2: 
                    print("data min and max " ,  torch.min(data), " ", torch.max(data))
                    print("pred min and max " ,  torch.min(prediction_squeezed), " ", torch.max(prediction_squeezed))
                    print("target min and max " ,  torch.min(target), " ", torch.max(target))
                    print("target_unbaselined min and max " ,  torch.min(target_unbaselined), " ", torch.max(target_unbaselined))
                    print("target_unbaselined ", target_unbaselined)
                    print("target ", target)
                    print("target shape ", target.shape)
                    plt.imshow(self.unprepArr(target, target_unbaselined)[0, 0].detach().numpy()),plt.title('train target ' + str(i))
                    plt.xticks([]), plt.yticks([])
                    plt.show()

                    print("data shape ", data.shape)
                    plt.imshow(self.unprepArr(data, data_unbaselined)[0, 0].detach().numpy()),plt.title('train data' + str(i))
                    plt.xticks([]), plt.yticks([])
                    plt.show()

                    print("prediction shape ", prediction.shape)
                    plt.imshow(prediction[0,0].detach().numpy()),plt.title('train prediction' + str(i))
                    plt.xticks([]), plt.yticks([])
                    plt.show()

                    if i == 2 : 
                        print2 = False
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    
    def test(self):
        for i in range(self.numModels): 
            self.models[i].eval()
        avg_psnr = 0
        
        with torch.no_grad():
            for batch_num, (datas, targets) in enumerate(self.testing_loader):
                for i in range(len(datas)) : 
                    data_unbaselined = datas[i if not self.predictColors else 0]
                    data = self.prepArr(data_unbaselined).to(self.device)

                    target_unbaselined = targets[i]
                    target = self.prepArr(target_unbaselined).to(self.device)  
                    
                    
                    prediction = self.models[i](data)
                    prediction_squeezed = self.prepArr(prediction)
                    prediction_unbaselined = self.unprepArr(prediction_squeezed, data_unbaselined)

                    mse = self.criterion(prediction_squeezed, target)
                    psnr = 10 * log10(1 / mse.item())
                    avg_psnr += psnr


                    global print1
                    if print1 and batch_num == 0 and theEpoch == self.nEpochs: 

                        print("data shape ", data.shape)
                        plt.imshow(self.unprepArr(data, data_unbaselined)[0, 0, :, :].numpy()),plt.title('data')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

                        print("prediction shape ", prediction.shape)
                        plt.imshow(prediction[0,0].detach().numpy()),plt.title('prediction')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

                        if i == 2 : 
                            print1 = False
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
