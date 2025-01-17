from __future__ import print_function

import argparse

from torch.utils.data import DataLoader

from DBPN.solver import DBPNTrainer
from DRCN.solver import DRCNTrainer
from EDSR.solver import EDSRTrainer
from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRGAN.solver import SRGANTrainer
from SubPixelCNN.solver import SubPixelTrainer
from VDSR.solver import VDSRTrainer
from dataset.data import get_training_set, get_test_set

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', '-n', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srcnn', help='choose which model is going to use')
parser.add_argument('--allColors', '-ac', type=str, default='false', help='true or false: true to train on cr and cb in addtion to y')
parser.add_argument('--allLayers', '-al', type=str, default='false', help='true or false: true to train 3 separate neural network layers to resolve color.')
parser.add_argument('--predictColors', '-pc', type=str, default='false', help='true or false: true to train 3 separate neural network layers to predict color.')
parser.add_argument("--outputPath", '-o', type=str, default='model_path.pth', help="the filename to save the model to")
# training and testing folder
parser.add_argument('--trainTestFolder', '-t', default="./dataset/MyTrainTest", help="filepath of folder containing train and test images")

args = parser.parse_args()
allColors = True if args.allColors.strip().lower() == 'true' else False 
allLayers = True if args.allLayers.strip().lower() == 'true' else False 
predictColors = True if args.predictColors.strip().lower() == 'true' else False 


def main():
    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('===> Loading datasets')
    print("allColors is " + str(allColors))

    train_set = get_training_set(args.trainTestFolder, args.upscale_factor, allColors or allLayers or predictColors)
    test_set = get_test_set(args.trainTestFolder, args.upscale_factor, allColors or allLayers or predictColors)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

    if args.model == 'sub':
        model = SubPixelTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srcnn':
        model = SRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'vdsr':
        model = VDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'edsr':
        model = EDSRTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'fsrcnn':
        model = FSRCNNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'drcn':
        model = DRCNTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'srgan':
        model = SRGANTrainer(args, training_data_loader, testing_data_loader)
    elif args.model == 'dbpn':
        model = DBPNTrainer(args, training_data_loader, testing_data_loader)
    else:
        raise Exception("the model does not exist")

    model.run()


if __name__ == '__main__':
    main()
