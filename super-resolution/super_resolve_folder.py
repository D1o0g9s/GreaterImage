from __future__ import print_function

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from torchvision.transforms import ToTensor
from math import log10

from torch.utils.data import DataLoader
from dataset.dataset import DatasetFromFolder, is_image_file
from SRCNN.model import Net

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
from os import listdir
import os

# ===========================================================
# helper functions
# ===========================================================

def downscale(filepath, factor) :
    img = Image.open(filepath)
    width, height = img.size
    newWidth = width // factor
    newHeight = height // factor
    img = img.resize((newWidth, newHeight))
    img.save(filepath)

def upscale(filepath, factor) :
    img = Image.open(filepath)
    width, height = img.size
    newWidth = width * factor
    newHeight = height * factor
    img = img.resize((newWidth, newHeight))
    img.save(filepath)
    
    
def blurrify(filepath, blurWidth, blurHeight) : 
    img = cv2.imread(filepath)
    blur = cv2.blur(img,(blurWidth, blurHeight))
    cv2.imwrite(filepath,blur)
    
def concat_images(img_a, img_b):
    """
    Combines two color image side-by-side.
    :param img_a: image a on left
    :param img_b: image b on right
    :return: combined image
    """
    height_a, width_a = img_a.shape[:2]
    height_b, width_b = img_b.shape[:2]
    max_height = np.max([height_a, height_b])
    total_width = width_a + width_b
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.float32)
    new_img[:height_a, :width_a] = img_a
    new_img[:height_b, width_a:total_width] = img_b
    return new_img

def enshape(pred, original_data) :
    if(pred.shape[2] > original_data.shape[2]) :
        pred = pred[:, :, :original_data.shape[2], :]
    if(pred.shape[2] < original_data.shape[2]) :
        original_data = original_data[:, :, :pred.shape[2], :]
    if(pred.shape[3] > original_data.shape[3]) :
        pred = pred[:, :, :, :original_data.shape[3]]
    if(pred.shape[3] < original_data.shape[3]) :
        original_data = original_data[:, :, :, :pred.shape[3]]
    return pred, original_data

# ===========================================================
# argument + constant settings
# ===========================================================

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--inputFolder', type=str, required=False, default='./images_input', help='input image folder to use')
parser.add_argument('--outputFolder', type=str, required=False, default='./images_output', help='where to save outputs')
parser.add_argument('--compareFolder', type=str, required=False, default='./images_compare', help='where to save comparison images')
parser.add_argument('--model', '-m', type=str, default='model_path.pth', help='model file to use')
parser.add_argument('--outputBW', '-b', type=str, default='false', help='true for output black and white, false for not')
parser.add_argument('--verbose', '-v', type=str, default='true', help='true for verbose output, false for not')
parser.add_argument('--allColors', '-ac', type=str, default='false', help='true or false: true to train on cr and cb in addtion to y')
parser.add_argument('--allLayers', '-al', type=str, default='false', help='true or false: true to train 3 separate neural network layers to predict color.')

args = parser.parse_args()


inputFolder = args.inputFolder
outputFolder = args.outputFolder
compareFolder = args.compareFolder

input_image_filenames = [x for x in listdir(inputFolder) if is_image_file(x)]

modelPath = args.model
upscaleFactor = 4
batchSize = 1
outputBW = True if args.outputBW.strip().lower() == 'true' else False
verbose = True if args.verbose.strip().lower() == 'true' else False

allColors = True if args.allColors.strip().lower() == 'true' else False 
allLayers = True if args.allLayers.strip().lower() == 'true' else False 



# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
myBlurryFile = 'blurry.jpg'
psnrs = list()

if not os.path.exists(outputFolder) :
    os.makedirs(outputFolder)
if not os.path.exists(compareFolder) : 
    os.makedirs(compareFolder)

results_output_file = open(join(outputFolder, "results.txt"),"w+")
results_compare_file = open(join(compareFolder, "results.txt"),"w+")
image_count = 0

report_string = "resolving image with " + modelPath + " upscaleFactor " + str(4) + " BWouput=" +  str(outputBW) + " allColors=" + str(allColors) + " allLayers=" + str(allLayers) +"\n"
print(report_string)
results_output_file.write(report_string)
results_compare_file.write(report_string)

print1 = False

# ===========================================================
# model import & setting
# ===========================================================
numModels = 3 if allLayers else 1
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
models = dict() 
pathExtension=""
modelPath = modelPath.split(".")
for i in range(numModels) :
    models[i] = torch.load(modelPath[0] + pathExtension + "." + modelPath[1], map_location=lambda storage, loc: storage) 
    print(modelPath[0] + pathExtension + "." + modelPath[1])
    pathExtension = pathExtension + "_"
    models[i] = models[i].to(device)
    models[i].eval()

model_y = models[0]
if(allLayers) : 
    model_cb = models[1]
    model_cr = models[2]
else : 
    model_cb = models[0]
    model_cr = models[0]

for inputFileName in input_image_filenames : 
    
    image_count += 1
    if(image_count == 1 or image_count % 100 == 0) : 
        print(image_count, end="")
    else : 
        print(".", end="")
    
    inputFilePath = join(inputFolder, inputFileName)
    
    # Get the original image for testing purposes
    original_img = Image.open(inputFilePath).convert('YCbCr')
    original_y, original_cb, original_cr = original_img.split()

    # Load file, get width and height
    im = Image.open(inputFilePath)
    width, height = im.size
    im.save(myBlurryFile)

    # Blur the image and save (100 is less blur)
    blurrify(myBlurryFile, width // 100, height // 100)

    # Downscale the image by the upscale factor
    downscale(myBlurryFile, upscaleFactor)

    # ===========================================================
    # load images
    # ===========================================================

    # Load the blurry image
    blurry_img = Image.open(myBlurryFile).convert('YCbCr')
    y, cb, cr = blurry_img.split()


    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = data.to(device)

    if GPU_IN_USE:
        cudnn.benchmark = True
    
        
    # ===========================================================
    # output and save image
    # ===========================================================
    pred = models[0](data)
    out = pred.clone().detach()
    out = out.cpu()
    pred = pred.cpu()
    out_img_y = out.data[0].numpy()

    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    if (not outputBW and allColors) :
        data_cb = (ToTensor()(cb)).view(1, -1, cb.size[1], cb.size[0])
        data_cb = data_cb.to(device)
        
        pred_cb = model_cb(data_cb)
        out_cb = pred_cb.clone().detach()

        out_img_cb = out_cb.data[0].numpy()
        out_img_cb *= 255.0
        out_img_cb = out_img_cb.clip(0, 255)
        out_img_cb = Image.fromarray(np.uint8(out_img_cb[0]), mode='L')
        #  print(out_img_cb)
        data_cr = (ToTensor()(cr)).view(1, -1, cr.size[1], cr.size[0])
        if print1 and inputFileName == 'img_002.png': 
            print("pred shape ", pred.shape)
            plt.imshow(pred[0, 0].detach().numpy()),plt.title('pred')
            plt.xticks([]), plt.yticks([])
            plt.show()


            print("data_cb shape ", data_cb.shape)
            plt.imshow(data_cb[0, 0].detach().numpy()),plt.title('data_cb')
            plt.xticks([]), plt.yticks([])
            plt.show()

        data_cr = data_cr.to(device)
        pred_cr = model_cr(data_cr)

        if print1 and inputFileName == 'img_002.png': 

            print("data_cb shape ", data_cb.shape)
            plt.imshow(data_cb[0, 0].detach().numpy()),plt.title('data_cb')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("pred_cb shape ", pred_cb.shape)
            plt.imshow(pred_cb[0, 0].detach().numpy()),plt.title('pred_cb')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("data_cr shape ", data_cr.shape)
            plt.imshow(data_cr[0, 0]),plt.title('data_cr')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("pred_cr shape ", pred_cr.shape)
            plt.imshow(pred_cr[0, 0].detach().numpy()),plt.title('pred_cr')
            plt.xticks([]), plt.yticks([])
            plt.show()
            print1 = False



        out_cr = pred_cr.clone().detach()
        out_img_cr = out_cr.data[0].numpy()

        
        #print(out_img_cr)
        out_img_cr *= 255.0
        out_img_cr = out_img_cr.clip(0, 255)
        out_img_cr = Image.fromarray(np.uint8(out_img_cr[0]), mode='L')
    elif (not outputBW):
        # original
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    elif (outputBW) :
        out_img_cb = Image.fromarray(np.transpose(128*np.uint8(torch.ones(out_img_y.size).numpy())), mode='L')
        out_img_cr = Image.fromarray(np.transpose(128*np.uint8(torch.ones(out_img_y.size).numpy())), mode='L')
    
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    outputPath = join(outputFolder, "output_" + inputFileName)
    out_img.save(outputPath)
    
    #print('output image saved to ', outputPath)

    # ===========================================================
    # calculate metrics and save compare image
    # ===========================================================

    original_data = (ToTensor()(original_y)).view(1, -1, original_y.size[1], original_y.size[0])
    pred, original_data = enshape(pred, original_data)
    criterion = torch.nn.MSELoss()
    mse = criterion(pred, original_data)

    if(not outputBW and allColors) :
        original_data_cb = (ToTensor()(original_cb)).view(1, -1, original_cb.size[1], original_cb.size[0])
        original_data_cr = (ToTensor()(original_cr)).view(1, -1, original_cr.size[1], original_cr.size[0])
        pred_cb, original_data_cb = enshape(pred_cb, original_data_cb)
        pred_cr, original_data_cr = enshape(pred_cr, original_data_cr)
        mse += criterion(pred_cr, original_data_cr)
        mse += criterion(pred_cb, original_data_cb)
        mse = mse / 3

    psnr = 10 * log10(1 / mse.item())
    report_string = inputFileName + "\t psnr between original = " + str(psnr) + "\n"
    if(verbose) :
        print(report_string, end="")
    results_output_file.write(report_string)
    results_compare_file.write(report_string)
    upscale(myBlurryFile, upscaleFactor)
    blurry_img = Image.open(myBlurryFile).convert('RGB')
    original_img = Image.open(inputFilePath).convert('RGB')

    compare_img = concat_images(np.array(blurry_img), np.array(out_img))
    compare_img = Image.fromarray(concat_images(np.array(compare_img), np.array(original_img)).astype('uint8'))
    draw = ImageDraw.Draw(compare_img)
    draw.text((0, 0),"PSNR " + str(psnr), fill=(255,255,255))

    comparePath = join(compareFolder, "compare_" + inputFileName)
    compare_img.save(comparePath)

    psnrs.append(psnr)

report_string = str(len(input_image_filenames)) + " files. Average psnr " + str(np.average(psnrs)) + "\n"
print(report_string, end="")
results_output_file.write(report_string)
results_compare_file.write(report_string)
results_output_file.close() 
results_compare_file.close() 

