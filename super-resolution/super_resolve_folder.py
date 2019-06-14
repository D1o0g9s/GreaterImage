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
from SRCNN.solver import baseline, original, unbaseline

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
from os import listdir
import os

from skimage.measure import compare_ssim as ssim


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
parser.add_argument('--allLayers', '-al', type=str, default='false', help='true or false: true to train 3 separate neural network layers to resolve color.')
parser.add_argument('--predictColors', '-pc', type=str, default='false', help='true or false: true to train 3 separate neural network layers to predict color.')

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
predictColors = True if args.predictColors.strip().lower() == 'true' else False 

print1 = False

# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
myBlurryFile = 'blurry.jpg'
metrics = list()

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

if GPU_IN_USE:
        cudnn.benchmark = True

# ===========================================================
# model import & setting
# ===========================================================
numModels = 3 if allLayers or predictColors else 1
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



prepArr = baseline if predictColors else original
unprepArr = unbaseline if predictColors else original


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

    # ===========================================================
    # output and save image
    # ===========================================================

    # y
    data_unbaselined = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = prepArr(data_unbaselined).to(device)
    pred = models[0](data)
    pred_squeezed = prepArr(pred)
    out = pred_squeezed.clone().detach()
    out = out.cpu()
    
    out_img_y = unprepArr(out, data_unbaselined).data.numpy()

    out_img_y_image = out_img_y * 255.0
    out_img_y_image = out_img_y_image.clip(0, 255)
    out_img_y_image = Image.fromarray(np.uint8(out_img_y_image[0,0]), mode='L')

    if (not outputBW and allColors) :
        # cb 
        data_cb_unbaselined = (ToTensor()(cb)).view(1, -1, cb.size[1], cb.size[0]) if not predictColors else data_unbaselined
        data_cb = prepArr(data_cb_unbaselined).to(device)
        
        pred_cb = model_cb(data_cb)
        pred_cb_squeezed = prepArr(pred_cb)
        out_cb = pred_cb_squeezed.clone().detach()
        out_cb = out_cb.cpu()

        out_img_cb = unprepArr(out_cb, data_cb_unbaselined).data.numpy()

        out_img_cb_image = out_img_cb * 255.0
        out_img_cb_image = out_img_cb_image.clip(0, 255)
        out_img_cb_image = Image.fromarray(np.uint8(out_img_cb_image[0,0]), mode='L')


        # cr
        data_cr_unbaselined = (ToTensor()(cr)).view(1, -1, cr.size[1], cr.size[0]) if not predictColors else data_unbaselined
        data_cr = prepArr(data_cr_unbaselined).to(device)

        pred_cr = model_cr(data_cr)
        pred_cr_squeezed = prepArr(pred_cr)
        out_cr = pred_cr_squeezed.clone().detach()
        out_cr = out_cr.cpu()

        out_img_cr = unprepArr(out_cr, data_cr_unbaselined).data.numpy()

        out_img_cr_image = out_img_cr * 255.0
        out_img_cr_image = out_img_cr_image.clip(0, 255)
        out_img_cr_image = Image.fromarray(np.uint8(out_img_cr_image[0,0]), mode='L')

        # debug prints
        if print1 and inputFileName == '3096.jpg': 
            # y data, prediction

            print("data_unbaselined shape ", data_unbaselined.shape)
            plt.imshow(data_unbaselined[0,0]),plt.title('data_unbaselined data')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("out_img_y shape ", out_img_y.shape)
            plt.imshow(out_img_y[0,0]),plt.title('out_img_y prediction')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("data_cb_unbaselined shape ", data_cb_unbaselined.shape)
            plt.imshow(data_cb_unbaselined[0,0]),plt.title('data_cb_unbaselined data')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("out_img_cb shape ", out_img_cb.shape)
            plt.imshow(out_img_cb[0,0]),plt.title('out_img_cb prediction')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("data_cr_unbaselined shape ", data_cr_unbaselined.shape)
            plt.imshow(data_cr_unbaselined[0,0]),plt.title('data_cr_unbaselined data')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("out_img_cr shape ", out_img_cr.shape)
            plt.imshow(out_img_cr[0,0]),plt.title('out_img_cr prediction')
            plt.xticks([]), plt.yticks([])
            plt.show()
            print1 = False
    elif (not outputBW):
        # original
        out_img_cb_image = cb.resize(out_img_y_image.size, Image.BICUBIC)
        out_img_cr_image = cr.resize(out_img_y_image.size, Image.BICUBIC)
    elif (outputBW) :
        # make cb and cr "zeroed"
        out_img_cb_image = Image.fromarray(np.transpose(128*np.uint8(torch.ones(out_img_y_image.size).numpy())), mode='L')
        out_img_cr_image = Image.fromarray(np.transpose(128*np.uint8(torch.ones(out_img_y_image.size).numpy())), mode='L')
    


    out_img = Image.merge('YCbCr', [out_img_y_image, out_img_cb_image, out_img_cr_image]).convert('RGB')
    outputPath = join(outputFolder, "output_" + inputFileName)
    out_img.save(outputPath)
    
    #print('output image saved to ', outputPath)

    # ===========================================================
    # calculate metrics and save compare image
    # ===========================================================
    criterion = ssim #torch.nn.MSELoss()

    original_data = (ToTensor()(original_y)).view(1, -1, original_y.size[1], original_y.size[0])
    #print("out_img_y shape ", out_img_y.shape)
    #print("original_data shape ", original_data.shape)
    out_img_y, original_data = enshape(out_img_y, original_data)
    metric_value = criterion(out_img_y[0,0], original_data[0,0].detach().numpy()) # need to add .detach().numpy() if using ssim

    if(not outputBW and allColors) :
        original_data_cb = (ToTensor()(original_cb)).view(1, -1, original_cb.size[1], original_cb.size[0])
        original_data_cr = (ToTensor()(original_cr)).view(1, -1, original_cr.size[1], original_cr.size[0])
        out_img_cb, original_data_cb = enshape(out_img_cb, original_data_cb)
        out_img_cr, original_data_cr = enshape(out_img_cr, original_data_cr)
        metric_value += criterion(out_img_cb[0,0], original_data_cr[0,0].detach().numpy()) # need to add .detach().numpy() if using ssim
        metric_value += criterion(out_img_cr[0,0], original_data_cb[0,0].detach().numpy()) # need to add .detach().numpy() if using ssim
        metric_value = metric_value / 3

    metric_value = metric_value.item() #10 * log10(1 / mse.item())
    report_string = inputFileName + "\t ssim between original = " + str(metric_value) + "\n"
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
    draw.text((0, 0),"SSIM " + str(metric_value), fill=(255,255,255))

    comparePath = join(compareFolder, "compare_" + inputFileName)
    compare_img.save(comparePath)

    metrics.append(metric_value)

report_string = str(len(input_image_filenames)) + " files. Average psnr " + str(np.average(metrics)) + "\n"
print(report_string, end="")
results_output_file.write(report_string)
results_compare_file.write(report_string)
results_output_file.close() 
results_compare_file.close() 

