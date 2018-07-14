from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import matplotlib
import numpy
import random
import torch
import os
from utils import *
from torchvision import transforms as transforms

RECTANGLE_SIZE = 4
imgId = 0
imgTextId = 0

# random color
def randomColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# input: numpy.ndarray
# topleft_Corner: (x,y)
# width: rectangle width
# height: rectangle height
# size: line size
def drawRectangle(input, x, y, width, height, size, k, className, confidence):
	# numpy->PIL
	inputImg = Image.fromarray(numpy.uint8(input))
	draw = ImageDraw.Draw(inputImg)
	color  = randomColor()
	font = ImageFont.truetype('Arial.ttf', 24)
	for i in range(1, size + 1):
		draw.rectangle((x + (size - i), y + (size - i), x + width + i, y + height + i), outline=color)
		draw.text((x+size+6,y), k, font = font, fill = color)
		draw.text((x+size+30,y), className, font = font, fill = color)
		draw.text((x+len(className)*14+36,y), confidence, font = font, fill = color)
	return inputImg


# M: k*batchsize*2*3
# sourceCoordinate: k*batchsize*2*2
# return: topleft, bottomright corners
def getSourceCoordinate(M):
	target = torch.tensor([-1.,1.,1.,-1.,1.,1.])
	target = target.view(3,2)
	sourceCoordinate = torch.matmul(M,target)
	if imgId == 0:
		for batch_index in range(M.size(1)):
			print("img", batch_index,"-----------")
			for k_index in range(M.size(0)):
				print("k: ", k_index)
				print(M[k_index, batch_index, :, :])
	x0 = sourceCoordinate[:,:,0,0]*256.+256.
	x1 = sourceCoordinate[:,:,0,1]*256.+256.
	y0 = -(sourceCoordinate[:,:,1,0]*256.-256.)
	y1 = -(sourceCoordinate[:,:,1,1]*256.-256.)
	sourceCoordinate[:,:,0,0] = torch.min(x0, x1)
	sourceCoordinate[:,:,0,1] = torch.max(x0, x1)
	sourceCoordinate[:,:,1,0] = torch.min(y0, y1)
	sourceCoordinate[:,:,1,1] = torch.max(y0, y1)
	return sourceCoordinate

# sourceCoordinate: k*batchsize*2*2
# rectangleInfo: k*batchsize*4
# return:k*batchsize*[x,y,width,height]
def getPredictedRectangle(M):
	coordinate = getSourceCoordinate(M)
	rectangleInfo = torch.zeros(coordinate.size(0), coordinate.size(1), 4)
	rectangleInfo[:,:,0] = coordinate[:,:,0,0]
	rectangleInfo[:,:,1] = coordinate[:,:,1,0]
	rectangleInfo[:,:,2] = coordinate[:,:,0,1]-coordinate[:,:,0,0]
	rectangleInfo[:,:,3] = coordinate[:,:,1,1]-coordinate[:,:,1,0]
	return rectangleInfo


def drawPictures(original_img, rectangle, className, confidence):
	for batch_index in range(shape(className)[1]):
		current_picture = transforms.ToPILImage()(original_img[batch_index]).convert('RGB')
		for iterator_index in range(shape(className)[0]):
			k          = str(iterator_index)
			x          = rectangle[iterator_index, batch_index, 0]
			y          = rectangle[iterator_index, batch_index, 1]
			width      = rectangle[iterator_index, batch_index, 2]
			height     = rectangle[iterator_index, batch_index, 3]
			classname  = className[iterator_index, batch_index]
			conf_value = str(round(confidence[iterator_index, batch_index].item(),3))
			current_picture = drawRectangle(current_picture, x, y, width, height, RECTANGLE_SIZE, k, classname, conf_value)
		global imgId
		current_picture.save("./result_visual/visualize_imgs_test0712/"+str(imgId)+".jpg")
		imgId += 1
	return


def visualize_attentional_regions(original_img, M, scores):
	# rectangle
	rectangle = getPredictedRectangle(M)
	# information
	confidence, className = getPredictedInfo(scores)
	drawPictures(original_img, rectangle, className, confidence)
	return


# write_prediction_target
# ================================================================================
# input:       numpy (one picture)
# prediction:  list  (prediction classname)
# target:      list  (target classname)
# return       PIL
# ================================================================================
def write_prediction_target(input, prediction, target):
	# numpy->PIL
	inputImg = Image.fromarray(numpy.uint8(input))
	draw     = ImageDraw.Draw(inputImg)
	color    = randomColor()
	font     = ImageFont.truetype('Arial.ttf', 24)
	# initialize x coordinates
	prediction_x = 10
	target_x     = 10
	# prediction
	for p_index in range(len(prediction)):
		draw.text((prediction_x,10), prediction[p_index], font = font, fill = color)
		prediction_x = p_index*10 + prediction_x + len(prediction[p_index])*14
	# target
	for t_index in range(len(target)):
		draw.text((target_x,40), target[t_index], font = font, fill = color)
		target_x = t_index*10 + target_x + len(target[t_index])*14

	return inputImg



def visualize_prediction(original_imgs, prediction_list, target_list):
	for batch_index in range(original_imgs.shape[0]):
		current_picture = transforms.ToPILImage()(original_img[batch_index]).convert('RGB')
		current_picture = write_prediction_target(current_picture,prediction_list[batch_index],target_list[batch_index])
		global imgTextId
		current_picture.save("./result_visual/visualize_imgs_test0712/"+str(imgTextId)+".jpg")
		imgTextId += 1
	return