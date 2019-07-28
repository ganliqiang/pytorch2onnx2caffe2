# load up the caffe2 workspace

import skimage.io
import skimage.transform
import os
import numpy as np
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from pypse import pse as pypse
import cv2


def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img


def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print idx, '/', len(img_paths), img_name
    #cv2.imwrite(output_root + img_name, res)
    skimage.io.imsave(output_root + img_name, res)



INIT_NET = '/home/user/psenet/PSENet/onnx/psenet/init_net.pb'
PREDICT_NET = '/home/user/psenet/PSENet/onnx/psenet/predict_net.pb'
img_path="/home/user/psenet/PSENet/onnx/psenet/5022.jpg"
def rescale(img,input_width=640):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %d") % ( input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
        # landscape orientation - wide image
    res = int(aspect * input_width)
    imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, res))
    
    return imgScaled
def scale(img,size):
    h, w = img.shape[0:2]

    img = cv2.resize(img, size)
    return img

img = skimage.img_as_float(skimage.io.imread(img_path)).astype(np.float32)
#img=cv2.imread(img_path)

org_img = img.copy()

text_box = org_img.copy()
#img=scale(img,(640,640))
#opencv里对应BGR，故通过C通道的 ::-1 就是把BGR转为RGB
#img_ = img[:,:,::-1].transpose((2,0,1))
h,w,_ = img.shape
#img = rescale(img)
img = skimage.transform.resize(img, (640,640))
print "img shape: " , img.shape
# switch to CHW
img = img.swapaxes(1, 2).swapaxes(0, 1)
# switch to BGR
img = img[(2, 1, 0), :, :]
mean = 0.456
std = 0.226
# remove mean for better results
#img = img  - mean
#img = (img-mean) / std
#print img
# add batch size
img = img[np.newaxis, :, :, :].astype(np.float32)
print "NCHW: ", img.shape

with open(INIT_NET) as f:
    init_net = f.read()
with open(PREDICT_NET) as f:
     predict_net = f.read()   
print("img type",type(img))
p = workspace.Predictor(init_net, predict_net)

results = p.run({'data': img})
outputs=results[0]/2
def sigmoid(x):  
    return 1/(1 + np.exp(-x))
score = sigmoid(outputs[:, 0, :, :])
outputs = (np.sign(outputs - 1) + 1) / 2
text = outputs[:, 0, :, :]

kernels = (outputs[:, 0:3, :, :] * text)

#score = score.data.cpu().numpy()[0].astype(np.float32)
#text = text.data.cpu().numpy()[0].astype(np.uint8)
#kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
score = score[0].astype(np.float32)
text = text[0].astype(np.uint8)
kernels = kernels[0].astype(np.uint8)
print("kernels.shape",kernels.shape)

pred = pypse(kernels,1.25)

scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
label = pred
label_num = np.max(label) + 1
bboxes = []
for i in range(1, label_num):
    points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

    if points.shape[0] < 10:
        continue

    score_i = np.mean(score[label == i])
    if score_i < 0.1:
        continue

    rect = cv2.minAreaRect(points)
    bbox = cv2.boxPoints(rect) * scale
    print("scale",scale)
    bbox = bbox.astype('int32')
    bboxes.append(bbox.reshape(-1))



for bbox in bboxes:
    cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)


text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
debug(0, [img_path], [[text_box]], 'outputs/vis_ic15/')

