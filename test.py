import csv
import numpy as np
import cv2
import sys
import os
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import VGG

#コマンドライン引数のリスト取得
args = sys.argv

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # @UndefinedVariable

chara_list = [] #キャラクター対応表
with open('file.csv','r') as f:
    reader = csv.reader(f)
    for r in reader:
        chara_list.append(r)

#モデルとデータの準備
model = VGG(9).to(device)
param = torch.load("output/model.pth")
model.load_state_dict(param)
model.eval()
#FILE_NAME = input("file name : ") #標準入力からファイル名を要求
#print(FILE_NAME)
FILE_NAME = args[1]

image = cv2.imread(FILE_NAME, 1)
if image is None:
    print('File Name Error!')
    sys.exit(1)
print(image)
#顔を抽出
height, width = image.shape[:2]
print('-------------------')
print(height)
print(width)
print('-------------------')
cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
print(cascade)
faces = cascade.detectMultiScale(image,
        scaleFactor = 1.1,
        minNeighbors = 4,
        minSize= (24, 24))
print(faces)
if len(faces) > 0:
    (x,y,w,h) = faces[0]
    if(x<0 or y+h>height or x<0 or x+w>width):
        image_face = image
    else:
        image_face = image[y:y+h,x:x+w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,255),2)
else:
    image_face = image

image_face = cv2.resize(image_face, (64, 64))
image_face = image_face.transpose(2,0,1)
image_face = torch.Tensor(image_face.reshape(1,3,64,64))
image_face = image_face.to(device)
#モデルへ適用
output = model(image_face)
par, pred = torch.max(F.softmax(output.data, dim=1), 1)  # @UndefinedVariable
score, chara = float(par), pred
print(chara)
chara_name=''
for cl in chara_list:
    num, name = cl
    num = int(num)
    if chara == num:
        chara_name = name

#結果を標準出力
print('')
print('===========RESULT===========')
print('FILE NAME   : '+FILE_NAME)
print('Character   : '+chara_name)
print('Recognition : '+str(round(score*100, 2))+'%')
print('============================')
print('')



#画像出力先
dirname = 'result'
if not os.path.exists(dirname):
    os.mkdir(dirname)

#結果をCSVに保存
with open(dirname + '/result.csv', 'w') as f:
    basename = os.path.basename(FILE_NAME)
    writer = csv.writer(f)
    writer.writerow([basename, chara_name, str(round(score*100, 2))+'%'])

#結果を画像出力
if len(faces) <= 0:
	x=5
	y=-1
cv2.putText(image, chara_name, (x-5, y+10),
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
cv2.putText(image, '('+str(round(score*100, 2))+'%)', (x-5, y+30),
    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
cv2.imwrite(os.path.join(dirname, 'result.png'),image)

