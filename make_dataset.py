# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from tqdm import tqdm

#アニメキャラクターの顔位置抽出用に使うカスケード分類機の定義
cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
#学習用キャラクター画像が入っているディレクトリパスの定義
data_dir_path = "static/"
file_list = os.listdir(data_dir_path)
dir_list = sorted([x for x in file_list if os.path.isdir(data_dir_path+x)])
input_image=[]
anime_class=[]
for count, dir_name in enumerate(tqdm(file_list)):
	file_list = os.listdir(data_dir_path+dir_name)
	for file_name in file_list:
		#1:画像を読み込む
		image_path = str(data_dir_path)+str(dir_name)+'/'+str(file_name)
		image = cv2.imread(image_path)
		height, width = image.shape[:2]
		#2:カスケード分類器でキャラクターの顔位置抽出を行う
		faces = cascade.detectMultiScale(image,
				scaleFactor = 1.1,
				minNeighbors = 4,
				minSize= (24, 24))
		print('-------------------------------------')
		print(image_path)
		print(faces)
		print('-------------------------------------')
		if len(faces) > 0:  #もし顔が見つかったら
			(x,y,w,h) = faces[0]  #最も確率の高かった顔位置情報を抽出
			if(x<0 or y+h>height or x<0 or x+w>width): #範囲外処理
				image = image  #元画像を使用
			else:
				image = image[y:y+h,x:x+w]  #顔位置のみを切り出す
				print('-------------------------------------3')
		else:
			image = image  #元画像を使用
		#3:画像サイズを64x64にリサイズ
		image = cv2.resize(image, (64, 64))
		#4:画像のチャンネルの位置を変更
		image = image.transpose(2,0,1)
		#5,6:入力画像と正解をそれぞれ保存する
		input_image.append(image)
		anime_class.append(count)
		
		print('-------------------------------------1')

#キャラクター名と正解をcsvファイルに書き出す(後で使う)
with open('file.csv', mode='w') as f:
	for n in dir_list:
		number, name = n.split('_',1)
		f.write(str(number)+','+str(name)+'\n')

#学習画像とキャラクターの番号(正解)を書き出す
np.save('anime_face_class.npy',np.array(anime_class))
np.save('anime_face_image.npy',np.array(input_image))