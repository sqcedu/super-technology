import numpy as np
import cv2
import time
import face_recognition

# Threshold = 0.65 # 人脸置信度阈值

'''
功能：计算两张图片的相似度，范围：[0,1]
输入：
	1）人脸A的特征向量
	2）人脸B的特征向量
输出：
	1）sim：AB的相似度
'''
def simcos(A,B):
	A=np.array(A)
	B=np.array(B)
	dist = np.linalg.norm(A - B) # 二范数
	sim = 1.0 / (1.0 + dist) #
	return sim


'''
功能：
输入：
	1）x:人脸库向量（n维）
	2）y：被测人脸的特征向量(1维)
输出：
	1)match：与人脸库匹配列表，如[False,True,True,False]
			 表示被测人脸y与人脸库x的第2,3张图片匹配，与1,4不匹配
	2)max(ressim):最大相似度
'''
def compare_faces(x,y,Threshold):
	ressim = []
	match = [False]*len(x)
	for fet in x:
		sim = simcos(fet,y)
		ressim.append(sim)
	if max(ressim) > Threshold:  #置信度阈值
		match[ressim.index(max(ressim))] = True
	return match,max(ressim)


'''
注册身份
输入：
	1）libpath：人脸库地址
输出：
	1）known_face_encodings：人脸库特征向量
	2）known_face_names：人脸库名字标签
'''
def registeredIdentity(libpath):
	known_face_encodings, known_face_names = [], []
	with open(libpath + 'liblist.txt', 'r') as f:
		lines = f.readlines()
	for line in lines:
		img_lable_name = line.split()
		image = face_recognition.load_image_file(libpath + str(img_lable_name[0]))
		face_locations = face_recognition.face_locations(image)
		# face_locations = face_recognition.face_locations(image, model='cnn')

		face_encoding = face_recognition.face_encodings(image, face_locations)[0]
		# face_encoding = face_recognition.face_encodings(image, face_locations)
		known_face_encodings.append(face_encoding)
		known_face_names.append(str(img_lable_name[1]))
	return known_face_encodings, known_face_names


'''
输入：
	1）testimg：测试图片
	2）known_face_encodings：人脸库特征向量
	3）known_face_names：人脸库名字标签
输出：
	1）retname：预测的名字
	2）retscore：相似度得分
	3）face_locations：人脸位置坐标
'''
def identityRecognition(testimg,known_face_encodings,known_face_names,Threshold):
	face_locations = face_recognition.face_locations(testimg)
	# face_locations = face_recognition.face_locations(testimg, model="cnn")
	face_encodings = face_recognition.face_encodings(testimg, face_locations)
	retname, retscore = "Noface", 0
	for face_encoding in face_encodings:
		matches, score = compare_faces(known_face_encodings, face_encoding,Threshold)
		retname, retscore = "Unknow", 0
		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]
			if score > retscore:
				retname = name
				retscore = score
	return retname, retscore,face_locations


'''
输入：
	1）img:摄像头得到的未裁剪图片
	2）face_locations:人脸位置坐标
	3) name:预测的名字
输出：
	img:加框加年龄备注之后的画面
'''
def age_show(img , face_locations,name):
	for (y0, x1, y1, x0) in face_locations:
		cv2.rectangle(img, (x0, y0), (x1, y1), ( 0, 0,255), 2)
		info = str(name)
		t_size = cv2.getTextSize(str(info), cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
		x2,y2 = x0 + t_size[0] + 3, y0 + t_size[1] + 4
		cv2.rectangle(img, (x0,y0), (x2,y2), (0, 0, 255), -1)  # -1填充作为文字框底色
		cv2.putText(img, info, (x0, y0 +t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
	return img