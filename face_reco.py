import numpy as np
import cv2
import os
import pickle
import time

dic_folders = {}
img_data = []
img_label = []


def add_new_person():
	face123 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	name = input('Enter your name: ')
	# name = 'su'
	if not os.path.isdir(name):
		os.mkdir(name)

	cap = cv2.VideoCapture(0)
	count = 0

	start = time.clock()
	while True:
		_,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face123.detectMultiScale(gray, 1.3, 6)
		# face = None
		for (x,y,w,h) in faces:
			
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
			face = frame[y:y+h,x:x+w]
			# print(face.shape)
			# print("sdsdfdf",(w,h))
			# cv2.imshow('grayface', face)
			if start+1 <= time.clock():
				# print(Y)
				cv2.imwrite(os.path.join(name,str(count)+ '.jpg'), face)
				start = time.clock()
				count +=1
		# time.sleep(0.6)
		cv2.imshow('frame', frame)
		k = cv2.waitKey(5)&0xFF
		if k==27 or count >= 20:
			break



	cap.release()
	cv2.destroyAllWindows()

def load_data():
	folders = [name for name in os.listdir('.') if os.path.isdir(name)]
	count =0
	
	# for name in folders:
		# print(name)
		# dic_folders[name] = count
		# count +=
	for i,j,k in os.walk('.',topdown=False):
			if not os.path.isdir(i[2:]):
				break
			print('loading data for -->'+ i[2:])
			count+=1
			dic_folders[count] = i[2:]
			# print(i[2:])
			# print(k)
			for name in k:
				img = cv2.imread(os.path.join(i[2:],name),0)
				# print(name)
				# if name == '1.jpg':
					# gray = img
					# cv2.putText(gray, i[2:], (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
					
					# cv2.imshow(i[2:],img)
				img_label.append(count)
				img_data.append(img)

				# np.append(img_data,img.reshape(1,-1))
				# np.append(img_label,count)
					
				# img = cv2.imread(os.path.join(name,k), 0)
	# cv2.imread(filename, flags)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# print(len(img_label))
	# print(len(img_data))

def train():
	print('\n\nloading data...')
	load_data()
	print('loading data complete')
	face_detector = cv2.face.LBPHFaceRecognizer_create()
	# print(img_label)
	# print(img_data)
	print('training...\n')
	face_detector.train(img_data,np.array(img_label))
	print('training complete')
	print('saving trained data...\n')
	traindata = open('traindata.pickle', 'wb')
	pickle.dump(dic_folders, traindata)
	face_detector.write('savefile.xml')
	# predict(face_detector)
	traindata.close()
	print('trained data saved!!\n\n')
	# print(img_data)

def predict(face_detector):
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	count = 0
	cap = cv2.VideoCapture(0)
	start = time.clock()
	while True:
		_,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face.detectMultiScale(gray,1.3,5)	        
		for (x,y,w,h) in faces:
			# dx = int(w/100)
			# dy = int(w/20)
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
			cv2.rectangle(frame,(x,y-20),(x+100,y),(0,255,0),-1)
			faceimg = gray[y:y+h,x:x+w]
			Y = face_detector.predict(faceimg)
			name = ''
			try:
				name = dic_folders[Y[0]]
			except Exception:
				name = 'unknown'

			
				# print(start)
				# print(time.clock())
			count +=1
			cv2.putText(frame, name, (x,y-3), cv2.FONT_HERSHEY_PLAIN , 1.5, (0,0,0),1)
		cv2.imshow('frame', frame)
		k = cv2.waitKey(5)&0xFF
		if k==27 :
			break
	cap.release()
	cv2.destroyAllWindows()

def test(face_detector):
	face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	result = {}
	folders = [name for name in os.listdir('.') if os.path.isdir(name)]
	count =0
	
	
	for i,j,k in os.walk('.',topdown=False):
			if not os.path.isdir(i[2:]):
				break
			
			print(i[2:])
			print(k)
			count =1
			for name in k:
				img = cv2.imread(os.path.join(i[2:],name),0)
				p = face_detector.predict(img)
				
				if dic_folders[p[0]] == i[2:]:
					result[i[2:]] = count
					count +=1 
					# cv2.putText(gray, i[2:], (10,10), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
					
					# cv2.imshow(i[2:],img)
				# img_label.append(count)
				# img_data.append(img)

				# np.append(img_data,img.reshape(1,-1))
				# np.append(img_label,count)
	print(result)		
				# img = cv2.imread(os.path.join(name,k), 0)
	# cv2.imread(filename, flags)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	




if __name__ == '__main__':
	while True:
		print('\t\tMenu\n1.Add New Person\n2.Train\n3.predict')
		k = int(input('Enter your choice'))
		if k == 1:
			add_new_person()
		elif k == 2:
			train()
		elif k == 4:
			save_file = open('traindata.pickle','rb')
			face_detector = cv2.face.LBPHFaceRecognizer_create()
			face_detector.read('savefile.xml')
			if not face_detector:
				print('not ok')
			else:
				dic_folders = pickle.load(save_file)
				print(dic_folders)
				# test(face_detector)
				save_file.close()
		else:
			save_file = open('traindata.pickle','rb')
			face_detector = cv2.face.LBPHFaceRecognizer_create()
			face_detector.read('savefile.xml')
			if not face_detector:
				print('not ok')
			else:
				dic_folders = pickle.load(save_file)
				print(dic_folders)
				predict(face_detector)
				save_file.close()