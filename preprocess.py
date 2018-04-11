import numpy as np
import cv2

def hisEqual(img):
	ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(ycrcb)
	#print len(channels)
	cv2.equalizeHist(channels[0], channels[0])
	cv2.merge(channels, ycrcb)
	cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
	return img

def avgBlur(img):
	# channels = cv2.split(img)
	# for c in channels:
	# 	c = cv2.blur(c,(3,3))
	# cv2.merge(channels,img)
	img = cv2.blur(img,(3,3))
	return img

def lapEnhance(img):
	kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
	# channels = cv2.split(img)
	# for c in channels:
	# 	c = cv2.filter2D(c,-1,kernel)
	# cv2.merge(channels,img)
	img = cv2.filter2D(img,-1,kernel)
	return img
