#!/usr/bin/env python
import cv2
import Image
import ImageChops
import cv
import os
import numpy as np
import skfuzzy as fuzz
#import skimage.io as io 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import norm
from scipy import sum, average



# params
n_centers = 4 

#n_centers
fuzziness_degree = 2
error = 0.005
maxiter = 1000
key_thresh=0

#define the threshold for key frame selection
th=400000

#threshold for update
bg_cluster_thresh = 100 
ROT=0
IMGDIR=0

def list_images(basePath, contains=None):
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath

#to_grayscale function -----------------------------------------------------------------
def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr


# rgb2gray function ---------------------------------------------------------------------
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])




#resize function -----------------------------------------------------------------------
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


#fuzzy function---------------------------------------------------------------------------
def fuzzy(img):
	img=rgb2gray(img);
	(h,w) = img.shape

	# pixel intensities
	I = img.reshape((1, -1))
	P=I.transpose()

	fpc_max=0
        c_best=1
	
	# fuzz c-means clustering
	centers, u, u0, d, jm, n_iters, fpc = fuzz.cluster.cmeans(I, c=n_centers, m=fuzziness_degree, error=error, maxiter=maxiter, init=None)
	img_clustered = np.argmax(u, axis=0).astype(float)
	
	print "FUzzy"
	img_clustered.shape = img.shape
	imgplot = plt.imshow(img_clustered)
	plt.imsave('bg.jpeg',img_clustered)
	
	return (centers,u)
	
#fuzzy predict function -----------------------------------------------------------------------------------
def fuzzy_predict(bg_frame,frame,centers,u):
	img2=rgb2gray(frame);
	(c,N)=u.shape
	
	# pixel intensities
	I2 = img2.reshape((1, -1))
	P1=I2.transpose()

	# fuzz c-means clustering
	ux, u0x, dx, jmx, n_itersx, fpcx=fuzz.cluster.cmeans_predict(I2, centers, fuzziness_degree, error, maxiter, init=None, seed=None)
	img_cluster = np.argmax(ux, axis=0).astype(float)
	img_cluster.shape=img2.shape
	

	imgplot = plt.imshow(img_cluster)
	plt.imsave('key.jpeg',img_cluster)
	print "Fuzzy predict"
	
	#CLASSIFY PIXELS AS BACKGROUD OR FOREGROUND
	ut=u.transpose() #dimesion changed to NxC
	uxt=ux.transpose()
	(h,w)=img2.shape
	N=h*w
	B=np.zeros(N)
	
	#define alpha - the learning rate
	alpha = 0.2
	for i in range(N):
		r1=ut[i]
		r2=uxt[i]
		rho=0
		for j in range(c):
			a=r1[j]*1000000
			b=r2[j]*1000000
			if a>b:
				rho=rho+r2[j]
			else:
				rho=rho+r1[j]
		
		#UPDATE THE BACKGROUND MODEL
	
		if(rho*1000000>th):
			B[i]=1
			#its a backgorund pixel, upate the cluster and the bgframe
			r1= (1-alpha)*r1 + alpha*r2
			ut[i]=r1
			for a in range(3):
				bg_frame[i/500][i%500][a]=(1-alpha)*bg_frame[i/500][i%500][a] + alpha*frame[i/500][i%500][a]
			
		else:
			#foreground pixel so don't update
			ut[i]=r1
			#print rho

	#update the new membership matrix / background model
	u=ut.transpose()

	#form the background subtracted frame
	sub = np.zeros(N)
	img_cluster.shape = img2.shape
	
	for i in range(h):
		for j in range(w):
			if B[i*w+j] == 0:
				sub[i*w+j]=1
			else:
				sub[i*w+j]=0

	plt.imsave('temp.jpeg',np.array(sub).reshape(img_cluster.shape),cmap=cm.gray)
	# display clusters

	imgplot = plt.imshow(mpimg.imread('temp.jpeg'))
	#plt.show()
	return (u,ux)

#NMS function ------------------------------------------------------------------------------
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
	if len(boxes) == 0:
		return []
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	if probs is not None:
		idxs = probs

	idxs = np.argsort(idxs)

	while len(idxs) > 0:
	
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		overlap = (w * h) / area[idxs[:last]]

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	return boxes[pick].astype("int")
	
###################################################################################################################

#MAIN CODE

#####################################################################################################################

def main1(imgs,ops,bgs,n_centersi,key_threshi,bg_cluster_threshi,bg_color_upi):
	
	#initialize the parameters
	n_centers=n_centersi
	key_thresh=key_threshi
	bg_cluster_thresh=bg_cluster_threshi
	bg_color_up=bg_color_upi
	
	# initialize the HOG descriptor/person detector
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	
	i=0

	# and (2) improve detection accuracy
	image = cv2.imread(bgs)

	# resize the frame, convert it to grayscale, and blur it
	bg_frame = resize(image, width=500)
	num_rows, num_cols = bg_frame.shape[:2]
	gray = to_grayscale(bg_frame.astype(float))
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	(centers,u)=fuzzy(bg_frame)
	prev=gray

	#for background updation initialize the color array,the freq array updation array
	temp_bg = np.zeros((num_rows,num_cols,3))
	freq_bg = np.zeros(bg_frame.shape[:2])


	#denotes the no of pixels updated in the actual bg
	pixels = 0

	# loop over the image paths
	p=list_images(imgs)
	p=sorted(p)
	counter=0
	for imagePath in p:
		counter=counter+1
		# load the image and resize it to (1) reduce detection time
		# and (2) improve detection accuracy
		image = cv2.imread(imagePath)
	
		#array that marks the rejected pixels
		mark_up=np.zeros(bg_frame.shape[:2])	
	
		# resize the frame, convert it to grayscale, and blur it
		frame = resize(image, width=500)
		num_rows, num_cols = frame.shape[:2]	
	
		#convert image to gray scale
		gray = to_grayscale(frame.astype(float))
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
	
		#identify the key frames by calculating abs difference with prev frame
		diff = sum(abs(gray - prev)) / gray.size
		print "Key frame Difference : ",diff
	
		if(diff > key_thresh):
	
			#call fuzzy if its a key frame (u,ux have dimension cXN)
			u,ux=fuzzy_predict(bg_frame,frame,centers,u)
			dif_img = resize(cv2.imread('temp.jpeg'),width=500)
			gray1 = cv2.cvtColor(dif_img, cv2.COLOR_BGR2GRAY)
		
			frameDelta = abs(gray1)
		
			if sum(gray1) is not 0:
				print "yo"
		
			thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

			# dilate the thresholded image to fill in holes,
			# then find contours on thresholded image
		
			thresh = cv2.dilate(thresh, None, iterations=2)
			(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
				cv2.CHAIN_APPROX_SIMPLE)
	
			for c in cnts:
		
				if cv2.contourArea(c) < 50 * 50:#args["min_area"]:
					continue
		
				# then find contours on thresholded image
				(x, y, w, h) = cv2.boundingRect(c)
				print x,y,w,h
		
				if w < 25 or h<25:
					continue
				#crop the rectangle and pass to hog and svm for detecting human
				crop_image=frame[y: y + h, x: x + w]
				#imgplot = plt.imshow(crop_image)
				#plt.imsave('crop.jpeg',crop_image)
				#plt.show()
		
				# detect people in the image
				(rects, weights) = hog.detectMultiScale(crop_image, winStride=(4, 4),
					padding=(16, 16), scale=1.05)
		
				print "Contour Boundaries",rects
			
				# draw the original bounding boxes on the original frame and don't consider those pixels for updation
				for (x1, y1, w1, h1) in rects:
					cv2.rectangle(frame, (x+x1, y+y1), (x+x1 + w1, y+y1 + h1), (0, 0, 255), 2)
			 
				# apply non-maxima suppression to the bounding boxes using a
				# fairly large overlap threshold to try to maintain overlapping
				# boxes that are still people
				rects = np.array([[x1, y1, x1 + w1, y1+ h1] for (x1, y1, w1, h1) in rects])
				pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
			 
				# draw the final bounding boxes
				for (xA, yA, xB, yB) in pick:
					for xu in range(xA,xB):
						for yu in range(yB,yA):
							mark[x+xu][y+yu]=1
					cv2.rectangle(frame, (x+xA, y+yA), (x+xB, y+yB), (0, 255, 0), 2)
	 
				text = "Motion Detected"
			
			#imgplot = plt.imshow(frame)
			plt.imsave(ops+'/res'+str(counter)+'.jpg',frame)
			#plt.show()
		
		
			for x in range(num_rows):
				for y in range(num_cols):
			
					#pixel has been marked for updation and is a foregorund pixel
					if mark_up[x][y] == 0:# and gray[x][y] == 1: 
				
						#when pixel has a different color from bg for the first time
						if freq_bg[x][y] == 0: 
							temp_bg[x][y][0]=frame[x][y][0]
							temp_bg[x][y][1]=frame[x][y][1]
							temp_bg[x][y][2]=frame[x][y][2]
							freq_bg[x][y]=1
						else:
							dif1= temp_bg[x][y][0] - frame[x][y][0]
							dif2= temp_bg[x][y][1] - frame[x][y][1]
							dif3= temp_bg[x][y][2] - frame[x][y][2]
						
							if dif1<0:
								dif1*=-1
							if dif2<0:
								dif2*=-1
							if dif3<0:
								dif3*=-1
							
							if dif1<10 and dif2<10 and dif3<10 and dif1+dif2+dif3<15:
								#if the color is nearby update the bg color by taking their average
								for m in range(3):
									temp_bg[x][y][m]=(temp_bg[x][y][m]+frame[x][y][m])/2 
							
								freq_bg[x][y]+=1
							
								if freq_bg[x][y] == bg_color_up:
									#update pixel by copying the distribution from ux to u
									for a in range(3):
										bg_frame[x][y][a]=temp_bg[x][y][a]
									pixels+=1
									freq_bg[x][y]=0
							else:
								freq_bg[x][y]=0
		imgplot = plt.imshow(bg_frame)
		plt.imsave('bg_frame.jpeg',bg_frame)
		
		if pixels > bg_cluster_thresh: 
			#time to update the background clustering
			(centers,u)=fuzzy(bg_frame)
			pixels=0
		
		if prev is not gray:
			prev=gray
		else:
			print "no"

	

		#for the rectangle boundary
	



