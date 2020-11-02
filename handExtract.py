from __future__ import print_function
import cv2
import argparse
import numpy as np
max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Hand Extract'
isColor = False

def nothing(x):
    pass

cam = cv2.VideoCapture(0)
cv2.namedWindow(window_name)    
cv2.createTrackbar(trackbar_type, window_name , 3, max_type, nothing)
# Create Trackbar to choose Threshold value
cv2.createTrackbar(trackbar_value, window_name , 0, max_value, nothing)
# Call the function to initialize
cv2.createTrackbar(trackbar_blur, window_name , 1, 20, nothing)
# create switch for ON/OFF functionality
color_switch = 'Color'
cv2.createTrackbar(color_switch, window_name,0,1,nothing)
cv2.createTrackbar('Contours', window_name,0,1,nothing)

while True:
	ret, frame = cam.read()

	if not ret:
		break

	#0: Binary
	#1: Binary Inverted
	#2: Threshold Truncated
	#3: Threshold to Zero
	#4: Threshold to Zero Inverted
	threshold_type = 1
	threshold_value = 0
	blur_value = 1
	blur_value = blur_value+ (  blur_value%2==0)
	isColor = 0
	findContours = 0

	###########################
	#    For skin calc        #
	###########################
	lower_HSV = np.array([0, 40, 0], dtype = "uint8")  
	upper_HSV = np.array([25, 255, 255], dtype = "uint8")  
  
	convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
	skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)  
  
	lower_YCrCb = np.array((0, 138, 67), dtype = "uint8")  
	upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  
      
	convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
	skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
  
	skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
  
	# blur the mask to help remove noise, then apply the  
	# mask to the frame  
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0) 
	skin = cv2.bitwise_and(frame, frame, mask = skinMask) 

	#########################################
	#    Convert to grayscale (start)       #
	#########################################
	if isColor == False:
		src_gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
		_, dst = cv2.threshold(src_gray, threshold_value, max_binary_value, threshold_type )
		blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
		if findContours:
			_, contours, hierarchy = cv2.findContours( blur, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE )
			blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  #add this line
			output = cv2.drawContours(blur, contours, -1, (0, 255, 0), 1)
			print(str(len(contours))+"\n")
		else:
			output = blur  
	else:
		_, dst = cv2.threshold(skin, threshold_value, max_binary_value, threshold_type )
		blur = cv2.GaussianBlur(dst,(blur_value,blur_value),0)
		output = blur
	#######################################
	#    Convert to grayscale (end)       #
	#######################################

	#######################################
	#    Component Analysis (start)       #
	#######################################
	# threshold and binarize the image
	gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  
	ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU ) 
	# colors different components in different shades of orange
	ret, markers, stats, centroids = cv2.connectedComponentsWithStats(output,ltype=cv2.CV_16U)  
	markers = np.array(markers, dtype=np.uint8)  
	label_hue = np.uint8(179*markers/np.max(markers))  
	blank_ch = 255*np.ones_like(label_hue)  
	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
	labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
	labeled_img[label_hue==0] = 0 

	#######################################
	#    Component Analysis (end)       #
	#######################################

	#visual = cv2.resize(labeled_img, (850, 480))
	cv2.imshow(window_name, labeled_img)
	k = cv2.waitKey(1) #k is the key pressed
	if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
		#exit
		cv2.destroyAllWindows()
		cam.release()
		break


