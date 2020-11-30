from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui

max_value = 255
max_type = 4
max_binary_value = 255
trackbar_type = 'Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted'
trackbar_value = 'Value'
trackbar_blur = 'Blur kernel size'
window_name = 'Hand Extract'
isColor = False

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 3
fontColor              = (255,255,255)
lineType               = 2

def nothing(x):
	pass

# WEBCAM INPUT
cam = cv2.VideoCapture("handMovieBest.mov")
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

# Will hold a triple containing ((x,y), numFingers) 
# (x, y) will bee the coordinate of the farthest point
# numFingers is number of fingers held up
farList = []
outputMessage = ''

# Colors
teal = [255, 243, 30]
green = [12, 175, 0]
red = [12, 49, 232]
purple = [255, 13, 164]


while True:
	# WEBCAM INPUT
	ret, frame = cam.read()

	if not ret:
		break

	# IMAGE INPUT
	#frame = cv2.imread("hand-sample.jpg", 1)

	#0: Binary
	#1: Binary Inverted
	#2: Threshold Truncated
	#3: Threshold to Zero
	#4: Threshold to Zero Inverted
	# threshold_type = 1
	# threshold_value = 0
	# blur_value = 1
	# blur_value = blur_value+ (  blur_value%2==0)
	# isColor = 0
	# findContours = 0
	threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
	threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
	blur_value = cv2.getTrackbarPos(trackbar_blur, window_name)
	blur_value = blur_value+ (  blur_value%2==0)
	isColor = (cv2.getTrackbarPos(color_switch, window_name) == 1)
	findContours = (cv2.getTrackbarPos('Contours', window_name) == 1)

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
			#print(str(len(contours))+"\n")
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
	#gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)  
	#PART TWO RET
	#ret, thresh = cv2.threshold(gray, 0, max_binary_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU ) 
	ret, thresh = cv2.threshold(output, 0, max_binary_value, cv2.THRESH_BINARY)
	#do this for part 3
	#subImg = thresh
	#######################################
	#    Component Analysis (end)       #
	#######################################

	##################################
	#    3a Hand Image (start)       #
	##################################
	#subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)
	thresholdedHandImage = thresh

	_, contours, _ = cv2.findContours(thresholdedHandImage.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
	contours=sorted(contours,key=cv2.contourArea,reverse=True) 
	thresholdedHandImage = cv2.cvtColor(thresholdedHandImage, cv2.COLOR_GRAY2RGB)      
	spacePressed = False
	count = 0
	try:
		if len(contours)>1:  
			largestContour = contours[0]
			M = cv2.moments(largestContour)  
			cX = int(M["m10"] / M["m00"])  
			cY = int(M["m01"] / M["m00"]) 
			#print("M = ", str(M))
			#print("cX = ", str(cX))
			#print("cY = ", str(cY))
			# dlist d for each point 
			# dlist = []
			# fflist holds the tuple for the (far, fingers)
			ffdlist = [] 
			hull = cv2.convexHull(largestContour, returnPoints = False)     
			for cnt in contours[:1]:  
				defects = cv2.convexityDefects(cnt,hull)  
				#dlist = []
				if(not isinstance(defects,type(None))): 
					fingerCount = 0 
					for i in range(defects.shape[0]):  
						# start point, end point, farthest point, approximate distance to farthest point
						s,e,f,d = defects[i][0]  
						start = tuple(cnt[s][0])  
						end = tuple(cnt[e][0])  
						far = tuple(cnt[f][0])
						
						#print("far = ", str(far))
						#print("start = ", str(start))
						#print("end = ", str(end))

						c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
						a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
						b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
						angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
						
						

						# lets only look at fingertips
						# this is where the distance between start and end points in current frame
						# is greater than 200
						if(c_squared>200):
							# print("---------------------------------------")
							# print(c_squared)
							# print(a_squared)
							# print(b_squared)
							# print(str(angle))
							#cv2.circle(thresholdedHandImage, far, 10, teal, -5) #teal
							#cv2.line(thresholdedHandImage, start, (cX, cY), green,2)
							#cv2.circle(thresholdedHandImage, start, 10, green, -5) #green
							#cv2.circle(thresholdedHandImage, end, 10, red, -5) #red
							#cv2.circle(thresholdedHandImage, (cX, cY), 10, green, -5) #purple
							# calculate distance between (cX, cY) and start
							radi = np.sqrt((start[0]-cX)**2 + (start[1]-cY)**2)
							#cv2.line(thresholdedHandImage,(cX,cY), far,[0,255,0],2)
							#print(radi)

							#print("c_squared = ", str(c_squared))
							#print("angle = ", str(angle))

							#dlist.append(d)
							

							#print("far is : ", far)

							if angle <= np.pi / 2.5:  
								fingerCount += 1  
								#print("far = ", str(far))
								#print("start = ", str(start))
								#print("end = ", str(end))
							ffdlist.append((start, fingerCount, radi, (cX, cY)))
							#print(ffdlist)
							
							key = cv2.waitKey(1000)
							#clear up colors for next frame visualization
							#cv2.line(thresholdedHandImage,start,end,[255,255,255],2)
							#cv2.circle(thresholdedHandImage, far, 10, [0, 0, 0], -5) 
							#cv2.line(thresholdedHandImage,start,end,[0, 0, 0],2)
							#cv2.line(thresholdedHandImage, start, (cX, cY), [255,255,255],2)
							#cv2.circle(thresholdedHandImage, start, 10, [0, 0, 0], -5) 
							#cv2.circle(thresholdedHandImage, end, 10, [0, 0, 0], -5) 
							#visual = cv2.resize(thresholdedHandImage, (850, 480))
							#print("max d detected: " , maxd)
							#resize for image input
							#visual = cv2.resize(visual, (426, 512))
							#print(ffdlist)
							#cv2.imshow(window_name, visual)
					#print("next segment!--------------------------")
			#print(dlist)
			
			#print(min(dlist))
			# inbuilt function to find the position of maximum 
			radilist = [item[2] for item in ffdlist]
			maxpos = radilist.index(max(radilist))
			#print(maxd)

			cv2.circle(thresholdedHandImage, tuple(ffdlist[maxpos][0]), 25, [86, 100, 222], -100) #teal
			
			maxfarX, maxfarY = tuple(ffdlist[maxpos][0])
			
			cv2.line(thresholdedHandImage, (cX, cY), (maxfarX, maxfarY), [0, 0, 255], 3)

			DifferenceX = cX - maxfarX
			print("DifferenceX = ", str(DifferenceX))
			
			# So this will give us the one with the largest d BUT NOT THE CORRECT FINGER COUNT
			ffd = list(ffdlist[maxpos])

			# Then lets update this so that it reflects the highest finger count from the list which can be found in most recent list item
			ffd[1] = ffdlist[len(ffdlist)-1][1]

			#print(ffdlist)
			#print(str(ffd))

			if  max(radilist) >300:
				#print("insider fingerlist")
				fingerCount += 1
				ffd[1] += 1
			
			#print(maxd)
			#maxf = max(flist)
			# Add the max ffd to the list of the last 50 frames 
			if len(farList) <= 50:
				farList.append(ffd)
			else:
				farList.pop(0)
				farList.append(ffd)
				#print(farList[0])
				#print(farList[49])

			outputMessage = str(fingerCount)

			print(ffd)
			# cv2.putText(thresholdedHandImage, outputMessage, 
			# 				bottomLeftCornerOfText, 
			# 				font, 
			# 				fontScale,
			# 				fontColor,
			# 				lineType)
			# visual = cv2.resize(thresholdedHandImage, (850, 480))
			# 	#print("max d detected: " , maxd)
			# 	#resize for image input
			# 	#visual = cv2.resize(visual, (426, 512))
			# cv2.imshow(window_name, visual)

			# key = cv2.waitKey(10000)

			#DifferenceX = cX - maxf[0]
			
			#else do nothing
			#######################################
			#    Gesture Recognition (start)      #
			####################################### 
			# GESTURE RECOGNITION/ACTION for youtube media player
			# 3 static/simple gestures
			#		1. 5 fingers open: pause/play
			#		2. 1 finger pointing left/right: skip back/forward 5 seconds respectively
			#		3. 2 finger pointing left/right: skip back/forward 10 seconds respectively
			# print(DifferenceX)
			# # Static Gesture 1 of 3: 5 fingers open: pause/play
			# if fingerCount is 5: # and farList[len(farList)-2][1] != 5:
			# 	outputMessage = str(fingerCount)
			# 	print("pause triggered!")
			# 	#pyautogui.press('space')
			
			# # Static Gesture 2 and 3: 1 finger pointing left/right: skip back/forward 5 seconds; 2 fingers pointing left and right skip back and forward
			# #alternatively can be done using the angle from Cx, Cy to far
			# elif DifferenceX > (maxd-1000): # and farList[len(farList)-1][0] not in range(DifferenceX-1000, DifferenceX+1000): #pointing right
			# 	if fingerCount == 1:
			# 		#pyautogui.press('right')
			# 		outputMessage = "right 1"
			# 	elif fingerCount == 2:
			# 		#pyautogui.press('l')
			# 		outputMessage = "right 2"

			# elif DifferenceX < (maxd-1000): #pointing left
			# 	if fingerCount == 1:
			# 		#pyautogui.press('left')
			# 		outputMessage = "left 1"
			# 	elif fingerCount == 2:
			# 		#pyautogui.press('j')
			# 		outputMessage = "left 2"
			# else:
			# 	outputMessage = str(fingerCount)

			# if maxd < 100000:
			# 	cv2.putText(thresholdedHandImage, outputMessage, 
			# 				bottomLeftCornerOfText, 
			# 				font, 
			# 				fontScale,
			# 				fontColor,
			# 				lineType)
			# 	visual = cv2.resize(thresholdedHandImage, (850, 480))
			# 	#print("max d detected: " , maxd)
			# 	#resize for image input
			# 	#visual = cv2.resize(visual, (426, 512))
			# 	cv2.imshow(window_name, visual)
			# else:
			# 	print("FRAME REJECTED")
	except:
		pass

		# 4 dynamic/complex gestures
		#		1. 2 finger swipe left/right: speed down/up video
		#		2. 2 finger up/down: volume up/down
		#		3. pinch to zoom in and out
		#		4. wave bye: close the window
		# Using youtube media player
		# SIMPLE GESTURES
		# if fingerCount is 5:
		#     pyautogui.press('space')
		#     spacePressed = !spacePressed
		# # where is the far point?
		# DifferenceX = Cx - maxf[0]
		# if fingerCount is 2 and DifferenceX>0: #Fix to make something other than zero
		#     pyautogui.press('right')
		# if fingerCount is 2 and DifferenceX<0:
		#     pyautogui.press('left')
		# COMPLEX GESTURES
		# Swipe up with one finger (volume up)

		# Swipe down with one finger (volume down)
		#######################################
		#    Gesture Recognition (end)        #
		#######################################

					#cv2.putText(thresholdedHandImage, fingerCount, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)
					#cv2.line(thresholdedHandImage,start,end,[0,255,0],2)  
					#cv2.circle(thresholdedHandImage,far,5,[0,0,255],-1)

	#thresholdedHandImage = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
	# _, contours, _ = cv2.findContours(thresholdedHandImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
	# contours=sorted(contours,key=cv2.contourArea,reverse=True)       
	# if len(contours)>1:  
	#     largestContour = contours[0]  
	#     hull = cv2.convexHull(largestContour, returnPoints = False)     
	#     for cnt in contours[:1]:  
	#         defects = cv2.convexityDefects(largestContour,hull)  
	#         if(not isinstance(defects,type(None))):  
	#             for i in range(defects.shape[0]):  
	#                 s,e,f,d = defects[i,0]  
	#                 start = tuple(cnt[s][0])  
	#                 end = tuple(cnt[e][0])  
	#                 far = tuple(cnt[f][0])  
					
	#                 #thresholdedHandImage = cv2.cvtColor(thresholdedHandImage, cv2.COLOR_GRAY2RGB)
	#                 cv2.line(thresholdedHandImage,start,end,[0,255,0],2)  
	#                 cv2.circle(thresholdedHandImage,far,20,[0,0,255],-1)
		

	# _, contours, _ = cv2.findContours(thresholdedHandImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
	# contours=sorted(contours,key=cv2.contourArea,reverse=True)       
	# if len(contours)>1:  
	#     largestContour = contours[0]  
	#     hull = cv2.convexHull(largestContour, returnPoints = False)    
	#     fingerCount = 0 
	#     for cnt in contours[:1]: 
	#         #print(cnt) 
	#         defects = cv2.convexityDefects(cnt,hull)  
	#         if(not isinstance(defects,type(None))): 
	#             #print(defects.shape) 
	#             for i in range(defects.shape[0]):  
	#                 #print(i)
	#                 s,e,f,d = defects[i,0]  
	#                 start = tuple(cnt[s][0])  
	#                 end = tuple(cnt[e][0])  
	#                 far = tuple(cnt[f][0])  

	#                 # c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
	#                 # a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
	#                 # b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
	#                 # angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
					  
	#                 # if angle <= np.pi / 3:  
	#                 #     fingerCount += 1
	#                 #     subImg = cv2.cvtColor(thresholdedHandImage, cv2.COLOR_GRAY2RGB);  
	#                     #cv2.circle(subImg, far, 4, (0, 0, 255), -1)
	#                 colorIndicators = cv2.cvtColor(thresholdedHandImage, cv2.COLOR_GRAY2RGB);
	#                 cv2.line(colorIndicators,start,end,(0,255,0),2)  
	#                 cv2.circle(colorIndicators,far,5,(0,0,255),-1)
	#                 visual = cv2.resize(colorIndicators, (850, 480))
	#                 cv2.imshow(window_name, visual)
	################################
	#    3a Hand Image (end)       #
	################################
	#########################################
	#    3b Detecting Fingers (start)       #
	#########################################
	# s,e,f,d = defects[i,0]  
	# start = tuple(cnt[s][0])  
	# end = tuple(cnt[e][0])  
	# far = tuple(cnt[f][0])  
		
	# c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
	# a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
	# b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
	# angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
					  
	# if angle <= np.pi / 3:  
	#     fingerCount += 1  
	#     cv2.circle(img4, far, 4, [0, 0, 255], -1)
	#######################################
	#    3b Detecting Fingers (end)       #
	#######################################


	
	k = cv2.waitKey(1) #k is the key pressed
	if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
		#exit
		cv2.destroyAllWindows()
		cam.release()
		break


