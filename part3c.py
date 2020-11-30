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

# arguments to create the legend  
font = cv2.FONT_HERSHEY_SIMPLEX 
org = (00, 185) 
fontScale = 1
color = (0, 0, 255)  
thickness = 2

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
    if len(contours)>1:  
        largestContour = contours[0]  
        hull = cv2.convexHull(largestContour, returnPoints = False) 
        fingerCount = 0    
        for cnt in contours[:1]:  
            defects = cv2.convexityDefects(cnt,hull)  
            if(not isinstance(defects,type(None))):  
                for i in range(defects.shape[0]):  
                    s,e,f,d = defects[i,0]  
                    start = tuple(cnt[s][0])  
                    end = tuple(cnt[e][0])  
                    far = tuple(cnt[f][0])
        
                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                      
                    if angle <= np.pi / 3:  
                        fingerCount += 1  
                        cv2.circle(img4, far, 4, [0, 0, 255], -1)
  
                    cv2.line(thresholdedHandImage,start,end,[0,255,0],2)  
                    cv2.circle(thresholdedHandImage,far,5,[0,0,255],-1)

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
        visual = cv2.resize(thresholdedHandImage, (850, 480))
        #output = cv2.cvtColor(thresholdedHandImage, cv2.COLOR_GRAY2RGB)
        #visual = thresholdedHandImage
        visual = cv2.resize(visual, (426, 512))
        cv2.imshow(window_name, visual)


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


