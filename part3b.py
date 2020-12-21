from __future__ import print_function
import cv2
import argparse
import numpy as np
import pyautogui
import time
from statistics import mean


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

# Initalizing time for simple gestures
timeSinceLastFive = time.time()
timeSinceLastLeftOne = time.time()
timeSinceLastRightOne = time.time()
timeSinceLastLeftTwo = time.time()
timeSinceLastRightTwo = time.time()

#Initializing time for complex gestures
timeLastTwoFingerSwipeRight = time.time()
timeLastTwoFingerSwipeLeft = time.time()
timeLastTwoFingerSwipeUp = time.time()
timeLastTwoFingerSwipeDown = time.time()
timeLastThreeFingerSwipeRight = time.time()
timeLastThreeFingerSwipeLeft = time.time()
timeLastThreeFingerSwipeUp = time.time()
timeLastThreeFingerSwipeDown = time.time()

def nothing(x):
    pass

# WEBCAM INPUT
cam = cv2.VideoCapture("handGreat.mov")
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
areaList = []
outputMessage = ''

# Colors
teal = [255, 243, 30]
green = [12, 175, 0]
red = [12, 49, 232]
purple = [255, 13, 164]

recentCommandList = []


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
            ffdlist = [] 
            hull = cv2.convexHull(largestContour, returnPoints = False)
            hullPoints = cv2.convexHull(largestContour)
            area = cv2.contourArea(hullPoints)    
            for cnt in contours[:1]:  
                defects = cv2.convexityDefects(cnt,hull)  
                if(not isinstance(defects,type(None))): 
                    fingerCount = 0 
                    for i in range(defects.shape[0]):  
                        # start point, end point, farthest point, approximate distance to farthest point
                        s,e,f,d = defects[i][0]  
                        start = tuple(cnt[s][0])  
                        end = tuple(cnt[e][0])  
                        far = tuple(cnt[f][0])
                        
                        

                        c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2  
                        a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2  
                        b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2  
                        angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                        
                        

                        # lets only look at fingertips
                        # this is where the distance between start and end points in current frame
                        # is greater than 200
                        if(c_squared>200):
                            radi = np.sqrt((start[0]-cX)**2 + (start[1]-cY)**2)
                            if angle <= np.pi / 2:  
                                fingerCount += 1  
                            ffdlist.append((start, fingerCount, radi, (cX, cY)))
            
        
            radilist = [item[2] for item in ffdlist]
            maxpos = radilist.index(max(radilist))

            cv2.circle(thresholdedHandImage, tuple(ffdlist[maxpos][0]), 25, teal, -100) #teal
            
            maxfarX, maxfarY = tuple(ffdlist[maxpos][0])
            
            cv2.line(thresholdedHandImage, (cX, cY), (maxfarX, maxfarY), green, 3)

            DifferenceX = cX - maxfarX
            DifferenceY = cY - maxfarY
            
            # So this will give us the one with the largest d BUT NOT THE CORRECT FINGER COUNT
            ffd = list(ffdlist[maxpos])

            # Then lets update this so that it reflects the highest finger count from the list which can be found in most recent list item
            ffd[1] = ffdlist[len(ffdlist)-1][1]

            

            if  max(radilist)>200 and DifferenceY > 0:
                fingerCount += 1
                ffd[1] += 1
                
            
            # Add the max ffd to the list of the last 50 frames 
            if len(farList) <= 30 or len(areaList)<= 50:
                farList.append(ffd)
                areaList.append(area)
            else:
                farList.pop(0)
                farList.append(ffd)
                areaList.pop(0)
                areaList.append(area)

            
            #######################################
            #    Gesture Recognition (start)      #
            ####################################### 
            mostRecentFingers = farList[len(farList)-1][1]
            mostRecentRadi= farList[len(farList)-1][2]
            
            lastFiveFrames = farList[len(farList)-6 : len(farList)-1]
            lastFiveFramesFingers = [item[1] for item in lastFiveFrames]

            # Area Calculation (for two and three finger swipe up and down)
            areaMax = max(areaList)
            areaMin = min(areaList)
            areaMaxPos = areaList.index(areaMax)
            areaMinPos = areaList.index(areaMin)
            AreaDifference = areaMax - areaMin
            AreaDifferencePos = areaMaxPos - areaMinPos
            
            # FARPOINT
            farPointList = [item[0] for item in farList]

            farPointListX = [item[0] for item in farPointList]
            farPointMaxX = max(farPointListX)
            farPointMinX = min(farPointListX)
            farPointXDifference = farPointMaxX - farPointMinX

            farPointListY = [item[1] for item in farPointList]
            farPointMaxY = max(farPointListY)
            farPointMinY = min(farPointListY)
            farPointYDifference = farPointMaxY - farPointMinY
            minimumYFarPointDifference = .75*(mostRecentRadi)

            # CENTROID
            centroidList = [item[3] for item in farList]

            centroidListX = [item[0] for item in centroidList]
            centroidMaxX = max(centroidListX)
            centroidMinX = min(centroidListX)
            centroidMaxXPos = centroidListX.index(centroidMaxX)
            centroidMinXPos = centroidListX.index(centroidMinX)
            centroidXDifference = centroidMaxX - centroidMinX
            centroidXDifferencePos = centroidMaxXPos - centroidMinXPos

            centroidListY = [item[1] for item in centroidList]
            centroidMaxY = max(centroidListY)
            centroidMinY = min(centroidListY)
            centroidYDifference = centroidMaxY - centroidMinY
            minimumYDifference = .85*(mostRecentRadi)
            
            minimumHorizontalDistance = .4*mostRecentRadi
            minimumVerticalDistance = .15*mostRecentRadi
            currentTime = time.time()

            
            # 3 static/simple gestures
            #       1. 5 fingers open: pause/play
            #       2. 1 finger pointing left/right: skip back/forward 5 seconds respectively
            #       3. 2 finger pointing left/right: skip back/forward 10 seconds respectively
            
            # 5 fingers open: pause/play
            if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 5: # and countLastTenFramesFingers < 6 and countLastFiveFramesFingers >= 3:
                if currentTime - timeSinceLastFive > 1:
                    outputMessage = str(fingerCount)
                    print("pause triggered!")
                    pyautogui.press('space')
                    currentCommand = ("pause")
                timeSinceLastFive = currentTime

            # Pointing right
            elif DifferenceX < (-0.4*mostRecentRadi) \
                 and centroidXDifference < 0.5*mostRecentRadi \
                 and (mostRecentFingers == 1 or mostRecentFingers ==2): # and farList[len(farList)-1][0] not in range(DifferenceX-1000, DifferenceX+1000): #pointing right
                # Pointing right 1 finger
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 1:
                    currentCommand = ("right 1")
                    if recentCommandList.count(currentCommand) > 8:
                        print(recentCommandList.count(currentCommand))
                        if currentTime - timeSinceLastRightOne > 2:
                            pyautogui.press('right')
                            time.sleep(1)
                            pyautogui.press('space')
                            print("Skip forward one frame, sleep for one second, continue")
                        outputMessage = "right 1"
                        timeSinceLastRightOne = currentTime
                # Pointing right 2 fingers
                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    currentCommand = ("right 2")
                    if recentCommandList.count(currentCommand) > 8:
                        print(currentCommand, recentCommandList.count(currentCommand))
                        if currentTime - timeSinceLastRightTwo > 2:
                            pyautogui.press('right', presses = 10)
                            time.sleep(1)
                            pyautogui.press('space')
                            print("Skip forward 10 frames")
                        outputMessage = "right 2"
                        timeSinceLastRightTwo = currentTime
                    
            # Pointing Left
            elif DifferenceX > .4*mostRecentRadi \
                 and centroidXDifference < 0.5*mostRecentRadi: 
                # Pointing left one finger
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 1:
                    currentCommand = ("left 1")
                    if recentCommandList.count(currentCommand) > 8:
                        outputMessage = "left 1"
                        timeSinceLastLeftOne = currentTime
                        if currentTime - timeSinceLastLeftOne > 2:
                            pyautogui.press('left')
                            time.sleep(1)
                            pyautogui.press('space')
                            print("Go back one frame, sleep for one second, continue")
                # Pointing left two fingers
                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    currentCommand = ("left 2")
                    if recentCommandList.count(currentCommand) > 8:
                        outputMessage = "left 2"
                        timeSinceLastLeftTwo = currentTime
                        if currentTime - timeSinceLastLeftTwo > 2:
                            pyautogui.press('left', presses = 10)
                            time.sleep(1)
                            pyautogui.press('space')
                            print("left 2 origin")
                        
            
            # 4 dynamic gestures
            #       1. Swipe left:
            #           1a. Two fingers
            #           1b. Three Fingers:
            #       2. Swipe right:
            #           2a. Two fingers
            #           2b. Three Fingers:
            #       3. Swipe up:
            #           3a. Two fingers
            #           3b. Three Fingers:
            #       4. Swipe down:
            #           4a. Two fingers
            #           4b. Three Fingers:
            
            # Swipe Left
            elif centroidXDifference > minimumHorizontalDistance \
                 and farPointXDifference > minimumHorizontalDistance \
                 and abs(centroidYDifference) < .25*mostRecentRadi \
                 and centroidXDifferencePos < 0:
                # With two fingers 
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    if currentTime - timeLastTwoFingerSwipeRight > 2:
                        outputMessage = ("two finger swipe left")
                        pyautogui.press('left', presses=30)
                    timeLastTwoFingerSwipeRight = currentTime
                # With three fingers 
                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
                    timeLastThreeFingerSwipeRight = currentTime
                    if currentTime - timeLastThreeFingerSwipeRight > 2:
                        outputMessage = ("three finger swipe left")
                        pyautogui.keydown('option')
                        pyautogui.press('left')
                        pyautogui.keyup('option')                    

            # Swipe right
            elif centroidXDifference > minimumHorizontalDistance \
                 and farPointXDifference > minimumHorizontalDistance \
                 and abs(centroidYDifference) < (.25*mostRecentRadi) \
                 and centroidXDifferencePos > 0:
                # With two fingers
                if currentTime - timeLastTwoFingerSwipeLeft > 2:
                    if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                        outputMessage = ("two finger swipe right")
                        pyautogui.press('right', presses=30)
                    timeLastTwoFingerSwipeLeft = currentTime 
                # With three fingers
                elif currentTime - timeLastThreeFingerSwipeLeft > 2:
                    if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
                        outputMessage = ("three finger swipe right")
                        #skip to end of video
                        pyautogui.keydown('option')
                        pyautogui.press('right')
                        pyautogui.keyup('option')
                    timeLastThreeFingerSwipeLeft = currentTime
            
            #Swipe down and up
            elif AreaDifference > 40000:
                #Swipe Up
                if AreaDifferencePos > 0:
                    # Swipe up two fingers
                    if currentTime - timeLastTwoFingerSwipeUp > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                            outputMessage = ("two finger swipe up")
                            pyautogui.press('volumeup')
                        timeLastTwoFingerSwipeUp = currentTime 
                    # Swipe up three fingers
                    elif currentTime - timeLastThreeFingerSwipeUp > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 \
                           and mostRecentFingers == 3 \
                           and AreaDifference > 70000:
                            outputMessage = ("three finger swipe up")
                            pyautogui.press('up', presses = 5)
                        timeLastThreeFingerSwipeUp = currentTime
                #Swipe Down
                elif AreaDifferencePos < 0:
                    # Swipe down two fingers
                    if currentTime - timeLastTwoFingerSwipeDown > 2:
                        timeLastTwoFingerSwipeDown = currentTime 
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                            outputMessage = ("two finger swipe down")
                            pyautogui.press('down')
                    # Swipe down three fingers
                    elif currentTime - timeLastThreeFingerSwipeDown > 2:
                        timeLastThreeFingerSwipeDown = currentTime
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 \
                           and mostRecentFingers == 3 \
                           and AreaDifference > 70000:
                            outputMessage = ("three finger swipe down")
                            pyautogui.press('up', presses = 5)
                        
            # Zero Fingers Fist Edge Case
            elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 0:
                currentCommand = (str(mostRecentFingers))
                outputMessage = (str(mostRecentFingers))
            else: 
                currentCommand = str(mostRecentFingers)
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4:
                    currentCommand = (str(mostRecentFingers))
                    outputMessage = (str(mostRecentFingers))


            if len(recentCommandList) < 10:
                recentCommandList.append(currentCommand)
            else: 
                recentCommandList.pop(0)
                recentCommandList.append(currentCommand)


            cv2.putText(thresholdedHandImage, outputMessage, 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
            visual = cv2.resize(thresholdedHandImage, (850, 480))
            cv2.imshow(window_name, visual)
            # Do not delete the following line! Or else the visualization will not show up at all.
            key = cv2.waitKey(100)
        #######################################
        #    Gesture Recognition (end)        #
        #######################################
    except Exception as e: print(e)
    