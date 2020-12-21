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
            hullPoints = cv2.convexHull(largestContour)
            area = cv2.contourArea(hullPoints)
            #areaList.append(area)     
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

                            if angle <= np.pi / 2:  
                                fingerCount += 1  
                                #print("far = ", str(far))
                                #print("start = ", str(start))
                                #print("end = ", str(end))
                            ffdlist.append((start, fingerCount, radi, (cX, cY)))
                            #print(ffdlist)
                            
                            #key = cv2.waitKey(1000)
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
            #print(ffdlist[maxpos][3])

            cv2.circle(thresholdedHandImage, tuple(ffdlist[maxpos][0]), 25, teal, -100) #teal
            
            maxfarX, maxfarY = tuple(ffdlist[maxpos][0])
            
            cv2.line(thresholdedHandImage, (cX, cY), (maxfarX, maxfarY), green, 3)

            DifferenceX = cX - maxfarX
            DifferenceY = cY - maxfarY
            #print("DifferenceY = ", str(DifferenceY))
            
            # So this will give us the one with the largest d BUT NOT THE CORRECT FINGER COUNT
            ffd = list(ffdlist[maxpos])

            # Then lets update this so that it reflects the highest finger count from the list which can be found in most recent list item
            ffd[1] = ffdlist[len(ffdlist)-1][1]

            #print(ffdlist)
            #print(str(ffd))

            if  max(radilist)>200 and DifferenceY > 0:
                #print("insider fingerlist")
                fingerCount += 1
                ffd[1] += 1
                #print(fingerCount)
            
            #print(maxd)
            #maxf = max(flist)
            # Add the max ffd to the list of the last 50 frames 
            if len(farList) <= 30 or len(areaList)<= 50:
                farList.append(ffd)
                areaList.append(area)
            else:
                farList.pop(0)
                farList.append(ffd)
                areaList.pop(0)
                areaList.append(area)
                #print(farList[0])
                #print(farList[49])

            #print(areaList)
            #outputMessage = str(fingerCount)

            #print(ffd)
            # cv2.putText(thresholdedHandImage, outputMessage, 
            #               bottomLeftCornerOfText, 
            #               font, 
            #               fontScale,
            #               fontColor,
            #               lineType)
            # visual = cv2.resize(thresholdedHandImage, (850, 480))
            # #   print("max d detected: " , maxd)
            #   #resize for image input
            #   #visual = cv2.resize(visual, (426, 512))
            # cv2.imshow(window_name, visual)

            # # Do not delete the following line! Or else the visualization will not show up at all.
            # key = cv2.waitKey(100)

            #DifferenceX = cX - maxf[0]
            
            #else do nothing
            #######################################
            #    Gesture Recognition (start)      #
            ####################################### 
            # GESTURE RECOGNITION/ACTION for youtube media player
            # 3 static/simple gestures
            #       1. 5 fingers open: pause/play
            #       2. 1 finger pointing left/right: skip back/forward 5 seconds respectively
            #       3. 2 finger pointing left/right: skip back/forward 10 seconds respectively
            # print(DifferenceX)
            # Check the number of fingers in the last 15 frames
            # We want to see that the the number of fingers presented shows up for the last five frames 
            # at a minimum (to ensure mistakes aren't counted)
            # But we want to ignore if the number of fingers shows up for most of last 10 frames before the 5
            mostRecentFingers = farList[len(farList)-1][1]
            mostRecentRadi= farList[len(farList)-1][2]
            #print(mostRecentFingers)
            #lastTenFrames = farList[len(farList)-16 : len(farList)-6]
            #lastTenFramesFingers = [item[1] for item in lastTenFrames]
            #countLastTenFramesFingers = lastTenFramesFingers.count(mostRecentFingers)
            lastFiveFrames = farList[len(farList)-6 : len(farList)-1]
            lastFiveFramesFingers = [item[1] for item in lastFiveFrames]

            areaMax = max(areaList)
            areaMin = min(areaList)
            areaMaxPos = areaList.index(areaMax)
            areaMinPos = areaList.index(areaMin)
            AreaDifference = areaMax - areaMin
            AreaDifferencePos = areaMaxPos - areaMinPos
            #print("num fingers = ", mostRecentFingers, (AreaDifference, AreaDifferencePos, areaMin, areaMax))

            #FARPOINT
            farPointList = [item[0] for item in farList]
            #print("ffdlist len = ", len(ffdlist))
            #print(len(farPointList))

            farPointListX = [item[0] for item in farPointList]
            #print(farPointListX)
            farPointMaxX = max(farPointListX)
            farPointMinX = min(farPointListX)
            # farPointMaxXPos = farPointListX.index(farPointMaxX)
            # farPointMinXPos = farPointListX.index(farPointMinX)
            farPointXDifference = farPointMaxX - farPointMinX
            # farPointXDifferencePos = farPointMaxXPos - farPointMinXPos
            # print(farPointXDifferencePos)
            #print(farPointXDifferencePos)

            farPointListY = [item[1] for item in farPointList]
            farPointMaxY = max(farPointListY)
            farPointMinY = min(farPointListY)
            # farPointMaxYPos = farPointListY.index(farPointMaxY)
            # farPointMinYPos = farPointListY.index(farPointMinY)
            farPointYDifference = farPointMaxY - farPointMinY
            minimumYFarPointDifference = .75*(mostRecentRadi)

            #THIS DOES NOT REFER TO THE CENTROID ANYMORE IT REFERS TO THE FARPOINT
            centroidList = [item[3] for item in farList]
            #print("ffdlist len = ", len(ffdlist))
            #print(len(centroidList))

            centroidListX = [item[0] for item in centroidList]
            #print(centroidListX)
            centroidMaxX = max(centroidListX)
            centroidMinX = min(centroidListX)
            centroidMaxXPos = centroidListX.index(centroidMaxX)
            centroidMinXPos = centroidListX.index(centroidMinX)
            centroidXDifference = centroidMaxX - centroidMinX
            centroidXDifferencePos = centroidMaxXPos - centroidMinXPos
            # print(centroidXDifferencePos)
            #print(centroidXDifferencePos)

            centroidListY = [item[1] for item in centroidList]
            centroidMaxY = max(centroidListY)
            centroidMinY = min(centroidListY)
            # centroidMaxYPos = centroidListY.index(centroidMaxY)
            # centroidMinYPos = centroidListY.index(centroidMinY)
            centroidYDifference = centroidMaxY - centroidMinY
            minimumYDifference = .85*(mostRecentRadi)
            # centroidYDifferencePos = centroidMaxYPos - centroidMinYPos
            # # # Static Gesture 2 and 3: 1 finger pointing left/right: skip back/forward 5 seconds; 2 fingers pointing left and right skip back and forward
            # # #alternatively can be done using the angle from Cx, Cy to far
            #countLastFiveFramesFingers = lastFiveFramesFingers.count(mostRecentFingers)
            #lastFifteenFramesFingers [item[1] for item in lastFifteenFrames]
            # print("countLastTenFramesFingers")
            # print(countLastTenFramesFingers)
            # print("countLastFiveFramesFingers")
            # print(countLastFiveFramesFingers)
            # Static Gesture 1 of 3: 5 fingers open: pause/play
            # Need at least 1 whole second since last time conditional was reached 
            minimumHorizontalDistance = .4*mostRecentRadi
            minimumVerticalDistance = .15*mostRecentRadi
            currentTime = time.time()

            #print("centroidXDifference = ", str(centroidXDifference))# , " with position diff of ", centroidXDifferencePos)
            #print("centroidXDifferencePos = ", str(centroidXDifferencePos))
            #print("minimumHorizontalDistance = ", minimumHorizontalDistance)
            #print("centroidYDifference = ", str(centroidYDifference)) #, " with position diff of ", centroidYDifferencePos)
            #print("farPointXDifference = ", str(farPointXDifference))# , " with position diff of ", centroidXDifferencePos)
            #print("farPointYDifference = ", str(farPointYDifference)) #, " with position diff of ", centroidYDifferencePos)
            #print(mostRecentFingers)
            #print("DifferenceX = ", DifferenceX)
            #print("mostRecentRadi = ", mostRecentRadi)

            if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 5: # and countLastTenFramesFingers < 6 and countLastFiveFramesFingers >= 3:
                if currentTime - timeSinceLastFive > 1:
                    outputMessage = str(fingerCount)
                    print("pause triggered!")
                    pyautogui.press('space')
                    currentCommand = ("pause")
                    # recentCommandList.append(currentCommand)
                timeSinceLastFive = currentTime
                #print("at the 5")
                #pyautogui.press('space')
            elif DifferenceX < (-0.4*mostRecentRadi) \
                 and centroidXDifference < 0.5*mostRecentRadi \
                 and (mostRecentFingers == 1 or mostRecentFingers ==2): # and farList[len(farList)-1][0] not in range(DifferenceX-1000, DifferenceX+1000): #pointing right
                 #and DifferenceY < minimumYDifference \
                #print("on the right size")
                #print(mean(lastFiveFramesFingers))
                # print("in 2")
                # print(-.4*mostRecentRadi)
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 1:
                    #print('rightside 1')
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
                    # recentCommandList.append(currentCommand)

                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    currentCommand = ("right 2")
                    # print("rightside 2")
                    if recentCommandList.count(currentCommand) > 8:
                        print(currentCommand, recentCommandList.count(currentCommand))
                        if currentTime - timeSinceLastRightTwo > 2:
                            pyautogui.press('right', presses = 10)
                            time.sleep(1)
                            pyautogui.press('space')
                            print("Skip forward 10 frames")
                        outputMessage = "right 2"
                        timeSinceLastRightTwo = currentTime
                    # recentCommandList.append(currentCommand)

            elif DifferenceX > .4*mostRecentRadi \
                 and centroidXDifference < 0.5*mostRecentRadi: # and farList[len(farList)-1][0] not in range(DifferenceX-1000, DifferenceX+1000): #pointing right
                 #and DifferenceY < minimumYDifference \
                # print("in 3")
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 1:
                    currentCommand = ("left 1")
                    if recentCommandList.count(currentCommand) > 8:
                        print(recentCommandList.count(currentCommand))
                        if currentTime - timeSinceLastLeftOne > 2:
                            pyautogui.press('left')
                            time.sleep(1)
                            pyautogui.press('space')
                            print("Go back one frame, sleep for one second, continue")
                        outputMessage = "left 1"
                        timeSinceLastLeftOne = currentTime
                    
                    # recentCommandList.append(currentCommand)
                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    currentCommand = ("left 2")
                    if recentCommandList.count(currentCommand) > 8:
                        print(recentCommandList.count(currentCommand))
                        if currentTime - timeSinceLastLeftTwo > 2:
                            pyautogui.press('left', presses = 10)
                            time.sleep(1)
                            pyautogui.press('space')
                            print("left 2 origin")
                        outputMessage = "left 2"
                        timeSinceLastLeftTwo = currentTime
                    # recentCommandList.append(currentCommand)
            
            # 4 dynamic/complex gestures
            #       1. 2 finger swipe left/right: speed down/up video
            #       2. 2 finger up/down: volume up/down
            #       3. pinch to zoom in and out
            #       4. wave bye: close the window
            
            # Setup for the complex gestures
            # Find the extremes over the last 50 frames
            # ffdlist.append((start, fingerCount, radi, (cX, cY)))
            # radilist = [item[2] for item in ffdlist]
            # maxpos = radilist.index(max(radilist))

            

            
            # this means it has moved to the right
            elif centroidXDifference > minimumHorizontalDistance \
                 and farPointXDifference > minimumHorizontalDistance \
                 and abs(centroidYDifference) < .25*mostRecentRadi \
                 and centroidXDifferencePos < 0: 

                # print("centroidXDifference = ", str(centroidXDifference))# , " with position diff of ", centroidXDifferencePos)
                # print("centroidYDifference = ", str(centroidYDifference)) #, " with position diff of ", centroidYDifferencePos)
                # print("farPointXDifference = ", str(farPointXDifference))# , " with position diff of ", centroidXDifferencePos)
                # print("farPointYDifference = ", str(farPointYDifference)) #, "
                #print("in left")# \
                 # and DifferenceX < 100 \
                 # and DifferenceY > minimumYDifference: #and centroidXDifferencePos > 0
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                    if currentTime - timeLastTwoFingerSwipeRight > 2:
                        outputMessage = ("two finger swipe left")
                        pyautogui.press('left', presses=30)
                    timeLastTwoFingerSwipeRight = currentTime 
                elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
                    if currentTime - timeLastThreeFingerSwipeRight > 2:
                        outputMessage = ("three finger swipe left")
                        pyautogui.keydown('option')
                        pyautogui.press('left')
                        pyautogui.keyup('option')
                    timeLastThreeFingerSwipeRight = currentTime

            # TWO AND THREE FINGER SWIPE LEFT
            # Check if both the centroid and far point have moved the minimumHorizontalDifference or more to the left (less than negative minimum horizontal difference)
            elif centroidXDifference > minimumHorizontalDistance \
                 and farPointXDifference > minimumHorizontalDistance \
                 and abs(centroidYDifference) < (.15*mostRecentRadi) \
                 and centroidXDifferencePos > 0:
                yay = 1
                # print("centroidXDifference = ", str(centroidXDifference)) # , " with position diff of ", centroidXDifferencePos)
                # print("centroidYDifference = ", str(centroidYDifference)) #, " with position diff of ", centroidYDifferencePos)
                # #print("farPointXDifference = ", str(farPointXDifference)) # , " with position diff of ", centroidXDifferencePos)
                # #print("farPointYDifference = ", str(farPointYDifference)) #, "
                # print("in right") # \
                 # and DifferenceX > -100 \
                 # and DifferenceY > minimumYDifference: #and centroidXDifferencePos < 0
                if currentTime - timeLastTwoFingerSwipeLeft > 2:
                    if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                        outputMessage = ("two finger swipe right")
                        pyautogui.press('right', presses=30)
                    timeLastTwoFingerSwipeLeft = currentTime 
                elif currentTime - timeLastThreeFingerSwipeLeft > 2:
                    if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
                        outputMessage = ("three finger swipe right")
                        #skip to end of video
                        pyautogui.keydown('option')
                        pyautogui.press('right')
                        pyautogui.keyup('option')
                    timeLastThreeFingerSwipeLeft = currentTime
            #Swipe down
            elif AreaDifference > 40000:
                if AreaDifferencePos > 0:
                    if currentTime - timeLastTwoFingerSwipeUp > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                            outputMessage = ("two finger swipe up")
                            #pyautogui.keydown('option')
                            pyautogui.press('up')
                            #pyautogui.keyup('option')
                        timeLastTwoFingerSwipeUp = currentTime 
                    elif currentTime - timeLastThreeFingerSwipeUp > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 \
                           and mostRecentFingers == 3 \
                           and AreaDifference > 70000:
                            outputMessage = ("three finger swipe up")
                            pyautogui.press('up', presses = 5)
                        timeLastThreeFingerSwipeUp = currentTime
                elif AreaDifferencePos < 0:
                    if currentTime - timeLastTwoFingerSwipeDown > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
                            outputMessage = ("two finger swipe down")
                            pyautogui.press('down')
                        timeLastTwoFingerSwipeDown = currentTime 
                    elif currentTime - timeLastThreeFingerSwipeDown > 2:
                        if lastFiveFramesFingers.count(mostRecentFingers) >= 4 \
                           and mostRecentFingers == 3 \
                           and AreaDifference > 70000:
                            outputMessage = ("three finger swipe down")
                            pyautogui.press('up', presses = 5)

                        timeLastThreeFingerSwipeDown = currentTime
            # two and three finger swipe up
            # elif centroidXDifference > minimumVerticalDistance \
            #      and farPointXDifference > minimumVerticalDistance \
            #      and abs(centroidXDifference) < (.25*mostRecentRadi) \
            #      and centroidXDifferencePos > 0:
            #     yay = 1
            #     # print("centroidXDifference = ", str(centroidXDifference)) # , " with position diff of ", centroidXDifferencePos)
            #     # print("centroidYDifference = ", str(centroidYDifference)) #, " with position diff of ", centroidYDifferencePos)
            #     # #print("farPointXDifference = ", str(farPointXDifference)) # , " with position diff of ", centroidXDifferencePos)
            #     # #print("farPointYDifference = ", str(farPointYDifference)) #, "
            #     # print("in right") # \
            #      # and DifferenceX > -100 \
            #      # and DifferenceY > minimumYDifference: #and centroidXDifferencePos < 0
            #     if currentTime - timeLastTwoFingerSwipeUp > 2:
            #         #if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
            #         outputMessage = ("two finger swipe up")

            #         timeLastTwoFingerSwipeup = currentTime 
            #     elif currentTime - timeLastThreeFingerSwipeUp > 2:
            #         #if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
            #         outputMessage = ("three finger swipe up")
            #         timeLastThreeFingerSwipeUp = currentTime
            # # two and three finger swipe up
            # elif centroidXDifference > minimumVerticalDistance \
            #      and farPointXDifference > minimumVerticalDistance \
            #      and abs(centroidXDifference) < (.25*mostRecentRadi) \
            #      and centroidXDifferencePos < 0:
            #     # yay = 1
            #     # print("centroidXDifference = ", str(centroidXDifference)) # , " with position diff of ", centroidXDifferencePos)
            #     # print("centroidYDifference = ", str(centroidYDifference)) #, " with position diff of ", centroidYDifferencePos)
            #     # #print("farPointXDifference = ", str(farPointXDifference)) # , " with position diff of ", centroidXDifferencePos)
            #     # #print("farPointYDifference = ", str(farPointYDifference)) #, "
            #     # print("in right") # \
            #      # and DifferenceX > -100 \
            #      # and DifferenceY > minimumYDifference: #and centroidXDifferencePos < 0
            #     if currentTime - timeLastTwoFingerSwipeDown > 2:
            #         #if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 2:
            #         outputMessage = ("two finger swipe down")

            #         timeLastTwoFingerSwipeDown = currentTime 
            #     elif currentTime - timeLastThreeFingerSwipeDown > 2:

            #         #if lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 3:
            #         outputMessage = ("three finger swipe down")
            #         timeLastThreeFingerSwipeDown = currentTime
            elif lastFiveFramesFingers.count(mostRecentFingers) >= 4 and mostRecentFingers == 0:
                currentCommand = (str(mostRecentFingers))
                outputMessage = (str(mostRecentFingers))
            else: 
                #print(mostRecentFingers)
                #print("count", lastFiveFramesFingers.count(mostRecentFingers))
                currentCommand = str(mostRecentFingers)
                if lastFiveFramesFingers.count(mostRecentFingers) >= 4:
                    currentCommand = (str(mostRecentFingers))
                    outputMessage = (str(mostRecentFingers))
                    #print("at the fingers")

            # COMPLEX GESTURES
            # Swipe up with one finger (volume up)

            # Swipe down with one finger (volume down)
            #print(currentCommand)

            if len(recentCommandList) < 10:
                recentCommandList.append(currentCommand)
            else: 
                recentCommandList.pop(0)
                recentCommandList.append(currentCommand)
            #print(recentCommandList)


            cv2.putText(thresholdedHandImage, outputMessage, 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            lineType)
            visual = cv2.resize(thresholdedHandImage, (850, 480))
            #print("max d detected: " , maxd)
            #resize for image input
            #visual = cv2.resize(visual, (426, 512))
            cv2.imshow(window_name, visual)
            # Do not delete the following line! Or else the visualization will not show up at all.
            key = cv2.waitKey(100)
            # else:
            #   print("FRAME REJECTED")
    except Exception as e: print(e)


        # 4 dynamic/complex gestures
        #       1. 2 finger swipe left/right: speed down/up video
        #       2. 2 finger up/down: volume up/down
        #       3. pinch to zoom in and out
        #       4. wave bye: close the window
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


    
    #k = cv2.waitKey(1) #k is the key pressed
    # if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
    #   #exit
    #   cv2.destroyAllWindows()
    #   cam.release()
    #   break


