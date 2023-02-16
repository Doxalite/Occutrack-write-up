# problem: capturing and processing biodata in the real world is noisy and tedious. Describe a DL /AI architecture, 
#          algorithm, framework, bioinformatics approach, protocol, method, or approach you would take to process and extract human gaze from a camera.

# what I have:
# 1. Video footage of eye

# What I want:
#       1. extract gaze from video footage, turn it into comprehensible data (e.g. time, a vector of gaze direction)
# update1: extract optokinetic response (OKR) from video footage, turn it into comprehensible data (e.g. time, a vector of gaze direction)
# update2: extract optokinetic nystagmus (OKN) from video footage, turn it into comprehensible data (e.g. time, a vector of gaze direction, 
#          amplitude, frequency of oscillation)
#          OKN is the repeated snapping back and forth of the eyes in response to a moving visual stimulus. I want to measure this.
# update3: To do that i first need to be able to identify the eye from the video to locate centre of eye. 
#          Then track centre of eye over many video frames to determine position change.
#          I can calculate pixels/distance of movement of centre of eye over many cycles of the OKR to determine the frequency, amplitude of the OKR.

# existing methods to extract OKR and OKN:
# 1. use a CNN for pupil detection (I don't know how they work yet)
# 2. neural networks leveraging on deep learning to detect eye. 
#    e.g. RCNN + use of Rectangular-intensity-gradient (RIG) for eye centre localisation (I don't know how they work yet also)

# how I would do it:
# opencv to convert video footage of eye into individual image frames
# train a SVM model to classify eye from non-eye images
# i need an algorithm that will tell me where the centre of the eye is from the image frame...




# References:

# https://www.frontiersin.org/articles/10.3389/fnbot.2021.796895/full
# https://www.frontiersin.org/files/Articles/796895/fnbot-15-796895-HTML/image_m/fnbot-15-796895-t001.jpg
# https://www.sciencedirect.com/topics/computer-science/optical-tracking
# https://ieeexplore.ieee.org/document/1047459
# https://www.quora.com/How-exactly-does-Computer-Vision-and-Machine-Learning-differ
# https://www.run.ai/guides/deep-learning-for-computer-vision#:~:text=Modern%20computer%20vision%20algorithms%20are,to%20traditional%20image%20processing%20algorithms.
# https://www.nature.com/articles/s41598-020-60531-3
# https://www.nature.com/articles/s41467-020-19712-x
# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ac942c4870e55fe1d9822d62edcdb685d41cd2bf
# https://www.mdpi.com/1424-8220/20/13/3785


# read images from video file. press esc to quit converting video to images
import cv2

img_list = []
vidcap = cv2.VideoCapture('test_video.MOV')
success,image = vidcap.read()
count = 0
# while success:
#     img_list.append(image)
#     cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
#     success,image = vidcap.read()
#     print('Read a new frame: ', success)
#     count += 1

for i in range(1):
    img_list.append(image)
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1

# SVM model to classify eye from non-eye images, remove all non-eye images

# model to identify position of centre of eye from image frame (distance either in pixels or mm from left edge of eye)
# a simple model that i came up with is to identify left and right edges of eye, then find the midpoint between the 2 edges
# use opencv functions to identify edges of eye
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
for img in img_list:
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(img_grey)
    for (x,y,w,h) in eyes:
        eye_centre = (x + w//2, y + h//2)
        radius = int(round((w + h)*0.25))
        img = cv2.circle(img, eye_centre, radius, (255, 0, 0 ), 4)
    cv2.imshow('Capture - Face detection', img) # do not run this, there is no eye to identify in test_video.MOV

# place positions of centre of eye into a dataframe

# do analysis on the dataframe to extract OKR and OKN
# Amplitude: calculate largest distance between 2 values in the dataframe
# Frequency: calculate number of frames per cycle of the OKR (i.e. number of frames between 2 consecutive peaks in the dataframe)

# done, pending further analysis of generated data