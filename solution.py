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

# methods to extract OKR and OKN:
# 1. use a CNN to extract features from the video footage
# 2. use a RNN to extract the gaze direction from the features
# 3. use a LSTM to extract the OKN from the gaze direction
# 4. neural networks leveraging on deep learning to detect eye. 
#    e.g. RCNN + use of Rectangular-intensity-gradient (RIG) for eye centre localisation (But I don't know how they work)
# 5. 

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


# read images from video file. press esc to quit converting video to images
import cv2

vidcap = cv2.VideoCapture('test_video.MOV')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1