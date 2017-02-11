from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import glob

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']

female_act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()

# Create the necessary directories

if not os.path.isdir('uncropped'):
    os.makedirs('uncropped')

if not os.path.isdir('cropped'):
    os.makedirs('cropped')

if not os.path.isdir('cropped/male'):
    os.makedirs('cropped/male')

if not os.path.isdir('cropped/female'):
    os.makedirs('cropped/female')

for a in act:
    name = a.split()[1].lower()
    i = 0
    if a in female_act:
        location = "facescrub_actresses.txt"
    if a in male_act:
        location = "facescrub_actors.txt"
    for line in open(location):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            try:
                x1, y1 = int(line.split()[-2].split(',')[0]), int(line.split()[-2].split(',')[1])
                x2, y2 = int(line.split()[-2].split(',')[2]), int(line.split()[-2].split(',')[3])
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)    #Save raw images into uncropped folder
                if not os.path.isfile("uncropped/"+filename):
                    continue
                print filename
            except Exception:
                continue
            try:
                img = imread("uncropped/" + filename)                   # Read image from uncropped folder into an array
                cropped = img[y1:y2, x1:x2]                             # crop the image
                zopped = rgb2gray(cropped)                              # Convert image to gray
                mopped = imresize(zopped, (32,32))                      # Resize image to 32x32
                if a in male_act:
                    imsave("cropped/male/" + filename, mopped)
                    print "imsaved:", filename
                else:
                    imsave("cropped/female/" + filename, mopped)
                    print "imsaved:", filename
                i += 1
                continue
            except Exception:                                           # Ignore image reading, cropping, resizing error that occured due to corrupted images
                continue
    
        

            
    
    