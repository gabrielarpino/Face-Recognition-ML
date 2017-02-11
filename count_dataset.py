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

act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

female_act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']
male_act = ['Alec Baldwin', 'Bill Hader', 'Steve Carell']

location1 = "facescrub_actresses.txt"
location2 = "facescrub_actors.txt"

carell = 0
hader = 0
baldwin = 0
butler = 0
radcliffe = 0
vartan = 0
for line in open(location2):
	if 'Steve Carell' in line:
		carell = carell+1
	if 'Bill Hader' in line:
		hader = hader+1
	if 'Alec Baldwin' in line:
		baldwin += 1
	if 'Butler' in line:
		butler = butler+1
	if 'Radcliffe' in line:
		radcliffe = radcliffe+1
	if 'Vartan' in line:
		vartan = vartan+1


print 'Steve Carell', carell
print 'Bill Hader', hader
print 'Alec Baldwin', baldwin
print 'Gerard Butler', butler
print 'Daniel Radcliffe', radcliffe
print 'Michael Vartan', vartan

drescher = 0
ferrera = 0
chenoweth = 0
bracco = 0
gilpin = 0
harmon = 0
for line in open(location1):
	if 'Drescher' in line:
		drescher = drescher+1
	if 'Ferrera' in line:
		ferrera = ferrera+1
	if 'Chenoweth' in line:
		chenoweth += 1
	if 'Bracco' in line:
		bracco = bracco+1
	if 'Gilpin' in line:
		gilpin = gilpin+1
	if 'Harmon' in line:
		harmon = harmon+1


print 'Fran Drescher', drescher
print 'America Ferrera', ferrera
print 'Kristin Chenoweth', chenoweth
print 'Lorraine Bracco', bracco
print 'Peri Gilpin', gilpin
print 'Angie Harmon', harmon