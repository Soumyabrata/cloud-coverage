import cv2
import matplotlib
import numpy as np
from utilities import *
import scipy.misc

# Mention your sky/cloud image path here. Please make sure that your input image is inside "image" folder.
image_loc = './image/2015-09-04-12-22-01-wahrsis3-med.jpg'


# This is the main component that performs the cloud coverage computation. It may take around 40s to run this component
(th_image,coverage) = cloudSegment(image_loc)


# Saving your final result. This is saved inside the "image" folder. The resultant image is appended with the suffix "-result" on the original file name
my_string = image_loc.split('.')
threshold_img_name = '.' + my_string[-2] + '-result.png'
scipy.misc.imsave(threshold_img_name, th_image)

print ('Binary image is saved')

print ('Coverage is ',coverage)
