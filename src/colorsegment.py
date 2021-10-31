import numpy as np
from PIL import Image

# Define constant values

sensitivity_yellow = 100
sensitivity_white = 50

colorcode_yellow = [255,255,0]
colorcode_white = [255,255,255]

threshold_yellow = [colorcode_yellow[0] - sensitivity_yellow, colorcode_yellow[1] - sensitivity_yellow, colorcode_yellow[2] + sensitivity_yellow]
threshold_white = [colorcode_white[0] - sensitivity_white, colorcode_white[1] - sensitivity_white, colorcode_white[2] - sensitivity_white]

#Test input 

#####################

img = Image.open('images/raw_obs/20211030_164024_1.jpg')
test_image_array = np.array(img) 


######################

#Function define

def colorsegment(numpy_array_image):
    for x in numpy_array_image:
        for y in x:
            if((y[0] - threshold_yellow[0] > 0) and (y[1] - threshold_yellow[1] > 0) and (y[2] - threshold_yellow[2] < 0)):  #Is yellow?
                y[0] = 255
                y[1] = 0
                y[2] = 0
            else:
                if((y[0] - threshold_white[0] > 0) and (y[1] - threshold_white[1] > 0) and (y[2] - threshold_white[2] > 0)): #Is white?
                    y[0] = 0
                    y[1] = 255
                    y[2] = 0
                else:
                    y[0] = 0
                    y[1] = 0
                    y[2] = 0
    return numpy_array_image


#Test output

######################

test_image_array = colorsegment(test_image_array)

img = Image.fromarray(test_image_array, 'RGB')
img.show()

######################
        
