# DuckPropagation
In this project we are training a duckiebot for autonomous driving.

## Authors
Attila Pethő  
Gyula Faragó  
Olivér Farkas  

## Wrappers
#### ResizeFrame
With this wrapper we are downscaling the images from their original size (480x640x3) to (84x84x3). The smaller dimension makes the training of the neural network faster, and it still carries enough information for the training.
#### CropFrame
In this wrapper we are cropping the useless information from the image, in our case it's the part above the horizon.
#### ColorSegmentFrame
Here we are segmenting the different parts of the image so we can feed the neural network more useful information. The segmentation is done with using intervals, we assigned red for the white line, and green for the yellow line. For lane following only this two information is useful for the neural network so we assign black for everything else.
#### NormalizeFrame
To make the training of the CNN easier and faster we are normalizing the pixel values in the interval of [0,1].
#### StackFrame
For better quality in training and more informations we are concatenating the last **nframes**.
