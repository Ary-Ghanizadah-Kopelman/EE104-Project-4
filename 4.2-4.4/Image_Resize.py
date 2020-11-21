#This program converts an imported PNG image (user specified) and converts it to 32x32 format for the CIFAR 10 image recognition library 

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Put name of image here (.PNG only), do not include file type e.g. '.png' 
image_import='truck_1'

#Show imported image
print("Image imported:",''+image_import+'.png')
img1 = mpimg.imread(''+image_import+'.png')
plt.figure(1)
plt.subplot(211)
plt.imshow(img1)

#Convert the image
img = Image.open(''+image_import+'.png')
new_width  = 32
new_height = 32
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save(''+image_import+'_converted.png') 

#Show converted image
print("Image has been converted to:",''+image_import+'_converted.png')
img2 = mpimg.imread(''+image_import+'_converted.png')
plt.subplot(212)
plt.imshow(img2)
plt.show()