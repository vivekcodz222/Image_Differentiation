#Import of packages
from skimage.metrics import structural_similarity as compare__ssim
import argparse
import imutils
import cv2

#Construct the argument prase and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f","--first", required=True,
                help="first input image")
ap.add_argument("-s","--second",required=True,
                help="second input image")
args= vars(ap.parse_args())

#Image Difference with OpenCV and python
# load the two input images
imageA = cv2.imread(args["first"])
imageB=cv2.imread(args["second"])

#converting the image into grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

#compute the structural Similarity Index (SSIM) between the two
#image, ensuring that the differance image is returned
(score,diff) = compare__ssim(grayA, grayB, full = True)
diff = (diff*255).astype("uint8")
print("SSIM:{}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
# The line you need to run on the terminal and inplace of the input1.png you have to give your first file name and in
# placeof the input2.png you have to give your second image.
# python image_diff.py --f input1.png --s input2.png
# I have tested it and it's working well i have also uploaded the results in the drive link.
# PS D:\CODING PAGE\InterviewProjects> python image_diff.py --f input1.png --s input2.png
# C:\Users\anitha\AppData\Local\Programs\Python\Python311\python.exe: can't open file 'D:\\CODING PAGE\\InterviewProjects\\image_diff.py': [Errno 2] No such file or directory
# PS D:\CODING PAGE\InterviewProjects> cd .\Image_Differance\
# PS D:\CODING PAGE\InterviewProjects\Image_Differance> python image_diff.py --f input1.png --s input2.png
# SSIM:0.9388101913918601

#Thank you
#Regards, Vivek.
