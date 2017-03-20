from inputdata import *
from gate import Gates

gates = Gates(force=True, imagesize = 64, binary = True, dilation = False)

# Gates.train.X : train images
# Gates.train.Y : train labels
# Gates.test.X  : test images
# Gates.test.Y  : test labels
# Gates.AND/NAND/XOR... : AND/NAND/XOR.. images


def show(img):
	plt.imshow(img*255, cmap = 'gray')
	plt.show()

# show the picture
for i in range(10):
	show(gates.train.X[i])
	show(gates.XNOR[i])
