import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import cv2

im1 = plt.imread('my_images/frame1.jpg')
im2 = plt.imread('my_images/canny_edge1.jpg')
im3 = plt.imread('my_images/mask2.jpg')
im4 = plt.imread('my_images/masked_image3.jpg')
im5 = plt.imread('my_images/line_image4.jpg')
im6 = plt.imread('my_images/final_image5.jpg')

fig = plt.figure(figsize=(6., 6.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(3, 3),
                 axes_pad=0.2,
                )

for ax, im in zip(grid, [im1, im2, im3, im4, im5, im6]):
    ax.imshow(im)

plt.show()
