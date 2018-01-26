from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

image = io.imread("../data/test10.jpg")
#image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

#blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
blobs_log = blob_dog(image_gray, max_sigma=2, threshold=.02)
#reduced the threshold to detect blobs with less intensities
#reduced the max_sigma to limit the size of detected blobs.

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

#blobs_log = blob_doh(image_gray, max_sigma=30, threshold=.01)

#blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
#blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

#blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

blobs_list = [blobs_log]
colors = [ 'lime']
titles = ['Laplacian of Gaussian']
sequence = zip(blobs_list, colors, titles)

#axes = plt.figure()
#fig, ax = plt.subplots(1, 1, figsize=(9, 3), sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})

#plt.set_title(title)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(image, interpolation='nearest')
for blob in blobs_log:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='lime', linewidth=0.2, fill=False)
    ax.add_patch(c)
ax.set_axis_off()

plt.tight_layout()
plt.savefig('result10(3).jpg', dpi = 1500)

plt.show()
