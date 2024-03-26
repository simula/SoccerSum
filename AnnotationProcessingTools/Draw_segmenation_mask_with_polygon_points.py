import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read an image
img = cv2.imread("/content/100000000032.jpg")

# Define the list of points (coordinates)
points = ['point1', 'point2', ...]

# Using list comprehension to create a list of tuples
point_tuples = [(int(points[i] * img.shape[1]), int(points[i+1] * img.shape[0])) for i in range(0, len(points), 2)]

# Define an array of endpoints of the polygon
points = np.array(point_tuples)

# Use fillPoly() function and give input as image,
# end points, color of polygon
# Here color of polygon will be green
cv2.fillPoly(img, pts=[points], color=(0, 255, 0))

# Displaying the image
cv2_imshow(img)

cv2.waitKey(0)

# Closing all open windows
cv2.destroyAllWindows()
