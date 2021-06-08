# import cv2
#
# image = cv2.imread('C:/Users/NPC-1/Desktop/waldo.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow('Original image', image)
# cv2.imshow('Gray image', gray)
# cv2.imwrite('result.png',gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2

image = cv2.imread('C:/Users/NPC-1/Desktop/mask1.png')
result = 255 - image
alternative_result = cv2.bitwise_not(image)

cv2.imshow('image', image)
cv2.imshow('result', result)
cv2.imshow('alternative_result', alternative_result)
cv2.imwrite('result1.png',alternative_result)
cv2.waitKey(0)