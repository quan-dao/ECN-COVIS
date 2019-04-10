import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# cap = cv.VideoCapture('video1.mp4')
# frame = None
# for i in range(500):
# 	ret, frame = cap.read()
# cv.imwrite('new_frame.png', frame)

src_img = cv.imread('first_frame.png', 0)
new_img = cv.imread('new_frame.png', 0) 

MIN_MATCH_COUNT = 7

# Create SURF object
surf = cv.xfeatures2d.SURF_create()
surf.setHessianThreshold(400)

# Creat mask to restrict search for keypoint 
x, y, h, w = 24, 46, 170, 160
mask0 = np.zeros(src_img.shape)
mask1 = np.zeros(src_img.shape)
mask2 = np.zeros(src_img.shape)
mask3 = np.zeros(src_img.shape)
mask_list = [mask0, mask1, mask2, mask3]

box_array = np.array([[x, y],
                      [x, y+h-1],
                      [x+w-1, y+h-1],
                      [x+w-1, y]], dtype=np.int32)

roi_corners = np.array([[(x, y), 
                        (x, y+h-1), 
                        (x+w-1, y+h-1), 
                        (x+w-1, y)]], dtype=np.int32)
# create a mask with white pixels
mask = np.zeros(src_img.shape, dtype=np.uint8)
mask.fill(0)

# fill the ROI into the mask
cv.fillPoly(mask, roi_corners, 255)

# Find keypoints & descriptors
kp_src, des_src = surf.detectAndCompute(src_img, mask.astype(np.uint8))
# kp_src, des_src = surf.detectAndCompute(src_img, None)
kp_dest, des_dest = surf.detectAndCompute(new_img, None)

# FLANN params
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des_src, des_dest, k=2)

# store all the good matches as per Lowe's ratio test
good = []
for m, n in matches:
	if True: #-m.distance < 0.7 * n.distance:
		good.append(m)
		good.append(n)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp_src[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp_dest[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    pts = np.float32([ [x, y],[x, y+h-1],[x+w-1, y+h-1],[x+w-1, y] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    # src_img = cv.polylines(new_img, [np.int32(pts)], True, 255, 3, cv.LINE_AA)
    new_img = cv.polylines(new_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA) 

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0),
					singlePointColor=None,
					matchesMask=matchesMask,
					flags=2)

img3 = cv.drawMatches(src_img, kp_src, new_img, kp_dest, good, None, **draw_params)


# # # cv.imshow('new frame', new_img)
# img2 = cv.rectangle(src_img, (x, y), (x + w, y + h), 255, 2)
# cv.imshow('first_frame', img2)
# # cv.imshow('mask', mask)
# # # cv.imshow('keypoints', img)
cv.imshow('match', img3)
cv.waitKey(0)
cv.destroyAllWindows()
# # cap.release()
