#Ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

import cv2 as cv
import numpy as np


cap = cv.VideoCapture('video1.mp4')

MIN_MATCH_COUNT = 10  # minimum number of found mathc points

# Create SURF object
surf = cv.xfeatures2d.SURF_create()
surf.setHessianThreshold(400)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)


def findQueryKeypoints(query_img, box_array):
	# ROI in query_img
	box_array = np.int32(np.round(box_array))
	x = box_array[0, 0]
	y = box_array[0, 1]
	w = box_array[3, 0] - box_array[0, 0] + 1
	h = box_array[1, 1] - box_array[0, 1] + 1
	# create mask
	mask = np.zeros(query_img.shape)
	mask[x: x+w, y: y+h] = 1
	# Find keypoints & descriptors
	kp_query, des_query = surf.detectAndCompute(query_img, mask.astype(np.uint8))
	return kp_query, des_query



def findMatchedKeyPoints(train_img, des_query):
	# Find key points & descriptors in train img
	kp_train, des_train = surf.detectAndCompute(train_img, None)

	matches = flann.knnMatch(des_query, des_train, k=2)

	# store all the good matches as per Lowe's ratio test
	good_matches = []
	for m, n in matches:
		if m.distance < 0.7 * n.distance:
			good_matches.append(m)

	return good_matches, kp_train


def localizeObject(kp_query, kp_train, good_matches, box_array):
	# Extract good mathces kp in query & train img
	src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
	dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	pts = box_array.reshape(-1, 1, 2)

	dst = cv.perspectiveTransform(pts, M)
	return dst


# Initial value
query_img = cv.imread('first_frame.png', 0)  
x, y, w, h = 24, 46, 170, 160
box_array = np.float32([[x, y],
					  [x, y+h-1],
					  [x+w-1, y+h-1],
					  [x+w-1, y]])

while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		# Get train_img
		train_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Find keypoints & descriptors in query_img
		kp_query, des_query = findQueryKeypoints(query_img, box_array)
		
		# Find matched keypoints
		good_matches, kp_train = findMatchedKeyPoints(train_img, des_query)
		
		if len(good_matches) >= MIN_MATCH_COUNT:
			dst = localizeObject(kp_query, kp_train, good_matches, box_array)
			# Update query_img & box_array
			query_img = train_img
			box_array = dst.squeeze()
			# draw bounding box for found object
			tracked_img = cv.polylines(train_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
			# Display
			cv.imshow('tracked', tracked_img)
		else:
			print "Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT)
			matchesMask = None
			# # reset query_img & box_array
			# query_img = cv.imread('first_frame.png', 0)  
			# x, y, w, h = 24, 46, 170, 160
			# box_array = np.float32([[x, y],
			# 					  [x, y+h-1],
			# 					  [x+w-1, y+h-1],
			# 					  [x+w-1, y]])
			# Display
			cv.imshow('tracked', train_img)
		
		cv.waitKey(1)
	else:
		break

cv.destroyAllWindows()
cap.release()
