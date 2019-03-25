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


def sign(x):
	if x > 0:
		return 1
	else:
		return -1


def findQueryKeypoints(query_img, box_array):
	# ROI in query_img
	box_array = np.int32(np.round(box_array))
	x0 = np.append(box_array[0], 1)
	x1 = np.append(box_array[1], 1)
	x2 = np.append(box_array[2], 1)
	x3 = np.append(box_array[3], 1)
	
	# Find lines cross 2 pair of vertexes
	l0 = np.cross(x0, x1)
	l1 = np.cross(x1, x2)
	l2 = np.cross(x2, x3) 
	l3 = np.cross(x3, x0)
	l = [l0, l1, l2, l3]

	# create mask
	mask_list = [np.zeros(query_img.shape) for i in range(4)]
	for h, mask in enumerate(mask_list):
	    for i in range(mask.shape[0]):
	        for j in range(mask.shape[1]):
	            x = np.array([i, j, 1])
	            if sign(l[h].dot(x)) > 0:
	                mask[i, j] = 1

	mask02 = 1 - cv.bitwise_or(mask_list[0], mask_list[2])
	mask13 = 1 - cv.bitwise_not(cv.bitwise_or(mask_list[1], mask_list[3]))
	mask = cv.bitwise_and(mask02, mask13)

	# Find keypoints & descriptors
	kp_query, des_query = surf.detectAndCompute(query_img, mask.astype(np.uint8))
	return mask, kp_query, des_query



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
		mask, kp_query, des_query = findQueryKeypoints(query_img, box_array)
		cv.imshow('mask', mask)
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


# Frame-2-Frame fails cause it doesn't have  reference
