import cv2 as cv
import numpy as np


cap = cv.VideoCapture('video1.mp4')

query_img = cv.imread('first_frame.png', 0)

MIN_MATCH_COUNT = 10  # minimum number of found mathc points

# Create SURF object
surf = cv.xfeatures2d.SURF_create()
surf.setHessianThreshold(400)

# Creat mask to restrict search for keypoint 
x, y, w, h = 24, 46, 170, 160  # hard-coded bounding box
mask = np.zeros(query_img.shape)
mask[x:x+w, y:y+h] = 1

# Find keypoints & descriptors
kp_query, des_query = surf.detectAndCompute(query_img, mask.astype(np.uint8))

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params)


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


def localizeObject(kp_query, kp_train, good_matches, original_bounding_box=(x, y, w, h)):
	# Extract good mathces kp in query & train img
	src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
	dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
	matchesMask = mask.ravel().tolist()

	x, y, w, h = original_bounding_box
	pts = np.float32([[x, y],
					[x, y + h - 1],
					[x + w - 1, y + h - 1],
					[x + w - 1, y]]).reshape(-1,1,2)
	dst = cv.perspectiveTransform(pts, M)
	return dst


while cap.isOpened():
	ret, frame = cap.read()
	if ret:
		train_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		good_matches, kp_train = findMatchedKeyPoints(train_img, des_query)
		if len(good_matches) > MIN_MATCH_COUNT:
			dst = localizeObject(kp_query, kp_train, good_matches)
			# draw bounding box for found object
			train_img = cv.polylines(train_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
		else:
			print "Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT)
			matchesMask = None

		cv.imshow('tracked', train_img)
		cv.waitKey(1)
	else:
		break

cv.destroyAllWindows()
cap.release()
