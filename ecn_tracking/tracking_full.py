import cv2 as cv
import numpy as np
import sys


class VideoTracker(object):
	"""docstring for VideoTracker"""
	def __init__(self, video_path, f2f=False):
		'''
		video_path: path to video file want to track
		f2f: tracking the previous frame
		'''
		self.video = cv.VideoCapture(video_path)
		assert(self.video is not None)
		
		# tracking mode
		self.f2f = f2f

		# create SURF obejct
		self.surf = cv.xfeatures2d.SURF_create()
		self.surf.setHessianThreshold(400)

  		# create FLANN-based matcher
  		FLANN_INDEX_KDTREE = 1
  		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  		search_params = dict(checks=50)   # or pass empty dictionary
  		self.flann = cv.FlannBasedMatcher(index_params,search_params)
  		
  		# Hard-coding def of ROI in 1st frame
		x, y, w, h = 24, 46, 170, 160 
		self.box_array = np.float32([[x, y],
				  					[x, y+h-1],
				  					[x+w-1, y+h-1],
				  					[x+w-1, y]])

		# Read the first frame for initial keypoints & descriptors
		_frame = None
		while True:
			ret, _frame = self.video.read()
			if ret:
				break
		self.query_img = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
		self.findQueryKeypoints(self.query_img, self.box_array) 
		 
	def findQueryKeypoints(self, query_img, box_array):
		roi_corners = self.box_array.reshape(1, 4, 2)
		mask = np.zeros(query_img.shape, dtype=np.uint8)
		mask.fill(0)

		# fill the ROI into the mask
		cv.fillPoly(mask, roi_corners.astype(np.int32), 255)

		# Find keypoints & descriptors
		self.kp_query, self.des_query = self.surf.detectAndCompute(query_img, mask.astype(np.uint8))

	def findMatchedKeyPoints(self, train_img):
		'''
		find keypoints in train_img that matched with keypoint in query image
		'''
		# Find keypoints & descriptors in train_img
		self.kp_train, des_train = self.surf.detectAndCompute(train_img, None)

		matches = self.flann.knnMatch(self.des_query, des_train, k=2)

		# find the good matches by per Lowe's ratio test
		self.good_matches = []
		for m, n in matches:
			if m.distance < 0.7 * n.distance:
				self.good_matches.append(m)

	def localizeObject(self, train_img):
		'''
		Find the bounding box of the object in the train image
		'''
		if self.f2f:
			self.findQueryKeypoints(self.query_img, self.box_array)  # self.query_img & self.box_array need to be updated in f2f

		# Find matched keypoints
		self.findMatchedKeyPoints(train_img)

		if len(self.good_matches) > 7:  # need at least 8 points to estimate homography
			# Extract good mathces kp in query & train img
			src_pts = np.float32([self.kp_query[m.queryIdx].pt for m in self.good_matches]).reshape(-1,1,2)
			dst_pts = np.float32([self.kp_train[m.trainIdx].pt for m in self.good_matches]).reshape(-1,1,2)

			M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
			# matchesMask = mask.ravel().tolist()

			pts = self.box_array.reshape(-1, 1, 2)
			dst = cv.perspectiveTransform(pts, M)
		else:
			print "Not enough matches are found - %d/%d" % (len(self.good_matches), 8)
			dst = None
		
		# Update self.query_img & self.box_array in case of f2f
		if self.f2f:
			self.query_img = train_img
			self.box_array = dst.squeeze()

		return dst


if __name__ == '__main__':
	video_path = sys.argv[1]
	tracking_mode = sys.argv[2]
	if tracking_mode == '0':
		print 'Tracking using the first frame as reference'
		f2f = False
	else:
		print 'Tracking using the first frame as reference'
		f2f = True
	vid_tracker = VideoTracker(video_path, f2f)
	while vid_tracker.video.isOpened():
		ret, frame = vid_tracker.video.read()
		if ret:
			# get train_img
			train_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
			# get the bounding box of the object in train_img
			dst = vid_tracker.localizeObject(train_img)
			if dst is not None:
				# draw bounding box for found object
				tracked_img = cv.polylines(train_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
				# Display
				cv.imshow('result', tracked_img)
			else:
				cv.imshow('result', train_img)
			cv.waitKey(1)
		else:
			break

	cv.destroyAllWindows()
	vid_tracker.video.release()
	