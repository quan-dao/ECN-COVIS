#include <visp/vpHomogeneousMatrix.h>

#include <visp/vpPoint.h>
#include <visp/vpSubColVector.h>
#include <visp/vpSubMatrix.h>
#include <visp/vpFeaturePoint.h>
#include <visp/vpFeatureBuilder.h>
#include <visp/vpExponentialMap.h>
#include <visp/vpAdaptiveGain.h>
#include <visp/vpIoTools.h>
#include <fstream>

#include <opencv2/calib3d/calib3d.hpp>

#include <vvs.h>
#include <grid_tracker.h>
#include <perspective_camera.h>
#include <distortion_camera.h>
#include <cb_tracker.h>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::stringstream;
using cv::waitKey;
using namespace covis;

int main()
{
    // load calibration images from hard drive
    const string base = "/home/mquan/ecn/covis/calibrationImages/";
    const string prefix = "img";

    //GridTracker tracker;      // this tracker detects a 6x6 grid of points
    CBTracker tracker(8,6);     // this one is to be given the chessboard dimension (8x6)
    // create a camera model (Perspective or Distortion)
    PerspectiveCamera cam(544.6583937,  546.1633695,  319.7788736,  235.378131);   // true intrinsic param

    // initiate virtual visual servoing with inter-point distance and pattern dimensions
    VVS vvs(cam, 0.03, 8, 6);

    vpHomogeneousMatrix M;
    M.buildFrom(
        0,0.,0.5,                                                                        // translation
        0,0,0);   // rotation

    for(int i = 0; i < 9; ++i) {
        stringstream ss;
        ss << prefix << i << ".jpg";
        std::ifstream testfile(base + ss.str());
        if(testfile.good()) {
            Pattern pat;
            pat.im = cv::imread(base + ss.str());
            tracker.detect(pat.im, pat.point);
            pat.window = ss.str();

            // draw extraction results
            drawSeq(pat.window, pat.im, pat.point);
            waitKey(0);

            // calibrate from this single image
            // the initial guess for M

            vvs.computePose(pat, M);
        }
    }

    // this will wait for a key pressed to stop the program
    waitKey(0);
}
