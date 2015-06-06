#include "IGFTT.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;

Ptr<IGFTT> IGFTT::create(string _desc_name,int _nfeatures, float _scaleFactor, int _nlevels, int _firstLevel, double _qualityLevel,int _blockSize, int _minDistance, int _aperture_size)
{
	 return makePtr<IGFTT_Impl>(_desc_name, _nfeatures, _scaleFactor, _nlevels, _firstLevel, _qualityLevel, _minDistance, _blockSize, _aperture_size);
}

IGFTT_Impl::IGFTT_Impl(string _desc_name, int _nfeatures, float _scaleFactor, int _nlevels,
		int _firstLevel, double _qualityLevel, int _blockSize,int _minDistance,
		int _aperture_size){
	nfeatures=_nfeatures;
	scaleFactor=_scaleFactor;
	nlevels=_nlevels;
	firstLevel=_firstLevel;
	qualityLevel=_qualityLevel;
	blockSize=_blockSize;
	aperture_size=_aperture_size;
	desc_name = _desc_name;
	useDescriptor = false;
	minDistance = _minDistance;
	extractors["FREAK"] =xfeatures2d::FREAK::create();
	extractors["SIFT"] =xfeatures2d::SIFT::create();
	extractors["SURF"] =xfeatures2d::SURF::create();
	extractors["BRIEF"] =xfeatures2d::BriefDescriptorExtractor::create(32);
	extractors["ORB"] = ORB::create();
	extractors["BRISK"] = BRISK::create();

	for (int level = 0; level < nlevels; ++level)
		scales.push_back(std::pow(scaleFactor, (double) (level - firstLevel)));

}

void IGFTT_Impl::computeOrientation(Mat img, vector<KeyPoint> &pts) {

	const uchar* ptr00 = img.ptr<uchar>();
	int step = (int) (img.step / img.elemSize1());
	int ptsize = pts.size();

	for (int ptidx = 0; ptidx < ptsize; ptidx++) {
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);

		const uchar* ptr = ptr00 + y0 * step + x0;
		int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1])
				+ (ptr[step + 1] - ptr[step - 1]);
		int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1])
				+ (ptr[step + 1] - ptr[-step + 1]);

		float a = Ix * Ix, b = Iy * Iy, c = Ix * Iy;

		float u = (a + c) * 0.5;
		float v = std::sqrt((a - c) * (a - c) * 0.25 + b * b);
		float l2 = u - v;

		float x = b;
		float y = l2 - a;
		float e = fabs(x);

		if (e + fabs(y) < 1e-4) {
			y = b;
			x = l2 - c;
			e = fabs(x);
			if (e + fabs(y) < 1e-4) {
				e = 1. / (e + fabs(y) + FLT_EPSILON);
				x *= e, y *= e;
			}
		}

		float d = 1. / std::sqrt(x * x + y * y + DBL_EPSILON);
		pts[ptidx].angle = fastAtan2(y * d, x * d);
	}
}

void IGFTT_Impl::computeDescriptor(Mat image, vector<KeyPoint>& _keypoints, Mat mask){
	Mat desc;
	extractors[desc_name]->compute(image,_keypoints,desc);
	descriptor.push_back(desc);
}

void IGFTT_Impl::computeKeyPoints(Mat image, vector<KeyPoint>& _keypoints, Mat mask) {
	imagePyramid.clear();
	imagePyramid.resize(nlevels);
	image.copyTo(imagePyramid[0]);

	for (int level = 1; level < nlevels; ++level) {
		double scale = 1 / scales[level];

		Size sz(cvRound(image.cols * scale), cvRound(image.rows * scale));
		resize(imagePyramid[level - 1], imagePyramid[level], sz);
	}

	// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
	allKeypoints.clear();
	_keypoints.clear();
	allKeypoints.resize(nlevels);
	Ptr<GFTTDetector> g = GFTTDetector::create(nfeatures, qualityLevel, minDistance, blockSize);
	vector<Point2f> corners;
	for (int level = 0; level < (int) imagePyramid.size(); ++level) {
		// Get the features and compute their orientation
		vector<KeyPoint>& keypoints = allKeypoints[level];
		//GaussianBlur(imagePyramid[level],imagePyramid[level],Size(5,5), 0.5);
		//Laplacian(imagePyramid[level],imagePyramid[level],-1,3,level);
		g->detect(imagePyramid[level], keypoints);
		computeOrientation(imagePyramid[level], keypoints);
		double scale = scales[level];
		if (useDescriptor)
			computeDescriptor(imagePyramid[level],keypoints, mask);
		if (level != firstLevel)
			for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
					keypointEnd = keypoints.end(); keypoint != keypointEnd;
					++keypoint) {
				keypoint->octave = level;
				keypoint->size = blockSize * scale;
				keypoint->pt *= scale;
			}
		// And add the keypoints to the output
		_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
	}
}

void IGFTT_Impl::detect(InputArray _image,std::vector<KeyPoint>& keypoints,
                                 InputArray _mask) {
	Mat image = _image.getMat(), mask = _mask.getMat();
	keypoints.clear();

	Mat mask;
	computeKeyPoints(image, keypoints, mask);
}


void IGFTT_Impl::detectAndCompute(InputArray image, InputArray mask,
                                           std::vector<KeyPoint>& keypoints,
                                           OutputArray _descriptors,
                                           bool useProvidedKeypoints){

	descriptor.release();
	Mat& desc = _descriptors.getMatRef();
	useDescriptor = true;
	detect(image,keypoints,mask);
	descriptor.copyTo(desc);
}
