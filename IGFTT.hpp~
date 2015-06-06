#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string>
#include <vector>
#include <map>
using namespace std;
using namespace cv;

class IGFTT: public Feature2D{
public:
	CV_WRAP static Ptr<IGFTT> create(string _desc_name = "SURF", int _nfeatures = 200, float _scaleFactor = 1.2f, int _nlevels = 8, int _firstLevel = 0, double _qualityLevel = 0.01,
int _blockSize = 31,int _minDistance=10, int _aperture_size = 3);
};

class IGFTT_Impl: public IGFTT {
public:
	IGFTT_Impl(string _desc_name = "SURF", int _nfeatures = 200, float _scaleFactor = 1.2f,
			int _nlevels = 8, int _firstLevel = 0, double _qualityLevel = 0.01,
			int _blockSize = 31,int _minDistance=10, int _aperture_size = 3);
	void detectAndCompute(InputArray image, InputArray mask,
                                           std::vector<KeyPoint>& keypoints,
                                           OutputArray descriptors,
                                           bool useProvidedKeypoints);
	void detect(InputArray image,std::vector<KeyPoint>& keypoints,
                                 InputArray mask);
protected:
	map< string, Ptr<Feature2D> > extractors;
	string desc_name;
	Mat descriptor;
	bool useDescriptor;
	int nfeatures;
	float scaleFactor;
	int nlevels;
	int firstLevel;
	double qualityLevel;
	int blockSize;
	int aperture_size;
	int minDistance;
	vector<Mat> imagePyramid;
	vector<vector<KeyPoint> > allKeypoints;
	vector<double> scales;
	void computeDescriptor(Mat image, vector<KeyPoint>& _keypoints, Mat mask);
	void computeKeyPoints(Mat image, vector<KeyPoint>& _keypoints, Mat mask);
	void computeOrientation(Mat img, vector<KeyPoint> &pts);
};
