#include "mrdir.h"
#include "mropencv.h"
#include "mrutil.h"
#include "MTCNN.h"

int minSize = 40;
float factor = 0.709f;
float thresholds[3] = { 0.7f, 0.6f, 0.6f };

void drawDetection(cv::Mat img, const std::vector<FaceInfo> &faceInfo) {
	for (int i = 0; i < faceInfo.size(); i++) {
		int x = (int)faceInfo[i].bbox.xmin;
		int y = (int)faceInfo[i].bbox.ymin;
		int w = (int)(faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
		int h = (int)(faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);
		cv::rectangle(img, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
		for (int j = 0; j < 5; j++) {
			cv::circle(img, cv::Point(faceInfo[i].landmark[j * 2], faceInfo[i].landmark[j * 2 + 1]), 3, cv::Scalar(0, 0, 255),-1);
		}
	}
}
void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color, int arrowMagnitude = 9, int thickness = 1, int line_type = 8, int shift = 0)
{
	//Draw the principle line
	cv::line(image, p, q, color, thickness, line_type, shift);
	const double PI = CV_PI;
	//compute the angle alpha
	double angle = atan2((double)p.y - q.y, (double)p.x - q.x);
	//compute the coordinates of the first segment
	p.x = (int)(q.x + arrowMagnitude * cos(angle + PI / 4));
	p.y = (int)(q.y + arrowMagnitude * sin(angle + PI / 4));
	//Draw the first segment
	cv::line(image, p, q, color, thickness, line_type, shift);
	//compute the coordinates of the second segment
	p.x = (int)(q.x + arrowMagnitude * cos(angle - PI / 4));
	p.y = (int)(q.y + arrowMagnitude * sin(angle - PI / 4));
	//Draw the second segment
	cv::line(image, p, q, color, thickness, line_type, shift);
}

void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d)
{
	cv::Scalar red(0, 0, 255);
	cv::Scalar green(0, 255, 0);
	cv::Scalar blue(255, 0, 0);
	cv::Scalar black(0, 0, 0);

	cv::Point2i origin = list_points2d[0];
	cv::Point2i pointX = list_points2d[1];
	cv::Point2i pointY = list_points2d[2];
	cv::Point2i pointZ = list_points2d[3];

	drawArrow(image, origin, pointX, red, 9, 2);
	drawArrow(image, origin, pointY, green, 9, 2);
	drawArrow(image, origin, pointZ, blue, 9, 2);
	cv::circle(image, origin, 2, black, -1);
}

void drawEAV(cv::Mat &img, cv::Vec3f &eav) {
	cv::putText(img, "x:" + double2string(eav[0]), cv::Point(0, 30), 3, 1, cv::Scalar(0, 0, 255));
	cv::putText(img, "y:" + double2string(eav[1]), cv::Point(0, 60), 3, 1, cv::Scalar(0, 255, 0));
	cv::putText(img, "z:" + double2string(eav[2]), cv::Point(0, 90), 3, 1, cv::Scalar(255, 0, 0));
}

class PoseEstimator {
public:
	cv::Vec3f estimateHeadPose(cv::Mat &img, const std::vector<Point2f > &imagePoints);
	PoseEstimator() { init(); }
private:
	std::vector<cv::Point3f > modelPoints;
	void init();
};

void PoseEstimator::init()
{
	modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
	modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
	modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
	modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
	modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695)   
}

cv::Vec3f PoseEstimator::estimateHeadPose(cv::Mat &img, const std::vector<cv::Point2f > &imagePoints)
{
	cv::Mat rvec, tvec;
	int max_d = (img.rows + img.cols) / 2;
	cv::Mat camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0, 0, max_d, img.rows / 2.0, 0, 0, 1.0);
	solvePnP(modelPoints, imagePoints, camMatrix, cv::Mat(), rvec, tvec, false, CV_EPNP);
	cv::Mat rotM;
	cv::Rodrigues(rvec, rotM);
	std::vector<cv::Point3f> axises;
	std::vector<cv::Point2f> pts2d;
	float l = 40;
	int x = modelPoints[2].x;
	int y = modelPoints[2].y;
	int z = modelPoints[2].z;
	axises.push_back(cv::Point3f(x, y, z));
	axises.push_back(cv::Point3f(x + l, y, z));
	axises.push_back(cv::Point3f(x, y + l, z));
	axises.push_back(cv::Point3f(x, y, z + l));
	projectPoints(axises, rvec, tvec, camMatrix, cv::Mat(), pts2d);
	draw3DCoordinateAxes(img, pts2d);
#if 0
	projectPoints(modelPoints, rvec, tvec, camMatrix, cv::Mat(), pts2d);
	for (int i = 0; i < pts2d.size(); i++) {
		cv::circle(img, pts2d[i], 5, cv::Scalar(255, 0, 0), -1);
	}
#endif
cv:Mat T;
	cv::Mat euler_angle;
	cv::Mat out_rotation, out_translation;
	cv::hconcat(rotM, rvec, T);
	cv::decomposeProjectionMatrix(T, camMatrix, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
	cv::Vec3f eav;
	for (int i = 0; i < 3; i++) {
		eav[i] = euler_angle.at<double>(0, i);
	}
	drawEAV(img, eav);
	return eav;
}

int test_dir(MTCNN &detector, std::string dir="../imgs"){
    std::vector<std::string>files=getAllFilesinDir(dir);
    for (int k =0;k<files.size();k++){
        std::cout<<files[k]<<std::endl;
        cv::Mat img = cv::imread(dir+"/"+files[k]);
        if (!img.data)
            continue;
        std::vector<FaceInfo> faceInfo = detector.Detect_mtcnn(img, minSize, thresholds, factor, 3);
		drawDetection(img, faceInfo);
        cv::imshow("image", img);
        cv::waitKey();
    }
    return 0;
}

int test_camera(MTCNN &detector){
    cv::VideoCapture cap(0);
    cv::Mat img;
	PoseEstimator pe;
    while(true){
        cap>>img;
        if (!img.data)
            break;
		cv::TickMeter tm;
		tm.start();
        std::vector<FaceInfo> faceInfo = detector.Detect_mtcnn(img, minSize, thresholds, factor, 3);
		tm.stop();
		std::cout << tm.getTimeMilli() << "ms"<<std::endl;
		for (int i = 0; i < faceInfo.size(); i++)
		{
			std::vector<cv::Point2f > imagePoints;
			auto fi = faceInfo[0];
			for (int i = 0; i < 5; i++) {
				imagePoints.push_back(cv::Point2f(fi.landmark[2*i], fi.landmark[2*i+1]));
			}
			auto eav = pe.estimateHeadPose(img, imagePoints);
		}
		drawDetection(img, faceInfo);
        cv::imshow("image", img);
        cv::waitKey(1);
    }
    return 0;
}

int main(int argc, char **argv)
{
    MTCNN detector("../model/fast");
    //test_dir(detector);
    test_camera(detector);
	return 0;
}