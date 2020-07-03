#include "MTCNN.h"
#include "mrdir.h"
#include "mropencv.h"
#include "mrutil.h"

using namespace mtcnn;
using namespace std;

const std::string rootdir = "../";
const std::string imgdir = rootdir+"/imgs";
const std::string resultdir = rootdir + "/results";
const std::string proto_model_dir = rootdir + "/model/caffe";

#if _WIN32
const string casiadir = "E:/Face/Datasets/CASIA-maxpy-clean";
const string outdir = "E:/Face/Datasets/CASIA-mtcnn";
#else
const string casiadir = "~/CASIA-maxpy-clean";
const string outdir = "~/CASIA-mtcnn";
#endif

std::vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<FaceInfo>&faceInfo)
{
	std::vector<cv::Point2f>  p2s;
	p2s.push_back(cv::Point2f(30.2946, 51.6963));
	p2s.push_back(cv::Point2f(65.5318, 51.5014));
	p2s.push_back(cv::Point2f(48.0252, 71.7366));
	p2s.push_back(cv::Point2f(33.5493, 92.3655));
	p2s.push_back(cv::Point2f(62.7299, 92.2041));
	vector<Mat>dsts;
	for (int i = 0; i < faceInfo.size(); i++)
	{
		std::vector<cv::Point2f> p1s;
		FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j < 5; j++)
		{
			p1s.push_back(cv::Point(facePts.y[j], facePts.x[j]));
		}
		cv::Mat t = cv::estimateRigidTransform(p1s, p2s, false);
		if (!t.empty())
		{
			Mat dst;
			cv::warpAffine(img, dst, t, cv::Size(96, 112));
			dsts.push_back(dst);
		}
		else
		{
			dsts.push_back(img);
		}
	}
	return dsts;
}

void drawArrow(cv::Mat image, cv::Point2i p, cv::Point2i q, cv::Scalar color,  int arrowMagnitude = 9, int thickness=1, int line_type=8, int shift=0)
{
    //Draw the principle line
    cv::line(image, p, q, color, thickness, line_type, shift);
    const double PI = CV_PI;
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle + PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle + PI/4));
    //Draw the first segment
    cv::line(image, p, q, color, thickness, line_type, shift);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x +  arrowMagnitude * cos(angle - PI/4));
    p.y = (int) ( q.y +  arrowMagnitude * sin(angle - PI/4));
    //Draw the second segment
    cv::line(image, p, q, color, thickness, line_type, shift);
}

void draw3DCoordinateAxes(cv::Mat image, const std::vector<cv::Point2f> &list_points2d)
{
    cv::Scalar red(0, 0, 255);
    cv::Scalar green(0,255,0);
    cv::Scalar blue(255,0,0);
    cv::Scalar black(0,0,0);

    cv::Point2i origin = list_points2d[0];
    cv::Point2i pointX = list_points2d[1];
    cv::Point2i pointY = list_points2d[2];
    cv::Point2i pointZ = list_points2d[3];

    drawArrow(image, origin, pointX, red, 9, 2);
    drawArrow(image, origin, pointY, green, 9, 2);
    drawArrow(image, origin, pointZ, blue, 9, 2);
    cv::circle(image, origin, 2, black, -1 );
}

void drawEAV(cv::Mat &img,cv::Vec3f &eav){
	cv::putText(img,"x:"+double2string(eav[0]),cv::Point(0,30),3,1,cv::Scalar(0,0,255));
	cv::putText(img,"y:"+double2string(eav[1]),cv::Point(0,60),3,1,cv::Scalar(0,255,0));
	cv::putText(img,"z:"+double2string(eav[2]),cv::Point(0,90),3,1,cv::Scalar(255,0,0));
}

class PoseEstimator{
public:
    cv::Vec3f estimateHeadPose(cv::Mat &img, const vector<Point2f > &imagePoints);
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

cv::Vec3f PoseEstimator::estimateHeadPose(cv::Mat &img,const vector<Point2f > &imagePoints)
{
    cv::Mat rvec, tvec;
    int max_d = (img.rows + img.cols)/2;
    cv::Mat camMatrix = (Mat_<double>(3, 3) << max_d, 0, img.cols / 2.0,0, max_d, img.rows / 2.0,0, 0, 1.0);
    solvePnP(modelPoints,imagePoints, camMatrix, cv::Mat(), rvec, tvec, false, CV_EPNP);
	cv::Mat rotM;
	cv::Rodrigues(rvec, rotM);
	std::vector<cv::Point3f> axises;
	std::vector<cv::Point2f> pts2d;
	float l = 40;
	int x = modelPoints[2].x;
	int y = modelPoints[2].y;
	int z = modelPoints[2].z;
	axises.push_back(cv::Point3f(x,y,z));
	axises.push_back(cv::Point3f(x+l,y,z));
	axises.push_back(cv::Point3f(x,y+l,z));
	axises.push_back(cv::Point3f(x,y,z+l));
	projectPoints(axises,rvec,tvec,camMatrix,cv::Mat(),pts2d);
	draw3DCoordinateAxes(img,pts2d);
	#if 0
		projectPoints(modelPoints,rvec,tvec,camMatrix,cv::Mat(),pts2d);
		for(int i = 0; i < pts2d.size(); i++){
			cv::circle(img,pts2d[i],5,cv::Scalar(255,0,0),-1);
		}
	#endif
	cv:Mat T;
	cv::Mat euler_angle;
	cv::Mat out_rotation, out_translation;
	cv::hconcat(rotM, rvec, T);
	cv::decomposeProjectionMatrix(T,camMatrix,out_rotation,out_translation,cv::noArray(),cv::noArray(),cv::noArray(),euler_angle);
	cv::Vec3f eav;
	for(int i = 0; i < 3; i++){
		eav[i] = euler_angle.at<double>(0,i);
	}
	drawEAV(img,eav);
	return eav;
}

int testcamera(int cameraindex = 0)
{
    PoseEstimator pe;
    MTCNN detector(proto_model_dir);
    cv::VideoCapture cap(cameraindex);
    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<FaceInfo> faceInfo;
        TickMeter tm;
        tm.start();
        detector.Detect(frame, faceInfo);
        tm.stop();
        cout << tm.getTimeMilli() << "ms" << endl;
        for (int i = 0; i < faceInfo.size(); i++)
        {
            vector<Point2f > imagePoints;
            auto fi = faceInfo[0];
            for (int i = 0; i < 5; i++){
                imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
            }
            auto eav=pe.estimateHeadPose(frame, imagePoints);
        }
        MTCNN::drawDection(frame, faceInfo);
        cv::imshow("img", frame);
        cv::waitKey(1);
    }
    return 0;
}

int testdir()
{
	MTCNN detector(proto_model_dir);
    PoseEstimator pe;
	vector<string>files=getAllFilesinDir(imgdir);
	cv::Mat frame;
	for (int i = 0; i < files.size(); i++)
	{
		string imageName = imgdir + "/" + files[i];
        std::cout << files[i];
		frame=cv::imread(imageName);
        if(!frame.data)
            continue;
		clock_t t1 = clock();
		std::vector<FaceInfo> faceInfo;
		detector.Detect(frame, faceInfo);
		std::cout << " : " << (clock() - t1)*1.0 / 1000 << std::endl;
		vector<Mat> alignehdfaces = Align5points(frame,faceInfo);
		for (int j = 0; j < alignehdfaces.size(); j++)
		{
			string alignpath="align/"+int2string(j)+"_"+files[i];
			imwrite(alignpath, alignehdfaces[j]);
		}
        vector<Point2f > imagePoints;
        auto fi = faceInfo[0];
        for (int i = 0; i < 5; i++)
        {
            imagePoints.push_back(cv::Point2f(fi.facePts.y[i], fi.facePts.x[i]));
        }
        pe.estimateHeadPose(frame, imagePoints);
		MTCNN::drawDection(frame,faceInfo);
		cv::imshow("img", frame);
		string resultpath = resultdir + "/"+files[i];
		cv::imwrite(resultpath, frame);
		cv::waitKey(1);
	}
	cv::waitKey();
	return 0;
}

int eval_fddb()
{
	const char* fddb_dir = "E:/Face/Datasets/fddb";
	string format = fddb_dir + string("/MTCNN/%Y%m%d-%H%M%S");
	time_t t = time(NULL);
	char buff[300];
	strftime(buff, sizeof(buff), format.c_str(), localtime(&t));
	MKDIR(buff);
	string result_prefix(buff);
	string prefix = fddb_dir + string("/images/");
	MTCNN detector(proto_model_dir);
	int counter = 0;
//#pragma omp parallel for
	for (int i = 1; i <= 10; i++) 
	{
		char fddb[300];
		char fddb_out[300];
		char fddb_answer[300];
		cout<<"Folds: "<<i<<endl;
		sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir, i);
		sprintf(fddb_out, "%s/MTCNN/fold-%02d-out.txt", fddb_dir, i);
		sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt", fddb_dir, i);

		FILE* fin = fopen(fddb, "r");
		FILE* fanswer = fopen(fddb_answer, "r");
#ifdef _WIN32
		FILE* fout = fopen(fddb_out, "wb"); // replace \r\n on Windows platform		
#else
		FILE* fout = fopen(fddb_out, "w");	
#endif // WIN32
		
		char path[300];
		int counter = 0;
		while (fscanf(fin, "%s", path) > 0)
		{
			string full_path = prefix + string(path) + string(".jpg");
			Mat img = imread(full_path);
			if (!img.data) {
				cout << "Cannot read " << full_path << endl;;
				continue;
			}
			clock_t t1 = clock();
			std::vector<FaceInfo> faceInfo;
			detector.Detect(img, faceInfo);
			std::cout << "Detect " <<i<<": "<<counter<<" Using : " << (clock() - t1)*1.0 / 1000 << std::endl;
			const int n = faceInfo.size();
			fprintf(fout, "%s\n%d\n", path, n);
			for (int j = 0; j < n; j++) {
				int x = (int)faceInfo[j].bbox.x1;
				if (x < 0)x = 0;
				int y = (int)faceInfo[j].bbox.y1;
				if (y < 0)y = 0;
				int h = (int)faceInfo[j].bbox.x2 - faceInfo[j].bbox.x1 + 1;
				if (h>img.rows - x)h = img.rows - x;
				int w = (int)faceInfo[j].bbox.y2 - faceInfo[j].bbox.y1 + 1;
				if (w>img.cols-y)w = img.cols - y;
				float score = faceInfo[j].bbox.score;
				cv::rectangle(img, cv::Rect(y, x, w, h), cv::Scalar(0, 0, 255), 1);
				fprintf(fout, "%d %d %d %d %lf\n", y, x, w, h, score);
			}
			for (int t = 0; t < faceInfo.size(); t++){
				FacePts facePts = faceInfo[t].facePts;
				for (int j = 0; j < 5; j++)
					cv::circle(img, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
			}
			cv::imshow("img", img);
			cv::waitKey(1);
			char buff[300];
			if (1) {
				counter++;
				sprintf(buff, "%s/%02d_%03d.jpg", result_prefix.c_str(), i, counter);
				// get answer
				int face_n = 0;
				fscanf(fanswer, "%s", path);
				fscanf(fanswer, "%d", &face_n);
				for (int k = 0; k < face_n; k++)
				{
					double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
					fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius, &minor_axis_radius, \
						&angle, &center_x, &center_y, &score);
					// draw answer
					angle = angle / 3.1415926*180.;
					cv::ellipse(img, cv::Point2d(center_x, center_y), cv::Size(major_axis_radius, minor_axis_radius), \
						angle, 0., 360., Scalar(255, 0, 0), 2);					
				}
				cv::imwrite(buff, img);
			}
		}
		fclose(fin);
		fclose(fout);
		fclose(fanswer);
	}
	return 0;
}

int main(int argc, char **argv)
{
    testcamera();
//	testdir();
//	eval_fddb();
}