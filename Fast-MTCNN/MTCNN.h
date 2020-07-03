#pragma once
#include "string"
#include "opencv2/opencv.hpp"

typedef struct FaceBox {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
} FaceBox;
typedef struct FaceInfo {
	float bbox_reg[4];
	float landmark_reg[10];
	float landmark[10];
	FaceBox bbox;
} FaceInfo;

class MTCNN {
public:
	MTCNN(const std::string& proto_model_dir);
	std::vector<FaceInfo> Detect_mtcnn(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
	//protected:
	std::vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
	std::vector<FaceInfo> NextStage(const cv::Mat& image, std::vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
	void BBoxRegression(std::vector<FaceInfo>& bboxes);
	void BBoxPadSquare(std::vector<FaceInfo>& bboxes, int width, int height);
	void BBoxPad(std::vector<FaceInfo>& bboxes, int width, int height);
	void GenerateBBox(cv::Mat* confidence, cv::Mat* reg_box, float scale, float thresh);
	std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);
public:
	cv::dnn::Net PNet_;
	cv::dnn::Net RNet_;
	cv::dnn::Net ONet_;

	std::vector<FaceInfo> candidate_boxes_;
	std::vector<FaceInfo> total_boxes_;
};