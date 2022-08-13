#ifndef _FLOW_H_
#define _FLOW_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

#include <mutex>
#include <condition_variable>
#include <list>
#include <vector>
#include <memory>
#include <thread>
#include <cmath>

#define UNKNOWN_THRESH  1e9
#define EPS 1e-10
#define pi 3.1415926

class flow
{
private:
    static flow* instance;
    static std::mutex mtx;
    torch::jit::script::Module model;
public:
	class CGarbo {
	public:
		~CGarbo(){
			if (flow::instance)
				delete flow::instance;
		}
	};
    static CGarbo Garbo;

    flow();
    void flowCalc(std::list<std::tuple<cv::Mat, cv::Mat> >& raw_img_queue);
    at::Tensor ToTensor(cv::Mat img);
    std::vector<torch::jit::IValue> ToInput(at::Tensor tensor_image1, at::Tensor tensor_image2);
    cv::Mat ToCVType(torch::Tensor flow_tensor);
    void flowVis(cv::Mat& flo, cv::Mat& img);
    void flo2img(cv::Mat& flow, cv::Mat& bgrimg);

public:
    static flow* getInstance();
};

#endif