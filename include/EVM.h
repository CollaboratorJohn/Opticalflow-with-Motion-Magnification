#ifndef _EVM_H_
#define _EVM_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>
#include <condition_variable>
#include <list>
#include <vector>
#include <memory>
#include <thread>
#include <cmath>
#include "../include/outputframe.h"
#include "../include/getframe.h"

class EVM
{
private:
    std::vector<double> factor;
	int frame_num;
    std::vector<cv::Mat> low_pass1_1;
	std::vector<cv::Mat> low_pass1_2;
	std::vector<cv::Mat> filtered1;

    std::vector<cv::Mat> low_pass2_1;
	std::vector<cv::Mat> low_pass2_2;
	std::vector<cv::Mat> filtered2;

    static EVM* instance;
    static std::mutex mtx;
    EVM():output_complete(false){};

public:
	bool output_complete;
	std::mutex output_mutex;
	std::condition_variable output_notifier;

	class CGarbo {
	public:
		~CGarbo(){
			if (EVM::instance)
				delete EVM::instance;
		}
	};
    static CGarbo Garbo;

public:
    static EVM* getInstance();

    double drawHistImg(cv::Mat &src, int th);

    void calcDiff(std::vector<cv::Mat> img1, std::vector<cv::Mat> img2);

    void calcEVM(cv::Mat input,
    std::vector<cv::Mat> pyramid,
    std::vector<cv::Mat>& low_pass1,
    std::vector<cv::Mat>& low_pass2,
    std::vector<cv::Mat>& filtered,
    cv::Mat& motion,
    settings setting
    );

    void calculatePair(std::list<std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>>>& input_queue,
	std::list<std::tuple<cv::Mat, cv::Mat> >& output_queue,
    inputFrame* input_instance,
    outFrame* output_instance,
    settings setting);
};

#endif