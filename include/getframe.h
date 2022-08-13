#ifndef _GET_FRAME_H_
#define _GET_FRAME_H_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <mutex>
#include <condition_variable>
#include <list>
#include <vector>
#include <memory>
#include <thread>

#include "../include/setting.h"

class inputFrame: protected settings
{
private:
    static inputFrame* instance;
    static std::mutex mtx;
    inputFrame():input_complete(false){};

public:
    bool input_complete;
    std::mutex input_mutex;
	std::condition_variable input_notifier;

	class CGarbo {
	public:
		~CGarbo(){
			if (inputFrame::instance)
				delete inputFrame::instance;
		}
	};
    static CGarbo Garbo;

public:
    static inputFrame* getInstance();
    void getFrame(cv::VideoCapture capture,std::list<std::vector<cv::Mat>>& input_queue);
	void pryDecompose(cv::Mat& input,std::vector<cv::Mat>& pyramid);
	void getFramePair(cv::VideoCapture capture,
				std::list<std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>>>& input_queue);
};


#endif