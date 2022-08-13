#ifndef _OUTPUT_FRAME_H_
#define _OUTPUT_FRAME_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <mutex>
#include <condition_variable>
#include <list>
#include <vector>
#include <memory>
#include <thread>

class outFrame
{
private:
    static outFrame* instance;
    static std::mutex mtx;
    outFrame():output_complete(false){};

public:
	bool output_complete;
	std::mutex output_mutex;
	std::condition_variable output_notifier;

	class CGarbo {
	public:
		~CGarbo(){
			if (outFrame::instance)
				delete outFrame::instance;
		}
	};
    static CGarbo Garbo;

public:
    static outFrame* getInstance();
    void putFrame(std::list<std::tuple<cv::Mat, cv::Mat> >& lab_queue,
			std::list<std::tuple<cv::Mat, cv::Mat> >& rgb_queue);
};

#endif