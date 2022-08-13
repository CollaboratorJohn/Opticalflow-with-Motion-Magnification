#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <list>
#include <mutex>
#include <condition_variable>
#include <cmath>

#include "../include/outputframe.h"
#include "../include/getframe.h"
#include "../include/EVM.h"
#include "../include/flow.h"

int main(int argc, char **argv)
{
	settings settings(argc,argv);
	cv::VideoCapture capture(settings.getFilename());
	// 输出相关信息
	std::cout << settings;

	// 第一级队列，将输入图像转换为六层金字塔
	std::list<std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>>> input_queue_pair;

	//第二级队列，输出为tuple封装的两张lab颜色图片
	std::list<std::tuple<cv::Mat, cv::Mat> > lab_queue;

	//第三级队列，输出为tuple封装的两张rgb颜色图片
	std::list<std::tuple<cv::Mat, cv::Mat> > rgb_queue;

	// 将视频中的图片放入队列线程
	auto input_thread_pair = std::thread(&inputFrame::getFramePair,inputFrame::getInstance(),
	capture,std::ref(input_queue_pair));

	// EVM运算线程，将运算好的一对lab图片放入队列
	auto evm_processor_pair = std::thread(&EVM::calculatePair,EVM::getInstance(),
	std::ref(input_queue_pair),
	std::ref(lab_queue),
	inputFrame::getInstance(),
	outFrame::getInstance(),
    settings
	);
	
	// 将lab图片转换为rgb线程
	auto output_thread_pair = std::thread(&outFrame::putFrame,outFrame::getInstance(),
	std::ref(lab_queue),
	std::ref(rgb_queue));

	//从输出队列中拿图片进行处理
	auto flow_process = std::thread(&flow::flowCalc,flow::getInstance(),
	std::ref(rgb_queue));
	
	evm_processor_pair.join();
	input_thread_pair.join();
	output_thread_pair.join();
	flow_process.join();
	return 0;
}
