#include "../include/getframe.h"

inputFrame* inputFrame::instance=new inputFrame();
std::mutex inputFrame::mtx;

inputFrame* inputFrame::getInstance()
{
    if(inputFrame::instance==nullptr)
    {
        inputFrame::mtx.lock();
        if(inputFrame::instance==nullptr)
            inputFrame::instance=new inputFrame();
        mtx.unlock();
    }
    return inputFrame::instance;
}

//输入图像预处理，做成金字塔塞到队列中
void inputFrame::getFrame(cv::VideoCapture capture,
std::list<std::vector<cv::Mat> >& input_queue)
{
	cv::Mat frame;
	while (true)
	{
        //已经全部读完了
        if (!capture.read(frame))
        {
            this->input_complete = true;
            this->input_notifier.notify_all();
            return;
        }

        auto input = frame.clone();

        //转为lab颜色空间
        input.convertTo(input, CV_32FC3, 1.0 / 255.0f);
        cv::cvtColor(input, input, cv::COLOR_BGR2Lab);

        //金字塔分解
        std::vector<cv::Mat> pyramid;
        {
            auto current = input;
            for (int l = 0; l < this->levels; l++)
            {
                cv::Mat down, up;

                pyrDown(current, down);
                pyrUp(down, up, current.size());

                pyramid.push_back(current - up);
                current = down;
            }

            pyramid.push_back(current);
            pyramid.push_back(input);
        }

        // Add to input queue
        std::unique_lock<std::mutex> lock(input_mutex);
        input_queue.push_back(pyramid);
        this->input_notifier.notify_one();
    }
}

//金字塔分解子线程函数
void inputFrame::pryDecompose(cv::Mat& input,std::vector<cv::Mat>& pyramid)
{
    auto current = input.clone();
    for (int l = 0; l < this->levels; l++)
    {
        cv::Mat down, up;

        pyrDown(current, down);
        pyrUp(down, up, current.size());

        pyramid.push_back(current - up);
        current = down;
    }

    pyramid.push_back(current);
    pyramid.push_back(input);
}

// 一对一对读入图像
void inputFrame::getFramePair(cv::VideoCapture capture,
std::list<std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>>>& input_queue)
{
	cv::Mat frame,input1,input2;
    capture.read(frame);
    input1 = frame.clone();
    input1.convertTo(input1, CV_32FC3, 1.0 / 255.0f);
    cv::cvtColor(input1, input1, cv::COLOR_BGR2Lab);
	while (true)
	{
        // 已经全部读完了
        if (!capture.read(frame))
        {
            this->input_complete = true;
            this->input_notifier.notify_all();
            return;
        }
		input2 = frame.clone();
        input2.convertTo(input2, CV_32FC3, 1.0 / 255.0f);
        cv::cvtColor(input2, input2, cv::COLOR_BGR2Lab);
        // 多线程金字塔分解
        std::vector<cv::Mat> pyramid1;
        std::vector<cv::Mat> pyramid2;
        auto pyrDecompose1=std::thread(&inputFrame::pryDecompose,inputFrame::getInstance(),
            std::ref(input1),std::ref(pyramid1));
        auto pyrDecompose2=std::thread(&inputFrame::pryDecompose,inputFrame::getInstance(),
            std::ref(input2),std::ref(pyramid2));
        pyrDecompose1.join();
        pyrDecompose2.join();
        input1 = input2.clone();

        // 把一对加入到队列
        std::unique_lock<std::mutex> lock(input_mutex);
        input_queue.push_back(std::make_tuple(pyramid1,pyramid2));
        this->input_notifier.notify_one();
    }
}