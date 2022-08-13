#include "../include/outputframe.h"
#include <iostream>
#include <ctime>

outFrame* outFrame::instance=new outFrame();
std::mutex outFrame::mtx;

outFrame* outFrame::getInstance()
{
    if(outFrame::instance==nullptr)
    {
        outFrame::mtx.lock();
        if(outFrame::instance==nullptr)
            outFrame::instance=new outFrame();
        mtx.unlock();
    }
    return outFrame::instance;
}

void outFrame::putFrame(std::list<std::tuple<cv::Mat, cv::Mat> >& lab_queue,
std::list<std::tuple<cv::Mat, cv::Mat> >& rgb_queue)
{
    std::tuple<cv::Mat, cv::Mat> output;
    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(this->output_mutex);
            this->output_notifier.wait(lock, [&]() {
                return this->output_complete || !lab_queue.empty();
            });
            if (this->output_complete && lab_queue.empty())
            {
                return;
            }
            output = lab_queue.front();
            lab_queue.pop_front();
        }
        auto amplified0 = std::get<0>(output);
        auto amplified1 = std::get<1>(output);

        // 将图像由YIQ转换为RGB
        cv::cvtColor(amplified0, amplified0, cv::COLOR_Lab2BGR);
        amplified0.convertTo(amplified0, CV_8UC3, 255.0, 1.0 / 255.0);

        cv::cvtColor(amplified1, amplified1, cv::COLOR_Lab2BGR);
        amplified1.convertTo(amplified1, CV_8UC3, 255.0, 1.0 / 255.0);

        rgb_queue.push_back(std::make_tuple(amplified0,amplified1));
        
        // 可视化
        /*
        cv::Mat concat;
        cv::vconcat(amplified0,amplified1,concat);
        cv::imshow("Input", concat);
        char chr = (char)cv::waitKey(1);*/
    }
}