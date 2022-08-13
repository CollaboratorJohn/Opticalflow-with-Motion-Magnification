#include <opencv2/opencv.hpp>

#include <iostream>
#include <ctime>

int dynamicPix(cv::Mat& src,int threshold)
{
    int dynamic=0;
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            dynamic+=(src.at<int>(i,j)>threshold);
        }
    }
    return dynamic;
}

//绘制直方图，src为输入的图像，histImage为输出的直方图，name是输出直方图的窗口名称
float drawHistImg(cv::Mat &src, cv::Mat &histImage,std::string name, int th)
{
	const int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::MatND hist;
	int channels[] = { 0 };
 
	cv::calcHist(&src, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
 
	double maxValue;
	cv::minMaxLoc(hist, 0, &maxValue, 0, 0);
	int scale = 1;
	int histHeight = 256;
    float binValue=0;
    for(int i=0;i<256;i++)
    {
            std::cout<<hist.at<float>(i)<<" ";
    }
    std::cout<<std::endl;
	for (int i = th; i < bins; i++)
	{
		binValue += hist.at<float>(i);
	}
    return binValue/ maxValue;
}

int main()
{
    cv::VideoCapture capture("./1.mp4");
    if (!capture.isOpened())
        return -1;
    double rate = capture.get(cv::CAP_PROP_FPS);//获取视频帧率
    int delay = 1000 / rate;
    cv::Mat framepro, frame, dframe;
    bool flag = false;
    clock_t start,end;
    while (capture.read(frame))
    {
        //将第一帧图像拷贝给framePro
        if (false == flag)
        {
            framepro = frame.clone();
            flag = true;
        }
        else
        {
            start =clock();
            absdiff(frame, framepro, dframe);//帧间差分计算两幅图像各个通道的相对应元素的差的绝对值。
            //end =clock();
            //std::cout<<(double)(end-start)/CLOCKS_PER_SEC<<std::endl;
            std::cout<<std::endl;
            framepro = frame.clone();//将当前帧拷贝给framepro
            cvtColor(dframe,dframe,cv::COLOR_BGR2GRAY);
            //threshold(dframe, dframe, 80, 255, CV_THRESH_BINARY);//阈值分割
            cv::imshow("image", frame);
            cv::imshow("test", dframe);
            
	        cv::Mat dframeImage = cv::Mat::ones(256, 256, CV_8UC3)*255;
            //start =clock();
	        //std::cout<<dynamicPix(dframe,20)<<std::endl;
            std::cout<<float(drawHistImg(dframe, dframeImage,"dframeImage",5))<<std::endl;
            //end =clock();
            //std::cout<<(double)(end-start)/CLOCKS_PER_SEC<<std::endl;
            cv::waitKey(delay);
        }
    }

    return 0;
}
