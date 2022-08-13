
#include "../include/EVM.h"

EVM* EVM::instance=new EVM();
std::mutex EVM::mtx;

EVM* EVM::getInstance()
{
    if(EVM::instance==nullptr)
    {
        EVM::mtx.lock();
        if(EVM::instance==nullptr)
            EVM::instance=new EVM();
        mtx.unlock();
    }
    return EVM::instance;
}

//返回直方图帧差，灰度差在th以外的比率
double EVM::drawHistImg(cv::Mat &src, int th)
{
	auto current=src.clone();
	cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
	const int bins = 256;
	int hist_size[] = { bins };
	float range[] = { 0, 256 };
	const float* ranges[] = { range };
	cv::MatND hist;
	int channels[] = { 0 };
 
	cv::calcHist(&current, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
 
    double binValue=0;
	for (int i = th; i < bins; i++)
	{
		binValue += hist.at<float>(i) ;
	}
	
    return binValue / src.rows / src.cols; //返回
}

//返回帧差
void EVM::calcDiff(std::vector<cv::Mat> img1, std::vector<cv::Mat> img2)
{
	this->factor.clear();
    //预处理图像
    for(int i = 0; i < img1.size(); i++)
    {
        cv::Mat dframe;
		img1[i].convertTo(img1[i], CV_8UC3, 255.0, 1.0 / 255.0);
		img2[i].convertTo(img2[i], CV_8UC3, 255.0, 1.0 / 255.0);
        absdiff(img1[i], img2[i], dframe);
		this->factor.push_back(sqrt(sqrt(1 - drawHistImg(dframe,5))));
    }
}
void EVM::calcEVM(
	cv::Mat input,
	std::vector<cv::Mat> pyramid,
	std::vector<cv::Mat>& low_pass1,
	std::vector<cv::Mat>& low_pass2,
	std::vector<cv::Mat>& filtered,
	cv::Mat& motion,
	settings setting

)
{
	if (frame_num == 0)
	{
		for (int l = 0; l < pyramid.size(); l++)
		{
			filtered.push_back(pyramid[l].clone());
			low_pass1.push_back(pyramid[l].clone());
			low_pass2.push_back(pyramid[l].clone());
		}
	}
	if (frame_num > 0)
	{
		auto delta = setting.getLambda_c() / 8.0 / (1.0 + setting.getAlpha());
		auto lambda = sqrt((float)(setting.getW() * setting.getW() + setting.getH() * setting.getH())) / 3;
		// 计算金字塔的级数
		std::vector<std::thread> workers;
		for (int level = setting.getLevels(); level >= 0; level--)
		{
			workers.push_back(std::thread([&, level, lambda]() {
				// First or last level we mostly ignore
				if (level == setting.getLevels() || level == 0)
				{
					filtered[level] *= 0;
					return;
				}

				// Temporal IIR Filter
				low_pass1[level] = (1 - setting.getCutoff_frequency_high()) * low_pass1[level] + 
									setting.getCutoff_frequency_high() * pyramid[level];
				low_pass2[level] = (1 - setting.getCutoff_frequency_low()) * low_pass2[level] + 
									setting.getCutoff_frequency_low() * pyramid[level];
				filtered[level] = low_pass1[level] - low_pass2[level];

				// Amplify
				double current_alpha = (lambda / delta / 8 - 1) * setting.getExaggeration_factor();
				
				filtered[level] *= std::min(setting.getAlpha(), current_alpha * this->factor[level]);
			}));

			lambda /= 2.0;
		}

		std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
			t.join();
		});
	}

	// 重新构建金字塔
	{
		auto current = filtered[setting.getLevels()];
		for (int level = setting.getLevels() - 1; level >= 0; --level)
		{
			cv::Mat up;
			pyrUp(current, up, filtered[level].size());
			current = up + filtered[level];
		}
		motion = current;
	}

	if (frame_num > 0)
	{
		// 对IQ通道进行增强
		cv::Mat planes[3];
		split(motion, planes);
		planes[1] = planes[1] * setting.getChrom_attenuation();
		planes[2] = planes[2] * setting.getChrom_attenuation();
		cv::merge(planes, 3, motion);

		// 将原图加上动作
		motion = input + motion;
	}
}

//EVM运算，针对多张图
void EVM::calculatePair(
    std::list<std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>>>& input_queue,
    std::list<std::tuple<cv::Mat, cv::Mat> >& output_queue,
    inputFrame* input_instance,
    outFrame* output_instance,
    settings setting
)
{
    frame_num = 0;
	while (true)
	{
		std::tuple<std::vector<cv::Mat>,std::vector<cv::Mat>> pyramid;
		{
			std::unique_lock<std::mutex> lock(input_instance->input_mutex);
			input_instance->input_notifier.wait(lock, [&]() {
				return input_instance->input_complete ||
				 !input_queue.empty();
			});
			if (input_instance->input_complete && 
			input_queue.empty())
			{
				std::unique_lock<std::mutex> lock(output_instance->output_mutex);
				output_instance->output_complete = true;
				output_instance->output_notifier.notify_all();
				break;
			}

			pyramid = input_queue.front();
			input_queue.pop_front();
		}

		// 获取原始输入的两张图像
		cv::Mat input0 = std::get<0>(pyramid)[setting.getLevels() + 1];
		cv::Mat input1 = std::get<1>(pyramid)[setting.getLevels() + 1];
		cv::Mat motion0, motion1;

		// 计算两图之间的帧差，指示放大因子
        auto calcDiffThread = std::thread(&EVM::calcDiff,EVM::getInstance(),
            std::get<0>(pyramid),std::get<1>(pyramid));

		auto calcEVM0=std::thread(&EVM::calcEVM,EVM::getInstance(),
            input0,
			std::get<0>(pyramid),
			std::ref(low_pass1_1),
			std::ref(low_pass1_2),
			std::ref(filtered1),
			std::ref(motion0),
			setting
			);

		auto calcEVM1=std::thread(&EVM::calcEVM,EVM::getInstance(),
            input1,
			std::get<1>(pyramid),
			std::ref(low_pass2_1),
			std::ref(low_pass2_2),
			std::ref(filtered2),
			std::ref(motion1),
			setting
			);

        calcEVM0.join();
        calcEVM1.join();

		// 帧计数+1
		frame_num++;

		//加入到输出队列
		output_queue.push_back(std::tuple<cv::Mat, cv::Mat>(motion0,motion1));
		output_instance->output_notifier.notify_one();
        calcDiffThread.join();
	}

}
