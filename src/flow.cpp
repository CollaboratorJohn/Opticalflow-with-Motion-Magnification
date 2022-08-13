#include "../include/flow.h"

#define UNKNOWN_THRESH  1e9
#define EPS 1e-10
#define pi 3.1415926

flow* flow::instance=new flow();
std::mutex flow::mtx;

flow::flow()
{
	model = torch::jit::load("./model/model.pt");
	model.to(at::kCUDA);
}

flow* flow::getInstance()
{
    if(flow::instance==nullptr)
    {
        flow::mtx.lock();
        if(flow::instance==nullptr)
            flow::instance=new flow();
        mtx.unlock();
    }
    return flow::instance;
}

at::Tensor flow::ToTensor(cv::Mat img)
{
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    at::Tensor tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::TensorOptions().dtype(torch::kByte)).to(at::kCUDA);
	tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.toType(torch::kFloat);
	return tensor_image;

}

std::vector<torch::jit::IValue> flow::ToInput(at::Tensor tensor_image1, at::Tensor tensor_image2)
{
    return std::vector<torch::jit::IValue>{tensor_image1,tensor_image2};
}

cv::Mat flow::ToCVType(torch::Tensor flow_tensor)
{
	flow_tensor = flow_tensor.squeeze().permute({ 1,2,0 }).toType(torch::kF32).contiguous();
	flow_tensor = flow_tensor.to(torch::kCPU); 

	cv::Mat resultImg(360, 640, CV_32FC2, flow_tensor.data_ptr<float>()); 
	return resultImg;
}

void flow::flowVis(cv::Mat& flo, cv::Mat& img)
{
	float colorwheel[55][3]={0};
	unsigned short RY = 15;
	unsigned short YG = 6;
	unsigned short GC = 4;
	unsigned short CB = 11;
	unsigned short BM = 13;
	unsigned short MR = 6;
	unsigned char ncols= RY + YG + GC + CB + BM + MR;
	unsigned short nchans = 3;
	unsigned short col = 0;
	//RY
	for (int i = 0; i<RY; i++)
	{
		colorwheel[col + i][0] = 255;
		colorwheel[col + i][1] = 255 * i / RY;
		colorwheel[col + i][2] = 0;
	}
	col += RY;
	//YG
	for (int i = 0; i<YG; i++)
	{
		colorwheel[col + i][0] = 255 - 255 * i / YG;
		colorwheel[col + i][1] = 255;
		colorwheel[col + i][2] = 0;
	}
	col += YG;
	//GC
	for (int i = 0; i < GC; i++)
	{
		colorwheel[col + i][1] = 255;
		colorwheel[col + i][2] = 255 * i / GC;
		colorwheel[col + i][0] = 0;
	}
	col += GC;
	//CB
	for (int i = 0; i < CB; i++)
	{
		colorwheel[col + i][1] = 255 - 255 * i / CB;
		colorwheel[col + i][2] = 255;
		colorwheel[col + i][0] = 0;
	}
	col += CB;
	//BM
	for (int i = 0; i < BM; i++)
	{
		colorwheel[col + i][2] = 255;
		colorwheel[col + i][0] = 255 * i / BM;
		colorwheel[col + i][1] = 0;
	}
	col += BM;
	//MR
	for (int i = 0; i < MR; i++)
	{
		colorwheel[col + i][2] = 255 - 255 * i / MR;
		colorwheel[col + i][0] = 255;
		colorwheel[col + i][1] = 0;
	}

	int row = flo.rows;
	int cols = flo.cols;
	float max_norm = 1e-3;

	//calculate the rgb value
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			float* data = flo.ptr<float>(i, j);
			unsigned char* img_data = img.ptr<unsigned char>(i, j);
			float u = data[0];
			float v = data[1];
			if(isnan(u) || isnan(v))
			{
				u=0;v=0;
			}
			float norm = sqrt(u*u + v*v);
			float angle = atan2(-v,-u) / pi;
			float fk = (angle + 1) / 2 * (float(ncols) - 1);
			int k0 = (int)floor(fk);
			int k1 = k0 + 1;
			if (k1 == ncols) {
				k1 = 0;
			}
			float f = fk - k0;
			
			for (int k = 0; k < 3; k++) {
				float col0 = (colorwheel[k0][k] / 255);
				float col1 = (colorwheel[k1][k] / 255);
				float col3 = (1 - f)*col0 + f*col1;
				if (norm <= 1) {
					col3 = 1 - norm*(1 - col3);
				}
				else {
					col3 *= 0.75;
				}
				img_data[k] = (unsigned char)(255 * col3);
			}
		}
	}

}

void flow::flowCalc(std::list<std::tuple<cv::Mat, cv::Mat> >& raw_img_queue)
{
	cv::Mat florgb(360,640,CV_8UC3);
	while(true)
	{
		if(raw_img_queue.empty())
		{
			std::cout<<"empty!"<<std::endl;
			continue;
		}
		std::tuple<cv::Mat,cv::Mat> raw_imgs = raw_img_queue.front();
		raw_img_queue.pop_front();
		auto t1 = ToTensor(std::get<0>(raw_imgs));
		auto t2 = ToTensor(std::get<1>(raw_imgs));
		torch::Tensor flow=model.forward(ToInput(t1,t2)).toTensor();
		auto flow_mat = ToCVType(flow);
		flowVis(flow_mat, florgb);
		cv::imshow("flow", florgb);
		cv::waitKey(1);
	}
}

//change flow tensor to visible images
void flow::flo2img(cv::Mat& flow, cv::Mat& bgrimg)
{
	cv::Mat magnitude, normalized_magnitude, angle; 
	cv::Mat hsv[3], merged_hsv, hsv_8u;

	cv::Mat flow_xy[2], flow_x, flow_y; 
	split(flow, flow_xy); 

	flow_x = flow_xy[0]; 
	flow_y = flow_xy[1];

	// convert from cartesian to polar coordinates 
	cv::cartToPolar(flow_xy[0], flow_xy[1], magnitude, angle, true); 
	
	// normalize magnitude from 0 to 1 
	cv::normalize(magnitude, normalized_magnitude, 0.0, 1.0, cv::NORM_MINMAX); 
	
	// get angle of optical flow 
	angle *= ((1 / 360.0) * (180 / 255.0)); 
	
	// build hsv image 
	hsv[0] = angle; 
	hsv[1] = cv::Mat::ones(360, 640, CV_32F);
	hsv[2] = normalized_magnitude; 
	cv::merge(hsv, 3, merged_hsv); 
	
	// multiply each pixel value to 255 
	merged_hsv.convertTo(hsv_8u, CV_8U, 255.0); 
	
	// convert hsv to bgr 
	cv::cvtColor(hsv_8u, bgrimg, cv::COLOR_HSV2BGR); 
}

