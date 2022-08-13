#include <string>

#include "../include/setting.h"

settings::settings(std::string s):levels(5),
alpha(20.0),
lambda_c(20.0),
cutoff_frequency_high(0.4),
cutoff_frequency_low(0.05),
chrom_attenuation(0.1),
exaggeration_factor(2.0)
{
	this->filename=s;
};

settings::settings():levels(5),
alpha(20.0),
lambda_c(20.0),
cutoff_frequency_high(0.4),
cutoff_frequency_low(0.05),
chrom_attenuation(0.1),
exaggeration_factor(2.0)
{}

settings::settings(int argc, char** argv):levels(5),
alpha(20.0),
lambda_c(20.0),
cutoff_frequency_high(0.4),
cutoff_frequency_low(0.05),
chrom_attenuation(0.1),
exaggeration_factor(2.0)
{
    //console inputs to modify parameters
	const std::tuple<std::regex, std::function<void(std::smatch)>> options[] = {
		{std::regex("^levels=(\\d+)$"), [&](std::smatch m) { this->levels = std::stoi(m.str(1), nullptr); }},
		{std::regex("^alpha=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) { this->alpha = std::stod(m.str(1), nullptr); }},
		{std::regex("^cutoff_frequency_low=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) { this->cutoff_frequency_low = std::stod(m.str(1), nullptr); }},
		{std::regex("^cutoff_frequency_high=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) {this->cutoff_frequency_high = std::stod(m.str(1), nullptr); }},
		{std::regex("^lambda_c=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) { this->lambda_c = std::stod(m.str(1), nullptr); }},
		{std::regex("^chrom_attenuation=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) { this->chrom_attenuation = std::stod(m.str(1), nullptr); }},
		{std::regex("^exaggeration_factor=([+-]?((\\d+(\\.\\d*)?)|(\\.\\d+)))$"), [&](std::smatch m) { this->exaggeration_factor = std::stod(m.str(1), nullptr); }},
		{std::regex("^(.*)$"), [&](std::smatch m) { this->filename = m.str(1); }},
	};
    // Parse options/settings
	for (int i = 1; i < argc; i++)
	{
		for (auto &option : options)
		{
			std::string str = argv[i];
			std::smatch m;
			std::regex_match(str, m, std::get<0>(option));
			if (m.size())
			{
				std::get<1>(option)(m);
			}
		}
	}
	this->w=this->getCapture().get(cv::CAP_PROP_FRAME_WIDTH);
	this->h=this->getCapture().get(cv::CAP_PROP_FRAME_HEIGHT);
}

std::ostream& operator<<(std::ostream& strm, const settings& s)
{
	strm<< "levels: " << s.getLevels() << std::endl
	<< "alpha: " << s.getAlpha() << std::endl
	<< "lambda_c: " << s.getLambda_c() << std::endl
	<< "cutoff_frequency_high: " << s.getCutoff_frequency_high() << std::endl
	<< "cutoff_frequency_low: " << s.getCutoff_frequency_low() << std::endl
	<< "chrom_attenuation: " << s.getChrom_attenuation() << std::endl
	<< "exaggeration_factor: " << s.getExaggeration_factor() << std::endl
	<< "filename: " << s.getFilename() << std::endl
	<< "frame height: "<< s.getH()<< std::endl
	<< "frame width: "<< s.getW()<<std::endl;
	return strm;
}
