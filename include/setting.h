#ifndef _PRINT_METHOD_H_
#define _PRINE_METHOD_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <regex>
#include <iostream>
#include <tuple>

//define basic settings
class settings
{
protected:
    int w;
    int h;
	int levels;
	double alpha;
	double lambda_c;
	double cutoff_frequency_high;
	double cutoff_frequency_low;
	double chrom_attenuation;
	double exaggeration_factor;

	std::string filename;
public:
    settings();
    settings(std::string s);
    settings(int argc, char** argv);
    //settings(int argc, char** argv, cv::VideoCapture capture);
    inline int getW() const;
    inline int getH() const;
    inline int getLevels() const;
    inline double getAlpha() const;
    inline double getLambda_c() const;
    inline double getCutoff_frequency_high() const;
    inline double getCutoff_frequency_low() const;
    inline double getChrom_attenuation() const;
    inline double getExaggeration_factor() const;
    inline std::string getFilename() const;
    inline cv::VideoCapture getCapture() const;

};
std::ostream& operator<<(std::ostream& strm, const settings& s); //output stream redefine

inline int settings::getW() const
{
	return this->w;
}

inline int settings::getH() const
{
	return this->h;
}

inline int settings::getLevels() const
{
    return this->levels;
}

inline double settings::getAlpha() const
{
    return this->alpha;
}

inline double settings::getLambda_c() const
{
    return this->lambda_c;
}

inline double settings::getCutoff_frequency_high() const
{
	return this->cutoff_frequency_high;
}

inline double settings::getCutoff_frequency_low() const
{
    return this->cutoff_frequency_low;
}

inline double settings::getChrom_attenuation() const
{
    return this->chrom_attenuation;
}

inline double settings::getExaggeration_factor() const
{
	return this->exaggeration_factor;
}

inline std::string settings::getFilename() const
{
	return this->filename;
}

inline cv::VideoCapture settings::getCapture() const
{
    cv::VideoCapture capture(this->getFilename());
    return capture;
}


#endif