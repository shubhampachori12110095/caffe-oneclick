#include "LenetClassifier.h"
std::pair<int, double>CLenetClassifier::predict(const cv::Mat &img)
{
	std::pair<int, double>p;
	if (!bloaded)
	{
		load();
	}
	else
	{
		cv::Mat inputBlob = blobFromImage(img, 1.0, cv::Size(20, 20),_mean);
		cv::Mat prob;
        _net.setInput(inputBlob, "data");
		prob = _net.forward("prob");
		cv::Mat probMat = prob.reshape(1, 1);
		cv::Point classNumber;
		cv::minMaxLoc(probMat, NULL, &p.second, NULL, &classNumber);
		p.first = classNumber.x;
	}

	return p;
}

bool CLenetClassifier::load(cv::String modelTxt, cv::String modelBin)
{
	_net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);
    _mean = cv::Scalar(66, 66, 66);
	bloaded = !_net.empty();
	return bloaded;
}