#include "mrdir.h"
#include "mrutil.h"
#include "mropencv.h"
using namespace std;

const string caffeplatedir = "../";
const string errordir = caffeplatedir + "/error";
const string platedatadir = caffeplatedir + "/data";
const string model_file = caffeplatedir + "/modeldef/deploy.prototxt";
const string trained_file = caffeplatedir + "/plate996.caffemodel";
const string mean_file = caffeplatedir + "/modeldef/mean.binaryproto";

class CLenetClassifier
{
public:
    static CLenetClassifier*getInstance()
    {
        static CLenetClassifier instance;
        return &instance;
    }
    std::pair<int, double>predict(const cv::Mat &img);
    bool load(String modelTxt = model_file, String modelBin = trained_file);
private:
    Net net;
    bool bloaded = false;
    CLenetClassifier() {
    }
};

std::pair<int, double>CLenetClassifier::predict(const cv::Mat &img)
{
    std::pair<int, double>p;
    if (!bloaded)
    {
        load();
    }
    else
    {
        cv::Mat inputBlob = blobFromImage(img, 1, Size(20, 20));
        cv::Mat prob;
        net.setInput(inputBlob, "data");
        prob = net.forward("prob");
        cv::Mat probMat = prob.reshape(1, 1);
        cv::Point classNumber;
        cv::minMaxLoc(probMat, NULL, &p.second, NULL, &classNumber);
        p.first = classNumber.x;
    }

    return p;
}

bool CLenetClassifier::load(String modelTxt, String modelBin)
{
    net = dnn::readNetFromCaffe(modelTxt, modelBin);
    bloaded = !net.empty();
    return bloaded;
}

void cleardir(const string dir)
{
	vector<string>files=getAllFilesinDir(dir);
	for (int i = 0; i < files.size(); i++)
	{
		string filepath = dir + "/" + files[i];
		remove(filepath.c_str());
	}
}

void clearerror(const string dir)
{
	vector<string>subdirs=getAllSubdirs(dir);
	for (int i = 0; i < subdirs.size(); i++)
	{
		string subdir = dir + "/" + subdirs[i];
		cleardir(subdir);
	}
}

int evaluation()
{
	string line;
	string label;
	int rightcount = 0, errorcount = 0, total = 0;	
	if (!EXISTS(errordir.c_str()))
	{
		cout << "Error dir not exist" << endl;
		MKDIR(errordir.c_str());
	}
	clearerror(errordir);
	vector<string>subdirs=getAllSubdirs(platedatadir);
	for (auto sub : subdirs)
	{
		string subdir = platedatadir + "/" + sub;
		vector<string>files=getAllFilesinDir(subdir);
		for (auto file : files)
		{
			string fileapth = subdir + "/" + file;
			cv::Mat img = cv::imread(fileapth);
			auto ret=CLenetClassifier::getInstance()->predict(img).first;
			if (ret == string2int(sub))
				rightcount++;
			else
			{
				errorcount++;
				string errorlabeldir = errordir;
				errorlabeldir = errorlabeldir + "/" + sub;
				if (!exist(errorlabeldir.c_str()))
				{
					_mkdir(errorlabeldir.c_str());
				}
				string errorfilepath = errorlabeldir + "/" + file.substr(0,file.size()-4) + "_" + sub + "_" + int2string(ret) + ".png";
				cout << sub + "/" + file.substr(0, file.size() - 4) + ":" + int2string(ret) << endl;
				imshow("error", img);
				imwrite(errorfilepath, img);
				cv::waitKey(1);
			}
			total++;
		}
	}
	cout << "acc:" << rightcount << "/" << total << endl;
	cout << rightcount*1.0 / total << endl;
	return 0;
}

int testimg(const std::string imgpath = "img/0.jpg")
{
    cv::Mat img = imread(imgpath);
    TickMeter tm;
    tm.start();
    auto p = CLenetClassifier::getInstance()->predict(img);
    tm.stop();
    std::cout << p.first << std::endl;// " " << p.second << endl;
    std::cout << tm.getTimeMilli() << "ms" << std::endl;
    return 0;
}

int testdir(const std::string dir = "img")
{
    auto files = getAllFilesinDir(dir);
    for (int i = 0; i < files.size(); i++)
    {
        std::string imgpath = dir + "/" + files[i];
        std::cout << files[i] << ":";
        testimg(imgpath);
    }
    return 0;
}

int main(int argc,char*argv[])
{
	if (argc==1)
		evaluation();
	else
	{
        testimg();
        testdir();
	}
	return 0;
}