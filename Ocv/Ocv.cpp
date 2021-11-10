// Ocv.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "windows.h"
using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

#define CAFFE
std::string path = "C:\\Users\\user\\source\\repos\\Ocv\\x64\\Debug\\";
const std::string caffeConfigFile = path+ "deploy.prototxt";
const std::string caffeWeightFile = path+"res10_300x300_ssd_iter_140000_fp16.caffemodel";

const std::string tensorflowConfigFile = path+"opencv_face_detector.pbtxt";
const std::string tensorflowWeightFile = path+"opencv_face_detector_uint8.pb";

void detectFaceOpenCVDNN(Net net, Mat& frameOpenCVDNN)
{
    int frameHeight = frameOpenCVDNN.rows;
    int frameWidth = frameOpenCVDNN.cols;

    //resize(frameOpenCVDNN, frameOpenCVDNN, Size(300, 300));


#ifdef CAFFE
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);
#else
    cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);
#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
      //  cout << confidence << endl;

        if (confidence > confidenceThreshold)
        {

            int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            
            cv::rectangle(frameOpenCVDNN, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
            imwrite("test.jpg", frameOpenCVDNN);
        }
    }

}


int main(int argc, const char** argv)
{
#ifdef CAFFE
    Net net = readNetFromCaffe(caffeConfigFile, caffeWeightFile);
#else
    Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
#endif

   VideoCapture source;
    if (argc == 1)
        source.open(0);
    else
        source.open(argv[1]);

    Mat frame;
   
   // source.read(frame);
    cout << "Channels: " + to_string(frame.channels()) << endl;
   Mat resized;
    //resize(frame, resized, Size(300, 300));

    double tt_opencvDNN = 0;
    double fpsOpencvDNN = 0;
    while(true)
    {
        source.read(frame);
        if(frame.empty())
            break;
        double t = cv::getTickCount();
    detectFaceOpenCVDNN(net, frame);
    tt_opencvDNN = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
    fpsOpencvDNN = 1/tt_opencvDNN;
    putText(frame, format("OpenCV DNN ; FPS = %.2f",fpsOpencvDNN), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
    imshow("OpenCV - DNN Face Detection", frame);
   // int k =
    waitKey(1);
    }
}
/*
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <string>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include "windows.h"
using namespace cv;
using namespace std;
int main()
{
   string path = "et_tps.png";
   Mat img = imread(path);
    imshow("Image", img);
    waitKey(0);
    std::cout << "Hello World!\n";*/
	/*
	string path = "D:\\Resources\\test.png";
	VideoCapture cap(0);
	Mat image_with_humanface;
	Mat img = imread(path);
	Mat cropped_faces[3];
	Mat faceROI[3];
	CascadeClassifier faceCascade;
	faceCascade.load("D:\\Resources\\haarcascade_frontalface_default.xml");

	if (faceCascade.empty()) { cout << "XML file not loaded" << endl; }

	vector<Rect> faces;
	while (true) {
		cap.read(img);
		cap.set(CAP_PROP_BUFFERSIZE,20000);
		faceCascade.detectMultiScale(img, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));

		for (int i = 0; i < faces.size(); i++)
		{

			faceROI[i] = img(faces[i]);
			cropped_faces[i] = faceROI[i];
			namedWindow("Face" + std::to_string(i));
			//imshow("Face"+ std::to_string(i), cropped_faces[i]);
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 1);
		}
		if (faces.size() > 0)
		{	
			
		//	cv::resize(img, img, Size(), 2, 2);
			//imwrite("test.jpg", img);
		}
		imshow("Image", img);
		
	waitKey(1);
	}
	

	   
}*/
