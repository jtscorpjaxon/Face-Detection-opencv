// Ocv.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
  /* string path = "et_tps.png";
   Mat img = imread(path);
    imshow("Image", img);
    waitKey(0);
    std::cout << "Hello World!\n";*/
	
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
		faceCascade.detectMultiScale(img, faces, 1.1, 10);

		for (int i = 0; i < faces.size(); i++)
		{

			faceROI[i] = img(faces[i]);
			cropped_faces[i] = faceROI[i];
			namedWindow("Face" + std::to_string(i));
			imshow("Face"+ std::to_string(i), cropped_faces[i]);
			rectangle(img, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 1);
		}
		if (faces.size() > 0)
		{	
			
		//	cv::resize(img, img, Size(), 2, 2);
			//imwrite("test.jpg", img);
		}
		imshow("Image", img);
	waitKey(2);
	}
	
   
}
