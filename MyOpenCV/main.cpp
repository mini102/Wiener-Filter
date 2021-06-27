
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include<math.h>

using namespace cv;
using namespace std;

void showFFT(const std::string& name, const Mat& f, Mat& o) {
	int w = f.cols / 2;
	int h = f.rows / 2;
	Mat M;
	std::vector<Mat> channels;
	split(f, channels);
	magnitude(channels[0], channels[1], M);
	o = M;
	Mat c(f.size(), CV_32FC1);
	M(Rect(0, 0, w, h)).copyTo(c(Rect(w, h, w, h)));
	M(Rect(w, 0, w, h)).copyTo(c(Rect(0, h, w, h)));
	M(Rect(0, h, w, h)).copyTo(c(Rect(w, 0, w, h)));
	M(Rect(w, h, w, h)).copyTo(c(Rect(0, 0, w, h)));
	log(c, c);
	normalize(c, c, 0, 1, NORM_MINMAX);
	imshow(name, c);
}

int main(int ac, char** av) {

	float K = 0.001;//0.001;
	Mat f= imread("Wiener_Input2.png",0); //두번째 파라미터로 0주면 흑백으로 read 
	Mat k = imread("Wiener_Kernel.png",0);
	Mat G;
	Mat H;
	imshow("image", f);
	imshow("kernel", k);
	f.convertTo(f, CV_32FC1, 1 / 255.f);//CV_32F,1/255.f);
	k.convertTo(k, CV_32FC1, 1 / 255.f); //CV_32F, 1 / 255.f);
	divide(k,sum(k),k);
	dft(f, G, DFT_COMPLEX_OUTPUT);
	dft(k, H, DFT_COMPLEX_OUTPUT);

	Mat W;
	Mat c;
	mulSpectrums(G, H, c, 0, true);
	Mat p;
	//showFFT("parent", H, p);
	std::vector<Mat> channels;
	split(H, channels);
	magnitude(channels[0], channels[1], p);
	pow(p,2.0,p);  //제곱
	add(p, K, p);

	std::vector<Mat> Channels;
	split(c, Channels);
	//magnitude(Channels[0], Channels[1], c);
	divide(Channels[0], p, Channels[0]);
	divide(Channels[1], p, Channels[1]);
	merge(Channels, W);

	idft(W, W, DFT_SCALE | DFT_REAL_OUTPUT);
	imshow("winenr", W);
	waitKey(0);
	return 0;
}

