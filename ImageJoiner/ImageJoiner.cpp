#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>


using namespace std;
using namespace cv;

int main(int argc, char const ** argv)
{
	Mat bryan = imread("Bryan4th.png", 1);
	Mat jolley = imread("Jolley4th.png", 1);

	if ( !bryan.data )
    {
    	cout << "Error reading bryan" << endl;
        return -1;
    }

    if ( !jolley.data )
    {
    	cout << "Error reading jolley" << endl;
        return -1;
    }


	Mat output (850, 1920*2 - 750 - 1000 - 150 - 200 - 200, CV_8UC3);

	int channels = output.channels();
	int nRows = output.rows;
	int nCols = output.cols;

	for (int i = 0; i < nRows; ++i)
	{
		uchar * dst = output.ptr<uchar>(i);
		uchar * brySrc = bryan.ptr<uchar>(i + (1080-850)/2);
		uchar * jolSrc = jolley.ptr<uchar>(i+ (1080-850)/2);

		for (int j = 0; j < nCols*channels; ++j)
		{
			if(j/channels >= 1920-1000 - 160)
				dst[j] = jolSrc[j - 920*channels + 750*channels + 320*channels];
			else
				dst[j] = brySrc[j+1000*channels];
		}

	}

	cvNamedWindow("Joined Image", WINDOW_NORMAL);
	imshow("Joined Image", output);
	waitKey(0);
	imwrite("Bryan+Jolley.png", output);

	return 0;
}