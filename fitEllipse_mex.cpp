#include "mex.h"
#include <vector>
#include "opencv2/opencv.hpp"
// Label for pixels with undefined gradient.
#define NOTDEF -1024.0
// PI 
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif // !M_PI 
#define M_1_2_PI 1.57079632679489661923

struct point2d
{
	double x, y;
};

//input : (xi,yi)
//output: x0,y0,a,b,phi,ellipara需要事先申请内存
//successfull, return 1; else return 0
int fitEllipse(point2d* dataxy, int datanum, double* ellipara)
{
	if (datanum < 5)
	{
		return 0;
	}
	std::vector<cv::Point2f> pts;
	for (int i = 0; i < datanum; ++i)
	{
		pts.push_back(cv::Point2f(dataxy[i].x, dataxy[i].y));
	}
	cv::RotatedRect rr = cv::fitEllipseAMS(pts);
	ellipara[0] = rr.center.x;
	ellipara[1] = rr.center.y;
	ellipara[2] = rr.size.width / 2;
	ellipara[3] = rr.size.height / 2;
	ellipara[4] = rr.angle / 180.0 * CV_PI;
	if (ellipara[2] < ellipara[3])
	{
		std::swap(ellipara[2], ellipara[3]);

		if (ellipara[4] < 0)
			ellipara[4] += M_1_2_PI;
		else
			ellipara[4] -= M_1_2_PI;
	}
	if (ellipara[4] <= -M_1_2_PI)
		ellipara[4] += M_PI;
	if (ellipara[4] >= M_1_2_PI)
		ellipara[4] -= M_PI;
	if (rr.boundingRect2f().area() == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs != 2)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:invalidNumInputs", "Two inputs required.");
	}
	else if (nlhs > 2)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:maxlhs", "Too many output arguments.");
	}
	double* x = (double*)mxGetData(prhs[0]);
	double* y = (double*)mxGetData(prhs[1]);
	int x_rows = (int)mxGetM(prhs[0]);
	int x_cols = (int)mxGetN(prhs[0]);
	int y_rows = (int)mxGetM(prhs[1]);
	int y_cols = (int)mxGetN(prhs[1]);

	if (x_cols != 1)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:invalidInputShape", "X must be Nx1.");
	}
	if (y_cols != 1)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:invalidInputShape", "Y must be Nx1.");
	}

	if (x_rows != y_rows)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:invalidInputShape", "X and Y shapes must be the same.");
	}

	int datanum = (int)mxGetM(prhs[0]);

	point2d* dataxy = (point2d*)malloc(datanum*sizeof(point2d)); ;
	for (int i = 0; i < datanum; ++i)
	{
		dataxy[i].x = x[i];
		dataxy[i].y = y[i];
	}
	plhs[0] = mxCreateDoubleMatrix(1, 5, mxREAL);
	double* ellipara = (double*)mxGetData(plhs[0]);
	double info=fitEllipse(dataxy, datanum, ellipara);


	plhs[1] = mxCreateDoubleScalar(info);
	//---------------------------------------------------------------------
	free(dataxy);
}
