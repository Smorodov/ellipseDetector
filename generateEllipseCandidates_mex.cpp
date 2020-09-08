#include "mex.h"
#include <vector>
#include "opencv2/opencv.hpp"
/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0
/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */
#define M_1_2_PI 1.57079632679489661923
#define M_1_4_PI 0.785398163

#define M_3_4_PI 2.35619449

#define M_1_8_PI 0.392699081
#define M_3_8_PI 1.178097245
#define M_5_8_PI 1.963495408
#define M_7_8_PI 2.748893572
#define M_4_9_PI 1.396263401595464  //80°
#define M_1_9_PI  0.34906585  //20°
#define M_1_10_PI 0.314159265358979323846   //18°
#define M_1_12_PI 0.261799387   //15°
#define M_1_15_PI 0.20943951    //12°
#define M_1_18_PI 0.174532925   //10°
/** 3/2 pi */
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI  6.28318530718
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

typedef struct image_double_s
{
	double* data;
	int xsize, ysize;
} *image_double;

struct point2i //(or pixel).
{
	int x, y;
};

struct point2d
{
	double x, y;
};

typedef struct PairGroup_s
{
	point2i pairGroupInd;
	point2d center;  //(x0,y0)
	point2d axis;    //(a,b)
	double  phi;     //angle of orientation  
}PairGroup;

typedef struct  PairGroupList_s
{
	int length;
	PairGroup* pairGroup;
}PairGroupList;

double* mylsd(int* n_out, double* img, int X, int Y, int** reg_img, int* reg_x, int* reg_y);
void groupLSs(double* lines, int line_num, int* region, int imgx, int imgy, std::vector<std::vector<int>>* groups);
void calcuGroupCoverage(double* lines, int line_num, std::vector<std::vector<int>> groups, double*& coverages);
void calculateGradient2(double* img_in, unsigned int imgx, unsigned int imgy, image_double* angles);
void calculateGradient3(double* img_in, unsigned int imgx, unsigned int imgy, image_double* angles);
PairGroupList* getValidInitialEllipseSet(double* lines, int line_num, std::vector<std::vector<int>>* groups, double* coverages, image_double angles, double distance_tolerance, int specified_polarity);
void generateEllipseCandidates(PairGroupList* pairGroupList, double distance_tolerance, double*& ellipse_candidates, int* candidates_num);
void free_image_double(image_double i);
void freePairGroupList(PairGroupList* list);

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if (nrhs != 3)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:invalidNumInputs", "One input required.");
	}
	else if (nlhs > 4)
	{
		mexErrMsgIdAndTxt("MATLAB:revord:maxlhs", "Too many output arguments.");
	}
	uchar* inputimg = (uchar*)mxGetData(prhs[0]);
	int imgy, imgx;
	int edge_process_select = (int)mxGetScalar(prhs[1]);
	int specified_polarity = (int)mxGetScalar(prhs[2]);
	imgy = (int)mxGetM(prhs[0]);
	imgx = (int)mxGetN(prhs[0]);
	double* data = (double*)malloc(imgy * imgx * sizeof(double));//将输入矩阵中的图像数据转存到一维数组中
	for (int c = 0; c < imgx; c++)
	{
		for (int r = 0; r < imgy; r++)
		{
			data[c + r * imgx] = inputimg[r + c * imgy];
		}
	}
	int n;
	std::vector<std::vector<int>> groups;
	double* coverages;
	int* reg;
	int reg_x;
	int reg_y;
	double* out = mylsd(&n, data, imgx, imgy, &reg, &reg_x, &reg_y);
	groupLSs(out, n, reg, reg_x, reg_y, &groups);//分组
	free(reg); //释放内存
	calcuGroupCoverage(out, n, groups, coverages);//计算每个组的覆盖角度

	printf("The number of output arc-support line segments: %i\n", n);
	printf("The number of arc-support groups:%i\n", groups.size());

	image_double angles;
	if (edge_process_select == 1)
	{
		calculateGradient2(data, imgx, imgy, &angles); //version2, sobel; version 3 canny
	}
	else
	{
		calculateGradient3(data, imgx, imgy, &angles); //version2, sobel; version 3 canny
	}
	 PairGroupList * pairGroupList;
	 double distance_tolerance = 2;//max( 2.0, 0.005*min(angles->xsize,angles->ysize) ); // 0.005%*min(xsize,ysize)
	 double * candidates; //候选椭圆
	 double * candidates_out;//输出候选椭圆指针
	 int  candidates_num = 0;//候选椭圆数量
	 //rejectShortLines(out,n,&new_n);
	 pairGroupList = getValidInitialEllipseSet(out,n,&groups,coverages,angles,distance_tolerance,specified_polarity);
	 if(pairGroupList != NULL)
	 {
		printf("The number of initial ellipses：%i \n",pairGroupList->length);
		generateEllipseCandidates(pairGroupList, distance_tolerance, candidates, &candidates_num);
		printf("The number of ellipse candidates: %i \n",candidates_num);
		
		plhs[0] = mxCreateDoubleMatrix(5,candidates_num,mxREAL);
		candidates_out = (double*)mxGetPr(plhs[0]);
		//候选圆组合(xi,yi,ai,bi,phi_i)', 5 x candidates_num, 复制到矩阵candidates_out中
		memcpy(candidates_out,candidates,sizeof(double)*5*candidates_num);

		freePairGroupList(pairGroupList);
		free(candidates);
	 }
	 else
	 {
		 printf("The number of initial ellipses：%i \n",0);
		 double *candidates_out;
		 plhs[0] = mxCreateDoubleMatrix(5,1,mxREAL);
		 candidates_out = (double*)mxGetPr(plhs[0]);
		 candidates_out[0] = candidates_out[1] = candidates_out[2] = candidates_out[3] = candidates_out[4] = 0;
	 }
	 uchar *edgeimg_out;
	 unsigned long edge_pixels_total_num = 0;//边缘总像素
	 double *gradient_vec_out;
	 plhs[1] = mxCreateNumericMatrix(imgy,imgx,mxUINT8_CLASS,mxREAL);
	 edgeimg_out = (uchar*)mxGetData(plhs[1]);
	 unsigned long addr,g_cnt = 0;
	 for (int c = 0; c < imgx; c++)
	 {
		 for (int r = 0; r < imgy; r++)
		 {
			 addr = r * imgx + c;
			 if (angles->data[addr] == NOTDEF)
			 {
				 edgeimg_out[c * imgy + r] = 0;
			 }
			 else
			 {
				 edgeimg_out[c * imgy + r] = 255;
				 //------------------------------------------------
				 edge_pixels_total_num++;
			 }
		 }
	 }

	 printf("edge pixel number: %i\n",edge_pixels_total_num);
	//申请edge_pixels_total_num x 2 来保存每一个边缘点的梯度向量，以列为优先，符合matlab的习惯
	 plhs[2] = mxCreateDoubleMatrix(2,edge_pixels_total_num,mxREAL);
	 gradient_vec_out = (double*)mxGetPr(plhs[2]);
	 for (int c = 0; c < imgx; c++)
	 {
		 for (int r = 0; r < imgy; r++)
		 {
			 addr = r * imgx + c;
			 if (angles->data[addr] != NOTDEF)
			 {
				 gradient_vec_out[g_cnt++] = cos(angles->data[addr]);
				 gradient_vec_out[g_cnt++] = sin(angles->data[addr]);
			 }
		 }
	 }
	 //---------------------------------------------------------------------
      if (nlhs == 4)
      {
          cv::Mat ls_mat = cv::Mat::zeros(imgy, imgx, CV_8UC1);
          for (int i = 0; i < n; i++)//draw lines
          {
			  cv::Point2d p1(out[8 * i], out[8 * i + 1]), p2(out[8 * i + 2], out[8 * i + 3]);
			  cv::line(ls_mat, p1, p2, cv::Scalar(255, 0, 0));
          }
          if (candidates_num > 0)//draw ellipses
          {
              for (int i = 0; i < candidates_num; i++)
              {
				  cv::ellipse(ls_mat, cv::Point((int)candidates_out[i * 5], (int)candidates_out[i * 5 + 1]), cv::Size(candidates_out[i * 5 + 2], candidates_out[i * 5 + 3]), candidates_out[i * 5 + 4] * 180 / M_PI, 0, 360, (cv::Scalar(255, 0, 0)), 1);
              }
          }
          plhs[3] = mxCreateDoubleMatrix(imgy, imgx, mxREAL);
          double* ls_img_out = (double*)mxGetPr(plhs[3]);
          //memcpy(ls_out_mat,ls_mat.data ,sizeof(unsigned char)*M*N);
          for (int i = 0; i < imgx; i++)
          {
              for (int j = 0; j < imgy; j++)
              {
                  ls_img_out[i * imgy + j] = ls_mat.data[j * imgx + i];
              }
          }
	}
	//---------------------------------------------------------------------
	free(data);
	free(coverages);
	free(out);
	free_image_double(angles);

}
