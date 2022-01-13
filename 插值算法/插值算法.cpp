#include "opencv2/opencv.hpp"



using namespace std;
using namespace cv;


//双线性插值
//sx、sy为缩放因子
void Inter_Linear(Mat img, Mat& dst, double sx, double sy)
{
	uchar *pImg = img.data;
	int channel = img.channels();
	int dst_cols = round(img.cols * sx);
	int dst_rows = round(img.rows * sy);
	dst = Mat(Size(dst_cols, dst_rows), img.type());
	uchar *pDst = dst.data;
	int step = img.step / sizeof(uchar);
	uchar *src_rows_ptr_1_l = NULL;
	uchar *src_rows_ptr_1_r = NULL;
	uchar *dst_rows_ptr_1 = NULL;
	Vec3b *src_rows_ptr_3_l = NULL;
	Vec3b *src_rows_ptr_3_r = NULL;
	Vec3b *dst_rows_ptr_3 = NULL;
	for (int i = 0; i < dst_rows; i++)
	{
		/*几何中心对齐*/
		double index_i = (i + 0.5) / sx - 0.5;
		/*防止越界*/
		if (index_i < 0)
		{
			index_i = 0;
		}
		if (index_i > (img.rows - 1))
		{
			index_i = img.rows - 1;
		}
		/*相邻4像素的坐标*/
		int i1 = floor(index_i);
		int i2 = ceil(index_i);
		//u为传到浮点型坐标的小数部分
		double u = index_i - i1;
		if (img.channels() == 1)
		{
			src_rows_ptr_1_l = img.ptr<uchar>(i1);
			src_rows_ptr_1_r = img.ptr<uchar>(i2);
			dst_rows_ptr_1 = dst.ptr<uchar>(i);
		}
		else
		{
			src_rows_ptr_3_l = img.ptr<Vec3b>(i1);
			src_rows_ptr_3_r = img.ptr<Vec3b>(i2);
			dst_rows_ptr_3 = dst.ptr<Vec3b>(i);
		}
		for (int j = 0; j < dst_cols; j++)
		{
			/*几何中心对齐*/
			double index_j = (j + 0.5) / sy - 0.5;
			/*防止越界*/
			if (index_j < 0)
			{
				index_j = 0;
			}
			if (index_j > (img.cols - 1))
			{
				index_j = img.cols - 1;
			}
			/*相邻4像素的坐标*/
			int j1 = floor(index_j);
			int j2 = ceil(index_j);
			//v为传到浮点型坐标的小数部分
			double v = index_j - j1;
			if (img.channels() == 1)
			{
				dst_rows_ptr_1[j] = (1 - u) * (1 - v) * src_rows_ptr_1_l[j1] + (1 - u) * v * src_rows_ptr_1_l[j2] + u * (1 - v) * src_rows_ptr_1_r[j1] + u * v * src_rows_ptr_1_r[j2];
				//dst_rows_ptr_1[j] = (1 - u) * (1 - v) * img.at<uchar>(i1,j1) + (1 - u) * v * img.at<uchar>(i1, j2) + u * (1 - v) * img.at<uchar>(i2, j1) + u * v * img.at<uchar>(i2, j2);
			}
			else
			{
				dst_rows_ptr_3[j][0] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][0] + (1 - u) * v * src_rows_ptr_3_l[j2][0] + u * (1 - v) * src_rows_ptr_3_r[j1][0] + u * v * src_rows_ptr_3_r[j2][0];
				dst_rows_ptr_3[j][1] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][1] + (1 - u) * v * src_rows_ptr_3_l[j2][1] + u * (1 - v) * src_rows_ptr_3_r[j1][1] + u * v * src_rows_ptr_3_r[j2][1];
				dst_rows_ptr_3[j][2] = (1 - u) * (1 - v) * src_rows_ptr_3_l[j1][2] + (1 - u) * v * src_rows_ptr_3_l[j2][2] + u * (1 - v) * src_rows_ptr_3_r[j1][2] + u * v * src_rows_ptr_3_r[j2][2];
			}
		}
	}
}

//最近邻插值
//sx、sy为缩放因子
void nearest_neighbor(Mat img, Mat& dst, double sx, double sy)
{
	uchar *pImg = img.data;
	int channel = img.channels();
	int dst_cols = round(img.cols * sx);
	int dst_rows = round(img.rows * sy);
	dst = Mat(Size(dst_cols, dst_rows), img.type());
	uchar *pDst = dst.data;
	int step = img.step / sizeof(uchar);
	uchar *src_rows_ptr_1 = NULL;
	uchar *dst_rows_ptr_1 = NULL;
	Vec3b *src_rows_ptr_3 = NULL;
	Vec3b *dst_rows_ptr_3 = NULL;
	for (int i = 0; i < dst_rows; i++)
	{
		int src_rows = round(i / sy);
		/*防止越界*/
		if (src_rows < 0)
		{
			src_rows = 0;
		}
		if (src_rows > (img.rows - 1))
		{
			src_rows = img.rows - 1;
		}
		if (img.channels() == 1)
		{
			src_rows_ptr_1 = img.ptr<uchar>(src_rows);
			dst_rows_ptr_1 = dst.ptr<uchar>(i);
		}
		else
		{
			src_rows_ptr_3 = img.ptr<Vec3b>(src_rows);
			dst_rows_ptr_3 = dst.ptr<Vec3b>(i);
		}
		for (int j = 0; j < dst_cols; j++)
		{
			int src_cols = round(j / sx);
			/*防止越界*/
			if (src_cols < 0)
			{
				src_cols = 0;
			}
			if (src_cols > (img.cols - 1))
			{
				src_cols = img.cols - 1;
			}
			int src_channels = src_cols * channel;
			int dst_channels = j * channel;
			if (img.channels() == 1)
			{
				dst_rows_ptr_1[j] = src_rows_ptr_1[src_cols];
			}
			else
			{
				dst_rows_ptr_3[j][0] = src_rows_ptr_3[src_cols][0];
				dst_rows_ptr_3[j][1] = src_rows_ptr_3[src_cols][1];
				dst_rows_ptr_3[j][2] = src_rows_ptr_3[src_cols][2];
			}
		}
	}
}


int main(void)
{
	string path = "F:/cycle_gril/lenna.png";
	Mat srcImg = imread(path);
	Mat grayImg;
	Mat dstImg, dstImg1;
	cvtColor(srcImg, grayImg, COLOR_RGB2GRAY);

	Inter_Linear(srcImg, dstImg, 0.8, 0.8);
	imshow("srcImg", srcImg);
	imshow("grayImg", dstImg);

	nearest_neighbor(grayImg, dstImg1, 0.5, 0.5);
	imshow("grayImg", dstImg1);

	waitKey(0);
}
