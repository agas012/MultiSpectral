#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <ranges>
#include <list>
#include <array>
#include <tuple>
#include <set>

#include "vartypes.h"
#include "imstack.h"

void main()
{
	imstack ImgI;
	float scalef = 1.0;
	ImgI.colorread("E:/TempData/tepalcate/T_1/Event_Version 1_0001_0002/", scalef);
	ImgI.getfiles("E:/TempData/tepalcate/T_1/Event_Version 1_0001_0002/ms/");
	ImgI.createmask(10, scalef);
	ImgI.readstackv(scalef);

	for (int idx = 1; idx < ImgI.statslabel.left.size(); idx++)
	{
		Rectangle tempr;
		tempr.left = ImgI.statslabel.left[idx];
		tempr.top = ImgI.statslabel.top[idx];
		tempr.width = ImgI.statslabel.width[idx];
		tempr.height = ImgI.statslabel.height[idx];
		ImgI.objpixelsv(tempr, idx, 0.02); //here you can put different thresholds, lower values more unique pixels, try with .1, 0.05, 0.01, etc
		//ImgI.labelimag vector contains mat files with the label objects;
		//ImgI.UobjpixelsV list of pixels with unique values in v;
		std::string filename;
		filename = "E:/TempData/tepalcate/T_1/Event_Version 1_0001_0002/ms/out/label_" + std::to_string(idx) + ".tif";
		cv::imwrite(filename, ImgI.labelimag[idx-1]);
	}
	int a = 0;
}