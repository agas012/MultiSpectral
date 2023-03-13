#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <ranges>
#include <list>
#include <array>
#include <tuple>
#include <set>
#include "imstack.h"

void main()
{
	float scalef = 1.0;
	PixelarrayV UobjpixelsVColec;
	//read all folders to process 
	const std::filesystem::path localdir{ "C:/Users/DR ALFONSO GASTELUM/Pictures/tepalcates/" };
	std::string outdir = localdir.string() + "Results2";
	std::string labeldirgeneral = outdir + "/Labels";
	std::filesystem::create_directory(outdir);
	std::filesystem::create_directory(labeldirgeneral);
	std::vector<std::string> dirlist;
	for (auto& p : std::filesystem::directory_iterator(localdir))
		if (p.is_directory())
		{
			std::size_t found = p.path().string().find_last_of("/\\");
			std::string dirtemp = p.path().string().substr(found + 1);
			if(dirtemp != "Results")
				dirlist.push_back(p.path().string());
		}
	int setid = 0;
	int labelid = 0;
	for (auto dir : dirlist)
	{
		imstack ImgI;
		std::size_t found = dir.find_last_of("/\\");
		std::string colordir = dir;
		std::string multidir = dir + "/ms/";

		std::string outgroupdir = outdir + "/" + dir.substr(found + 1);
		std::filesystem::create_directory(outgroupdir);
		std::string multidirout = outgroupdir + "/out/";
		std::filesystem::create_directory(multidirout);
		std::string labeldir = multidirout + "label/";
		std::filesystem::create_directory(labeldir);

		ImgI.colorread(colordir, scalef);
		ImgI.getfiles(multidir);
		ImgI.createmask(45, 50, scalef, 0.124f, 21, 21);
		//ImgI.createmask(45, 50, scalef,0.144f); 
		//
		ImgI.readstackv(scalef);
		ImgI.UobjpixelsV.val_v = UobjpixelsVColec.val_v;
		setid = setid + 1;
		for (int idx = 1; idx < ImgI.statslabel.left.size(); idx++)
		{
			Rectangle tempr;
			tempr.left = ImgI.statslabel.left[idx];
			tempr.top = ImgI.statslabel.top[idx];
			tempr.width = ImgI.statslabel.width[idx];
			tempr.height = ImgI.statslabel.height[idx];
			ImgI.objpixelsv(tempr, idx, 0.05, multidirout); //here you can put different thresholds, lower values more unique pixels, try with .1, 0.05, 0.01, etc
			//ImgI.labelimag vector contains mat files with the label objects;
			//ImgI.UobjpixelsV list of pixels with unique values in v;
			std::string filename;
			labelid = labelid + 1;
			//filename = labeldir + std::to_string(labelid) + ".tif"; labeldirgeneral
			filename = labeldirgeneral + "/" + std::to_string(setid) + "_" + std::to_string(labelid) + ".tif";
			cv::imwrite(filename, ImgI.labelimag[idx - 1]);
			cv::Mat color_im;
			cv::Rect crop_region(tempr.left, tempr.top, tempr.width, tempr.height);
			color_im = ImgI.colorimg(crop_region);
			filename = labeldirgeneral + "/" + std::to_string(setid) + "_" + std::to_string(labelid) + "c.tif";
			cv::imwrite(filename, color_im);
		}
		UobjpixelsVColec.val_v.insert(UobjpixelsVColec.val_v.end(), ImgI.UobjpixelsV.val_v.begin(), ImgI.UobjpixelsV.val_v.end());
	}
	std::string filename;
	filename = localdir.string() + "uvalues.csv";
	std::ofstream outFile(filename);
	// the important part
	for (const auto& array : UobjpixelsVColec.val_v)
	{
		for (const auto& value : array)
		{
			outFile << value << ",";
		}
		outFile << std::endl;
	}
	outFile.close();

	int a = 0;
}