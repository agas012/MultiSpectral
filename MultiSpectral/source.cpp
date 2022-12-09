// Daniel Bandala @ dec-2022
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
// processing functions
#include "imstack.h"
// opencv dependencies
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// opencv cuda library
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
// cuda api
#include <cuda.h>
#include <cuda_runtime_api.h>

int main() {
	// start program
	std::clock_t begin = std::clock();
	float scalef = 1;
	PixelarrayV UobjpixelsVColec;
	//read all folders to process 
	const std::filesystem::path localdir{ "images/" };
	std::string dir_str = localdir.string();
	std::string outdir = "results";
	std::filesystem::create_directory(outdir);
	std::vector<std::string> dirlist;
	for (auto& p : std::filesystem::directory_iterator(localdir)) {
		if (p.is_directory()) {
			std::size_t found = p.path().string().find_last_of("/\\");
			std::string dirtemp = p.path().string().substr(found + 1);
			if(dirtemp != "results")
				dirlist.push_back(p.path().string());
		}
	}
	// create result directory
	std::filesystem::create_directory(outdir);

	// process images
	for (auto dir : dirlist) {
		imstack ImgI;
		std::string colordir = dir;
		std::string multidir = dir;
		
		std::string multidirout = outdir + "/out/";
		std::filesystem::create_directory(multidirout);
		std::string labeldir = outdir + "/label/";
		std::filesystem::create_directory(labeldir);

		// get colored image
		ImgI.colorread(dir_str, scalef);
		// get stack files
		ImgI.getfiles(dir);
		// create segmentation mask
		ImgI.createmask(0, 10, scalef, 0.132f, 5, 5);

		// read stack images
		ImgI.readstackv(scalef);
		ImgI.UobjpixelsV.val_v = UobjpixelsVColec.val_v;

		for (int idx = 1; idx < ImgI.statslabel.left.size(); idx++) {
			Rectangle tempr;
			tempr.left = ImgI.statslabel.left[idx];
			tempr.top = ImgI.statslabel.top[idx];
			tempr.width = ImgI.statslabel.width[idx];
			tempr.height = ImgI.statslabel.height[idx];
			ImgI.objpixelsv(tempr, idx, 0.12, multidirout); //here you can put different thresholds, lower values more unique pixels, try with .1, 0.05, 0.01, etc
			//ImgI.labelimag vector contains mat files with the label objects;
			//ImgI.UobjpixelsV list of pixels with unique values in v;
			std::string filename;
			filename = labeldir + std::to_string(idx) + ".tif";
			cv::imwrite(filename, ImgI.labelimag[idx - 1]);
		}
		UobjpixelsVColec.val_v.insert(UobjpixelsVColec.val_v.end(), ImgI.UobjpixelsV.val_v.begin(), ImgI.UobjpixelsV.val_v.end());
	}
	std::string filename;
	filename = outdir+"/uvalues.csv";
	std::ofstream outFile(filename);
	// the important part
	for (const auto& array : UobjpixelsVColec.val_v) {
		for (const auto& value : array)
			outFile << value << ",";
		outFile << std::endl;
	}
	outFile.close();
	// elapsed time
	std::clock_t end = std::clock();
    std::cout << double(end-begin) / CLOCKS_PER_SEC  << std::endl;
	// exit code
	return 0;
}

// compile: 
// nvcc -c kernel.cu `pkg-config opencv --cflags --libs` -o kernel.o
// g++ source.cpp imstack.cpp `pkg-config opencv --cflags --libs` -o source kernel.o -lstdc++ -lcuda -lcudart