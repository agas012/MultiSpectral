#pragma once

struct HSVStack {
	std::vector<cv::Mat> h;
	std::vector<cv::Mat> s;
	std::vector<cv::Mat> v;
	cv::Mat average_h;
	cv::Mat average_s;
	cv::Mat average_v;
	cv::Mat std_h;
	cv::Mat std_s;
	cv::Mat std_v;
	cv::Mat cv_h;
	cv::Mat cv_s;
	cv::Mat cv_v;
};

struct StatsLabel {
	std::vector<int> left;
	std::vector<int> top;
	std::vector<int> width;
	std::vector<int> height;
};

#define NUM_SPECT 100
struct PixelArray {
	std::vector<std::array<float, NUM_SPECT>> val_h;
	std::vector<std::array<float, NUM_SPECT>> val_s;
	std::vector<std::array<float, NUM_SPECT>> val_v;
};

