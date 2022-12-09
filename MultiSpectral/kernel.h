// Daniel Bandala @ dec-2022
#pragma once
#include <set>
#include <numeric>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// opencv lib
#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>

#define NUM_SPECT 100
#define THREADS_PER_BLOCK 512

namespace Wrapper {
	extern "C" cv::Mat unique_pixels_wrapper(std::vector<cv::Mat>& v, std::vector<std::array<float, NUM_SPECT>>& val_v, float limsimi);
}