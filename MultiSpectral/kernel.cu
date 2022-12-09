// Daniel Bandala @ dec-2022
#include "kernel.h"

__global__ void unique_pixels(cv::Mat* v, std::array* val_v, cv::Mat* label_im, float limsimi) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

	std::array<float, NUM_SPECT> tempch_v = { 0 };
	int numch = 0;
	for (auto& ch : v)
	{
		tempch_v.at(numch) = ch.at<float>(row, col);
		numch++;
	}
	//save only vectors with at least a non zero element
	if (!(std::all_of(tempch_v.begin(), tempch_v.end(), [](float i) { return i == 0; }))) 
	{
		int inisize = val_v.size();
		int stoploop = 0;
		if (inisize == 0)
		{
			val_v.push_back(tempch_v);
			stoploop = 1;
			label_im.at<float>(row, col) = val_v.size();
		}
		for (int idy = 0; idy < inisize; idy++)
		{
			int counter = 0;
			for (int idc = 0; idc < NUM_SPECT; idc++)
			{
				if (std::abs(tempch_v[idc] - (val_v[idy])[idc]) < limsimi)
					counter++;
			}
			if (counter == NUM_SPECT)
			{
				stoploop = 1;
				label_im.at<float>(row, col) = idy+1; 
				break;
			}
				
		}
		if (stoploop == 0)
		{
			val_v.push_back(tempch_v);
			label_im.at<float>(row, col) = val_v.size();
		}
	}
}
namespace Wrapper {
	extern "C" cv::Mat unique_pixels_wrapper(std::vector<cv::Mat>& v, std::vector<std::array<float, NUM_SPECT>>& val_v, float limsimi) {
		cv::Mat *dev_v = 0;
		std::array *dev_val_v = 0;
		cv::Mat *dev_label_im = 0;
		cv::Mat label_im = cv::Mat::zeros(v[0].rows, v[0].cols, CV_32FC1);

		// allocate memory in device
		cudaMalloc((void**)&dev_v, v.size() * v[0].rows*v[0].cols);
		cudaMalloc((void**)&dev_val_v, sizeof(val_v) * sizeof(float));
		cudaMalloc((void**)&dev_label_im, v[0].rows*v[0].cols);

		// copy data to device
		cudaMemcpy(dev_v, v.data(), v.size() * v[0].rows*v[0].cols, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_v, val_v.data(), sizeof(val_v) * sizeof(float), cudaMemcpyHostToDevice);

		// execute kernel
		unique_pixels<<<v[0].rows,v[0].cols>>>(dev_v, dev_val_v, dev_label_im);
		
		// copy result to cpu
		cudaDeviceSynchronize();

		// Copy output vector from GPU buffer to host memory.
    	cudaStatus = cudaMemcpy(label_im.data(), dev_label_im, v[0].rows*v[0].cols, cudaMemcpyDeviceToHost);

		cudaFree(dev_v);
		cudaFree(dev_val_v);
		cudaFree(dev_label_im);

		// return label image
		return label_im;
	}
}

// compile: 
// nvcc -c kernel.cu `pkg-config opencv --cflags --libs` -o kernel.o
// g++ source.cpp imstack.cpp `pkg-config opencv --cflags --libs` -o source kernel.o -lstdc++ -lcuda -lcudart