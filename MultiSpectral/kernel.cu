// Daniel Bandala @ dec-2022
#include "kernel.h"

__global__ void unique_pixels(float* A, float* B, float* C, int N) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    // if (ROW < N && COL < N) {
    //     // each thread computes one element of the block sub-matrix
    //     for (int i = 0; i < N; i++) {
    //         tmpSum += A[ROW * N + i] * B[i * N + COL];
    //     }
    // }
    // C[ROW * N + COL] = tmpSum;
}
namespace Wrapper {
	extern "C" cv::Mat unique_pixels_wrapper(std::vector<cv::Mat>& v, std::vector<std::array<float, NUM_SPECT>>& val_v, float limsimi) {
		cv::Mat label_im = cv::Mat::zeros(v[0].rows, v[0].cols, CV_32FC1);

		//unique_pixels<<<v[0].rows,v[0].cols>>>(A, B, C, N);

		for (int i = 0; i < v[0].rows; i++)
		{
			for (int j = 0; j < v[0].cols; j++)
			{
				std::array<float, NUM_SPECT> tempch_v = { 0 };
				int numch = 0;
				for (auto& ch : v)
				{
					tempch_v.at(numch) = ch.at<float>(i, j);
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
						label_im.at<float>(i, j) = val_v.size();
					}
					for (int idy = 0; idy < inisize; idy++)
					{
						int counter = 0;
						for (int idc = 0; idc < NUM_SPECT; idc++)
						{
							if (std::abs(tempch_v[idc] - (val_v[idy])[idc]) < limsimi)
							{
								counter++;
							}
						}
						if (counter == NUM_SPECT)
						{
							stoploop = 1;
							label_im.at<float>(i, j) = idy+1; 
							break;
						}
							
					}
					if (stoploop == 0)
					{
						val_v.push_back(tempch_v);
						label_im.at<float>(i, j) = val_v.size();
					}
				}	
			}
		}
		return label_im;
	}
}

// compile: 
// nvcc -c kernel.cu `pkg-config opencv --cflags --libs` -o kernel.o
// g++ source.cpp imstack.cpp `pkg-config opencv --cflags --libs` -o source kernel.o -lstdc++ -lcuda -lcudart