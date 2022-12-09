// Daniel Bandala @ dec-2022
#include "imstack.h"

imstack::imstack()
{
}

imstack::~imstack()
{
}

void imstack::colorread(std::string path, float scalef)
{
	for (const auto& entry : std::filesystem::directory_iterator(path))
	{
		if (!is_directory(entry.path()))
		{
			std::cout << entry.path().string() << std::endl;
			colorimg = cv::imread(entry.path().string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
			std::cout << colorimg.size()  << std::endl;
			if (scalef != 1)
				cv::resize(colorimg, colorimg, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);
			colorimg.convertTo(colorimg32, CV_32F, 1.0 / 65535.0f);
			break;
		}
	}
}

void imstack::getfiles(std::string path)
{
	for (auto& entry : std::filesystem::directory_iterator(path))
		if (!is_directory(entry.path()))
			sort_filename.push_back(entry.path());
}

void imstack::createmask(int initialid, int numsamples, float scalef, float limit_v, int gx, int gy)
{
	std::cout << std::endl << "Create Mask"  << std::endl;
	HSVstack hsvstack;
	cv::Mat img, mask_h, result_cpu;
	cv::cuda::GpuMat h, s, v, hsv;
	cv::cuda::GpuMat channels[3];
	cv::cuda::GpuMat average_h;
	cv::cuda::GpuMat average_s;
	cv::cuda::GpuMat average_v;
	cv::cuda::GpuMat std_h;
	cv::cuda::GpuMat std_s;
	cv::cuda::GpuMat std_v;
	cv::cuda::GpuMat cv_h;
	cv::cuda::GpuMat cv_s;
	cv::cuda::GpuMat cv_v;
	cv::cuda::GpuMat dst, src, gpu_h, gpu_s, gpu_v;
	cv::cuda::GpuMat result;

	int breackid = 0;
	std::vector<std::filesystem::path>::const_iterator first = sort_filename.begin() + initialid;
	std::vector<std::filesystem::path>::const_iterator last = sort_filename.begin() + initialid +numsamples;
	std::vector<std::filesystem::path> sub_sort_filename(first, last);
	for (auto& filename : sub_sort_filename)
	{
		std::cout << filename.string() << std::endl;
		img = cv::imread(filename.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if(scalef!=1)
			cv::resize(img, img, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);		
		// upload image to gpu
		src.upload(img);
		src.convertTo(dst, CV_32F, 1.0 / 65535.0f);
		// get hsv channels
		cv::cuda::cvtColor(dst, hsv, cv::COLOR_BGR2HSV); // cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::cuda::split(hsv, channels);
		cv::cuda::divide(channels[0], 360.0, h);
		s = channels[1];
		v = channels[2];
		// create stack for segmentation
		hsvstack.h.push_back(h);
		hsvstack.s.push_back(s);
		hsvstack.v.push_back(v);
		breackid++;
		if (breackid == numsamples)
			break;
	}

	//average using 10 elements
	average_h.create(colorimg.rows, colorimg.cols, CV_32F);
	average_h.setTo(cv::Scalar(0));
	average_s.create(colorimg.rows, colorimg.cols, CV_32F);
	average_s.setTo(cv::Scalar(0));
	average_v.create(colorimg.rows, colorimg.cols, CV_32F);
	average_v.setTo(cv::Scalar(0));
	for (int i = 0; i < numsamples; ++i)
	{
		cv::cuda::add(average_h, hsvstack.h[i], average_h);
		cv::cuda::add(average_s, hsvstack.s[i], average_s);
		cv::cuda::add(average_v, hsvstack.v[i], average_v);
	}
	cv::cuda::divide(average_h, hsvstack.h.size(), average_h);
	cv::cuda::divide(average_s, hsvstack.h.size(), average_s);
	cv::cuda::divide(average_v, hsvstack.h.size(), average_v);

	//std using 10 elements
	cv::cuda::GpuMat temp;
	std_h.create(colorimg.rows, colorimg.cols, CV_32F);
	std_h.setTo(cv::Scalar(0));
	std_s.create(colorimg.rows, colorimg.cols, CV_32F);
	std_s.setTo(cv::Scalar(0));
	std_v.create(colorimg.rows, colorimg.cols, CV_32F);
	std_v.setTo(cv::Scalar(0));
	for (int i = 0; i < numsamples; ++i)
	{
		cv::cuda::absdiff(hsvstack.h[i], average_h, temp);
		cv::cuda::pow(temp, 2, temp);
		cv::cuda::add(std_h, temp, std_h);
		cv::cuda::absdiff(hsvstack.s[i], average_s, temp);
		cv::cuda::pow(temp, 2, temp);
		cv::cuda::add(std_s, temp, std_s);
		cv::cuda::absdiff(hsvstack.v[i], average_v, temp);
		cv::cuda::pow(temp, 2, temp);
		cv::cuda::add(std_v, temp, std_v);
	}
	cv::cuda::divide(std_h, numsamples, std_h);
	cv::cuda::divide(std_s, numsamples, std_s);
	cv::cuda::divide(std_v, numsamples, std_v);
	cv::cuda::sqrt(std_h, std_h);
	cv::cuda::sqrt(std_s, std_s);
	cv::cuda::sqrt(std_v, std_v);

	//coefficient of variatiob using 10 elements
	cv_h.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_h.setTo(cv::Scalar(0));
	cv_s.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_s.setTo(cv::Scalar(0));
	cv_v.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_v.setTo(cv::Scalar(0));
	cv::cuda::divide(std_h, average_h, cv_h);
	cv::cuda::bitwise_or(average_h, 0, dst);
	cv::cuda::bitwise_not(dst, dst);
	cv_h.setTo(0, (dst));
	cv::cuda::divide(std_s, average_s, cv_s);
	cv::cuda::bitwise_or(average_s, 0, dst);
	cv::cuda::bitwise_not(dst, dst);
	cv_s.setTo(0, (dst));
	cv::cuda::divide(std_v, average_v, cv_v);
	cv::cuda::bitwise_or(average_v, 0, dst);
	cv::cuda::bitwise_not(dst, dst);
	cv_v.setTo(0, (dst));

	// filtering and generate mask
	cv::Ptr<cv::CLAHE> clahe = cv::cuda::createCLAHE(4);

	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(average_v.type(), average_v.type(), cv::Size(gx, gy), 0, 0);
	filter->apply(average_v, dst);
	dst.convertTo(dst, CV_16U, 65535.0f);
	clahe->apply(dst, dst);
	dst.convertTo(gpu_v, CV_32F, 1.0 / 65535.0f);
	//dst.download(average_v);

	//cv::GaussianBlur(average_s, average_s, cv::Size(gx, gy), 0, 0);
	filter = cv::cuda::createGaussianFilter(average_s.type(), average_s.type(), cv::Size(gx, gy), 0, 0);
	filter->apply(average_s, dst);
	dst.convertTo(dst, CV_16U, 65535.0f);
	clahe->apply(dst, dst);
	dst.convertTo(gpu_s, CV_32F, 1.0 / 65535.0f);
	//dst.download(average_s);

	//cv::GaussianBlur(average_h, average_h, cv::Size(gx, gy), 0, 0);
	filter = cv::cuda::createGaussianFilter(average_h.type(), average_h.type(), cv::Size(gx, gy), 0, 0);
	filter->apply(average_h, dst);
	dst.convertTo(dst, CV_16U, 65535.0f);
	clahe->apply(dst, dst);
	dst.convertTo(gpu_h, CV_32F, 1.0 / 65535.0f);
	//dst.download(average_h);
	
	cv::cuda::multiply(gpu_v, gpu_s, result);
	result.download(result_cpu);
	cv::inRange(result_cpu, cv::Scalar(0), cv::Scalar(limit_v), mask_h);
	cv::bitwise_not(mask_h, mask_h);

	cv::Mat im_dilate;
	int dilation_size = 8;
	cv::Mat elementd = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		cv::Point(dilation_size, dilation_size));
	cv::dilate(mask_h, im_dilate, elementd);

	//connected components filter
	cv::Mat labels, stats, centroids;
	int nlabels = cv::connectedComponentsWithStats(im_dilate, labels, stats, centroids, 8);
	std::vector<int> grayval(nlabels);
	Statslabel statslabeltemp;
	int newlabel = 1;
	grayval[0] = 0;
	statslabeltemp.height.push_back(stats.at<int>(0, cv::CC_STAT_HEIGHT));
	statslabeltemp.width.push_back(stats.at<int>(0, cv::CC_STAT_WIDTH));
	statslabeltemp.left.push_back(stats.at<int>(0, cv::CC_STAT_LEFT));
	statslabeltemp.top.push_back(stats.at<int>(0, cv::CC_STAT_TOP));
	for (int idx = 1; idx < nlabels; idx++)
	{
		grayval[idx] = newlabel;
		if (stats.at<int>(idx, cv::CC_STAT_AREA) < (10000 * scalef))
		{
			grayval[idx] = 0;
		}
		else
		{
			newlabel = newlabel + 1;
			statslabeltemp.height.push_back(stats.at<int>(idx, cv::CC_STAT_HEIGHT));
			statslabeltemp.width.push_back(stats.at<int>(idx, cv::CC_STAT_WIDTH));
			statslabeltemp.left.push_back(stats.at<int>(idx, cv::CC_STAT_LEFT));
			statslabeltemp.top.push_back(stats.at<int>(idx, cv::CC_STAT_TOP));
		}
	}
	final_label = cv::Mat::zeros(labels.rows, labels.cols, CV_32F);
	for (int y = 0; y < labels.rows; y++)
	{
		for (int x = 0; x < labels.cols; x++)
		{
			int label = labels.at<int>(y, x);
			final_label.at<float>(y, x) = grayval[label];
		}
	}

	//fill holes
	cv::Mat im_th, im_floodfill, im_floodfill_inv;
	src.upload(final_label);
	cv::cuda::threshold(src, src, 0, 255, cv::THRESH_BINARY);
	src.convertTo(src, CV_8U);
	src.download(im_th);
	im_floodfill = im_th.clone();
	cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255));
	src.upload(im_floodfill);
	cv::cuda::bitwise_not(src, src);
	src.download(im_floodfill_inv);
	cv::Mat im_fill = (im_th | im_floodfill_inv);
	//erode
	cv::Mat im_erode;
	int erosion_size = 6;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));
	cv::erode(im_fill, im_erode, element);
	cv::erode(im_erode, im_erode, element);
	cv::erode(im_erode, im_erode, element);
	//new connected components
	nlabels = cv::connectedComponentsWithStats(im_erode, labels, stats, centroids, 8);
	std::vector<int> grayval2(nlabels);

	newlabel = 1;
	grayval2[0] = 0;
	statslabel.height.push_back(stats.at<int>(0, cv::CC_STAT_HEIGHT));
	statslabel.width.push_back(stats.at<int>(0, cv::CC_STAT_WIDTH));
	statslabel.left.push_back(stats.at<int>(0, cv::CC_STAT_LEFT));
	statslabel.top.push_back(stats.at<int>(0, cv::CC_STAT_TOP));
	for (int idx = 1; idx < nlabels; idx++)
	{
		grayval2[idx] = newlabel;
		if (stats.at<int>(idx, cv::CC_STAT_AREA) < (10000*scalef))
		{
			grayval2[idx] = 0;
		}
		else
		{
			newlabel = newlabel + 1;
			statslabel.height.push_back(stats.at<int>(idx, cv::CC_STAT_HEIGHT));
			statslabel.width.push_back(stats.at<int>(idx, cv::CC_STAT_WIDTH));
			statslabel.left.push_back(stats.at<int>(idx, cv::CC_STAT_LEFT));
			statslabel.top.push_back(stats.at<int>(idx, cv::CC_STAT_TOP));
		}
	}
	final_label = cv::Mat::zeros(labels.rows, labels.cols, CV_32F);
	for (int y = 0; y < labels.rows; y++)
	{
		for (int x = 0; x < labels.cols; x++)
		{
			int label = labels.at<int>(y, x);
			final_label.at<float>(y, x) = grayval2[label];
		}
	}
	// save mask image
	cv::imwrite("results/mask.tif", final_label);
}

void imstack::readstack(float scalef)
{
	std::cout << std::endl << "Read Stack"  << std::endl;
	cv::Mat img;
	cv::cuda::GpuMat hsv, h, s, v, channels[3];
	cv::cuda::GpuMat src,dst;
	for (auto& filename : sort_filename)
	{
		std::cout << filename.string() << std::endl;
		img = cv::imread(filename.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if (scalef != 1)
			cv::resize(img, img, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);
		//img.convertTo(img32, CV_32F, 1.0 / 65535.0f);
		//cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		// upload image to gpu
		src.upload(img);
		src.convertTo(dst, CV_32F, 1.0 / 65535.0f);
		// get hsv channels
		cv::cuda::cvtColor(dst, hsv, cv::COLOR_BGR2HSV); // cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::cuda::split(hsv, channels);
		cv::cuda::divide(channels[0], 360.0, h);
		s = channels[1];
		v = channels[2];
		images.h.push_back(h);
		images.s.push_back(s);
		images.v.push_back(v);
	}
}

void imstack::objpixels(Rectangle rectangle, int labelid, float limsimi)
{
	cv::Mat tempimg = final_label.clone();
	tempimg.setTo(0, (tempimg != labelid));
	tempimg.setTo(1, (tempimg == labelid));
	tempimg.convertTo(tempimg, CV_8U);
	HSVstack temp;
	cv::Mat src;
	rectangle.height = rectangle.height - 2;
	for (int idspectrum = 0; idspectrum < images.h.size(); idspectrum++)
	{
		cv::cuda::GpuMat cropimage;
		cropimage = images.h[idspectrum].clone();
		cropimage.setTo(0, (tempimg != 1));
		cropimage = cropimage(cv::Rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height));
		temp.h.push_back(cropimage);
		cropimage = images.s[idspectrum].clone();
		cropimage.setTo(0, (tempimg != 1));
		cropimage = cropimage(cv::Rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height));
		temp.s.push_back(cropimage);
		cropimage = images.v[idspectrum].clone();
		cropimage.setTo(0, (tempimg != 1));
		cropimage = cropimage(cv::Rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height));
		temp.v.push_back(cropimage);
	}
	Pixelarray objpixels;
	for (int i = 0; i < temp.h[0].rows; i++)
	{
		for (int j = 0; j < temp.h[0].cols; j++)
		{
			std::array<float, NUM_SPECT> tempch_h = { 0 };
			std::array<float, NUM_SPECT> tempch_s = { 0 };
			std::array<float, NUM_SPECT> tempch_v = { 0 };
			
			int numch = 0;
			for (auto& ch : temp.h)
			{
				ch.download(src);
				tempch_h.at(numch) = src.at<float>(i, j);
				numch++;
			}

			numch = 0;
			for (auto& ch : temp.s)
			{
				ch.download(src);
				tempch_s.at(numch) = src.at<float>(i, j);
				numch++;
			}

			numch = 0;
			for (auto& ch : temp.v)
			{
				ch.download(src);
				tempch_v.at(numch) = src.at<float>(i, j);
				numch++;
			}
			if (!(std::all_of(tempch_h.begin(), tempch_h.end(), [](float i) { return i == 0; })) || !(std::all_of(tempch_s.begin(), tempch_s.end(), [](float i) { return i == 0; })) || !(std::all_of(tempch_v.begin(), tempch_v.end(), [](float i) { return i == 0; })))
			{
				objpixels.val_h.push_back(tempch_h);
				objpixels.val_s.push_back(tempch_s);
				objpixels.val_v.push_back(tempch_v);
			}
		}
	}
	int inidx = 0;
	if (Uobjpixels.val_h.size() == 0)
	{
		Uobjpixels.val_h.push_back(objpixels.val_h[0]);
		Uobjpixels.val_s.push_back(objpixels.val_s[0]);
		Uobjpixels.val_v.push_back(objpixels.val_v[0]);
		inidx = 1;
	}


	for(int idx= inidx; idx < objpixels.val_h.size(); idx++)
	{
		int inisize = Uobjpixels.val_h.size();
		int stoploop = 0;
		for (int idy = 0; idy < inisize; idy++)
		{
			for (int idc = 0; idc < objpixels.val_h[idx].size(); idc++)
			{
				if ((std::abs((objpixels.val_h[idx])[idc] - (Uobjpixels.val_h[idy])[idc]) < limsimi) || (std::abs((objpixels.val_s[idx])[idc] - (Uobjpixels.val_s[idy])[idc]) < limsimi) || (std::abs((objpixels.val_v[idx])[idc] - (Uobjpixels.val_v[idy])[idc])) < limsimi)
				{
					stoploop = 1;
					break;
				}
			}
			if (stoploop == 1)
				break;
		}
		if (stoploop == 0)
		{
			Uobjpixels.val_h.push_back(objpixels.val_h[idx]);
			Uobjpixels.val_s.push_back(objpixels.val_s[idx]);
			Uobjpixels.val_v.push_back(objpixels.val_v[idx]);
		}
	}
}

void imstack::readstackv(float scalef)
{
	std::cout << std::endl << "Read Stack - Vector" << std::endl;
	cv::Mat img, img32, channel;
	cv::cuda::GpuMat hsv, h, s, v, src, dst;
	for (auto& filename : sort_filename)
	{
		cv::cuda::GpuMat channels[3];
		std::cout << filename.string() << std::endl;
		img = cv::imread(filename.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if (scalef != 1)
			cv::resize(img, img, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);
		// send image to gpu
		src.upload(img);
		src.convertTo(dst, CV_32F, 1.0 / 65535.0f);
		// get hsv channels
		cv::cuda::cvtColor(dst, hsv, cv::COLOR_BGR2HSV); // cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::cuda::split(hsv, channels);
		channels[2].download(channel);
		imagesv.v.push_back(channel);
	}
}

void imstack::objpixelsv(Rectangle rectangle, int labelid, float limsimi, std::string multidirout)
{
	std::cout << std::endl << "Object Pixels - Vector ("+std::to_string(labelid)+")" << std::endl;
	cv::Mat src;
	cv::Mat tempimg = final_label.clone();
	
	// create mask
	cv::cuda::GpuMat tempimg_gpu;
	tempimg.setTo(0, (tempimg != labelid));
	tempimg.setTo(1, (tempimg == labelid));
	tempimg.convertTo(tempimg, CV_8U);
	tempimg_gpu.upload(tempimg);
	cv::cuda::bitwise_or(tempimg_gpu, 0, tempimg_gpu);
	cv::cuda::bitwise_not(tempimg_gpu, tempimg_gpu);
	
	Vstack temp;
	rectangle.height = rectangle.height - 2;
	std::string objdir = multidirout + "/object_" + std::to_string(labelid)  + "/";
	std::filesystem::create_directory(objdir);
	for (int idspectrum = 0; idspectrum < imagesv.v.size(); idspectrum++)
	{
		cv::Mat cropimage;
		cropimage = imagesv.v[idspectrum].clone();
		cropimage.setTo(0, (tempimg != 1));
		cropimage = cropimage(cv::Rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height));
		temp.v.push_back(cropimage);
		std::string filename = objdir + "S_" + std::to_string(idspectrum) + ".tif";
		cv::imwrite(filename, cropimage);
	}
	
	// get unique pixels with cuda kernell
	cv::Mat label_im = Wrapper::unique_pixels_wrapper(temp.v, UobjpixelsV.val_v, limsimi);
	// save label image
	labelimag.push_back(label_im);
}