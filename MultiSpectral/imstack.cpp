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
	HSVstack hsvstack;
	cv::Mat img, img32, hsv, h, s, v;
	cv::Mat channels[3];
	cv::Mat average_h;
	cv::Mat average_s;
	cv::Mat average_v;
	cv::Mat std_h;
	cv::Mat std_s;
	cv::Mat std_v;
	cv::Mat cv_h;
	cv::Mat cv_s;
	cv::Mat cv_v;
	cv::Mat mask_h;

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
		img.convertTo(img32, CV_32F, 1.0 / 65535.0f);
		cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::split(hsv, channels);
		h = channels[0]/360.0;
		s = channels[1];
		v = channels[2];
		hsvstack.h.push_back(h);
		hsvstack.s.push_back(s);
		hsvstack.v.push_back(v);
		breackid++;
		if (breackid == numsamples)
		{
			break;
		}
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
		average_h += hsvstack.h[i];
		average_s += hsvstack.s[i];
		average_v += hsvstack.v[i];
	}

	average_h = average_h / hsvstack.h.size();
	average_s = average_s / hsvstack.h.size();
	average_v = average_v / hsvstack.h.size();

	//std using 10 elements
	cv::Mat temp;
	std_h.create(colorimg.rows, colorimg.cols, CV_32F);
	std_h.setTo(cv::Scalar(0));
	std_s.create(colorimg.rows, colorimg.cols, CV_32F);
	std_s.setTo(cv::Scalar(0));
	std_v.create(colorimg.rows, colorimg.cols, CV_32F);
	std_v.setTo(cv::Scalar(0));
	for (int i = 0; i < numsamples; ++i)
	{
		cv::absdiff(hsvstack.h[i], average_h, temp);
		cv::pow(temp, 2, temp);
		std_h += temp;
		cv::absdiff(hsvstack.s[i], average_s, temp);
		cv::pow(temp, 2, temp);
		std_s += temp;
		cv::absdiff(hsvstack.v[i], average_v, temp);
		cv::pow(temp, 2, temp);
		std_v += temp;
	}
	cv::sqrt(std_h / numsamples, std_h);
	cv::sqrt(std_s / numsamples, std_s);
	cv::sqrt(std_v / numsamples, std_v);

	//coefficient of variatiob using 10 elements
	cv_h.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_h.setTo(cv::Scalar(0));
	cv_s.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_s.setTo(cv::Scalar(0));
	cv_v.create(colorimg.rows, colorimg.cols, CV_32F);
	cv_v.setTo(cv::Scalar(0));
	cv::divide(std_h, average_h, cv_h);
	cv_h.setTo(0, (average_h == 0));
	cv::divide(std_s, average_s, cv_s);
	cv_s.setTo(0, (average_s == 0));
	cv::divide(std_v, average_v, cv_v);
	cv_v.setTo(0, (average_v == 0));

	//mask
	cv::GaussianBlur(average_v, average_v, cv::Size(gx, gy), 0, 0);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat src, dst;
	average_v.convertTo(src, CV_16U, 65535.0f);
	clahe->apply(src, dst);
	dst.convertTo(average_v, CV_32F, 1.0 / 65535.0f);

	cv::GaussianBlur(average_s, average_s, cv::Size(gx, gy), 0, 0);
	average_s.convertTo(src, CV_16U, 65535.0f);
	clahe->apply(src, dst);
	dst.convertTo(average_s, CV_32F, 1.0 / 65535.0f);

	cv::GaussianBlur(average_h, average_h, cv::Size(gx, gy), 0, 0);
	average_h.convertTo(src, CV_16U, 65535.0f);
	clahe->apply(src, dst);
	dst.convertTo(average_h, CV_32F, 1.0 / 65535.0f);
	
	cv::Mat result, resultg;
	cv::multiply(average_v, average_s, result);

	cv::inRange(result, cv::Scalar(0), cv::Scalar(limit_v), mask_h);
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
	cv::Mat im_th;
	cv::threshold(final_label, im_th, 0, 255, cv::THRESH_BINARY);
	im_th.convertTo(im_th, CV_8U);
	cv::Mat im_floodfill = im_th.clone();
	cv::floodFill(im_floodfill, cv::Point(0, 0), cv::Scalar(255));
	cv::Mat im_floodfill_inv;
	cv::bitwise_not(im_floodfill, im_floodfill_inv);
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
	int a = 0;
}

void imstack::readstack(float scalef)
{
	cv::Mat img, img32, hsv, h, s, v;
	cv::Mat channels[3];
	for (auto& filename : sort_filename)
	{
		std::cout << filename.string() << std::endl;
		img = cv::imread(filename.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if (scalef != 1)
			cv::resize(img, img, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);
		img.convertTo(img32, CV_32F, 1.0 / 65535.0f);
		cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::split(hsv, channels);
		h = channels[0];
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
	rectangle.height = rectangle.height - 2;
	for (int idspectrum = 0; idspectrum < images.h.size(); idspectrum++)
	{
		cv::Mat cropimage;
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
				tempch_h.at(numch) = ch.at<float>(i, j);
				numch++;
			}
			numch = 0;
			for (auto& ch : temp.s)
			{
				tempch_s.at(numch) = ch.at<float>(i, j);
				numch++;
			}
			numch = 0;
			for (auto& ch : temp.v)
			{
				tempch_v.at(numch) = ch.at<float>(i, j);
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
	cv::Mat img, img32, hsv, h, s, v;
	cv::Mat channels[3];
	std::cout << std::endl << std::endl << std::endl;
	for (auto& filename : sort_filename)
	{
		cv::Mat channels[3];
		std::cout << filename.string() << std::endl;
		img = cv::imread(filename.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
		if (scalef != 1)
			cv::resize(img, img, cv::Size(), scalef, scalef, cv::INTER_LANCZOS4);
		img.convertTo(img32, CV_32F, 1.0 / 65535.0f);
		cv::cvtColor(img32, hsv, cv::COLOR_BGR2HSV);
		cv::split(hsv, channels);
		imagesv.v.push_back(channels[2]);
	}
}

void imstack::objpixelsv(Rectangle rectangle, int labelid, float limsimi, std::string multidirout)
{
	cv::Mat tempimg = final_label.clone();
	tempimg.setTo(0, (tempimg != labelid));
	tempimg.setTo(1, (tempimg == labelid));
	tempimg.convertTo(tempimg, CV_8U);
	Vstack temp;
	rectangle.height = rectangle.height - 2;
	std::string objdir = multidirout + "/" + std::to_string(labelid)  + "/";
	std::filesystem::create_directory(objdir);
	for (int idspectrum = 0; idspectrum < imagesv.v.size(); idspectrum++)
	{
		cv::Mat cropimage;
		cropimage = imagesv.v[idspectrum].clone();
		cropimage.setTo(0, (tempimg != 1));
		cropimage = cropimage(cv::Rect(rectangle.left, rectangle.top, rectangle.width, rectangle.height));
		temp.v.push_back(cropimage);
		std::string filename;
		filename = objdir + "S_" + std::to_string(idspectrum) + ".tif";
		cv::imwrite(filename, cropimage);
	}
	cv::Mat label_im = cv::Mat::zeros(temp.v[0].rows, temp.v[0].cols, CV_32FC1);
	for (int i = 0; i < temp.v[0].rows; i++)
	{
		for (int j = 0; j < temp.v[0].cols; j++)
		{
			std::array<float, NUM_SPECT> tempch_v = { 0 };
			int numch = 0;
			for (auto& ch : temp.v)
			{
				tempch_v.at(numch) = ch.at<float>(i, j);
				numch++;
			}
			//save only vectors with at least a non zero element
			if (!(std::all_of(tempch_v.begin(), tempch_v.end(), [](float i) { return i == 0; }))) 
			{
				int inisize = UobjpixelsV.val_v.size();
				int stoploop = 0;
				if (inisize == 0)
				{
					UobjpixelsV.val_v.push_back(tempch_v);
					stoploop = 1;
					label_im.at<float>(i, j) = UobjpixelsV.val_v.size();
				}
				for (int idy = 0; idy < inisize; idy++)
				{
					int counter = 0;
					for (int idc = 0; idc < NUM_SPECT; idc++)
					{
						if (std::abs(tempch_v[idc] - (UobjpixelsV.val_v[idy])[idc]) < limsimi)
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
					UobjpixelsV.val_v.push_back(tempch_v);
					label_im.at<float>(i, j) = UobjpixelsV.val_v.size();
				}
			}
		}
	}
	labelimag.push_back(label_im);
}