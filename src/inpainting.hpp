#ifndef INPAINTING_HPP
#define INPAINTING_HPP

#include <opencv2/opencv.hpp>

void inpaint(cv::Mat3b const& image, cv::Mat1b const& mask, cv::Mat3b& output, int patchSize, float lambda, int AnnIt);

#endif
