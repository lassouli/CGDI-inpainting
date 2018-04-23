#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include <opencv2/opencv.hpp>

#include "ANNsearch.hpp"

void upSample(cv::Mat3b& target, cv::Mat1b const& mask, PatchMap const& pm);
void upSample(cv::Mat2f& features, cv::Mat1b const& mask, PatchMap const& pm);
float reconstruction(cv::Mat3b& target, cv::Mat2f& features, cv::Mat1b const& mask, PatchMap const& pm);
void finalReconstruction(cv::Mat3b& target, cv::Mat1b const& mask, PatchMap const& pm);
void onionPeelInitialization(cv::Mat3b& target, cv::Mat2f& features, cv::Mat1b const& mask, PatchMap const& pm, PatchDistance const& patch_distance);

#endif
