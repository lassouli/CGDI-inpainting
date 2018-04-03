#ifndef PATCH_MATCH_HPP
#define PATCH_MATCH_HPP

#include <opencv2/opencv.hpp>

void
patch_match(
    cv::Mat3b const& source,
    cv::Mat1b const& mask,
    cv::Mat3b& target,
    int patchSize,
    const int nbIterations);

#endif
