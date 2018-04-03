#include <iostream>
#include <opencv2/opencv.hpp>

#include "patch_match.hpp"

using namespace cv;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "usage: Inpainting <Source> <Mask>\n";
        return -1;
    }

    Mat3b image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
        std::cout << "No image data \n";
        return -1;
    }

    Mat1b mask;
    mask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (!mask.data) {
        std::cout << "No mask data \n";
        return -1;
    }

    Mat3b target;
    patch_match(image, mask, target, 7, 6);

    waitKey();
}
