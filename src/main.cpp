#include <iostream>
#include <opencv2/opencv.hpp>

#include "patch_match.hpp"

using namespace cv;
using namespace std;

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
    //cvtColor(image, image, CV_BGR2Lab);

    Mat1b mask;
    mask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (!mask.data) {
        std::cout << "No mask data \n";
        return -1;
    }

    namedWindow("Inpainting", WINDOW_AUTOSIZE);

    mask.forEach([](uchar& p, const int[]){ if (p < 255) p = 0;});

    Mat3b target;
    RandomizedPatchMatch rpm(image, mask, target, 11);
    rpm.computeNN(3);

    //cvtColor(target, target, CV_Lab2BGR);
    imshow("Inpainting", target);

    while (waitKey(100) != 'q') {}
}
