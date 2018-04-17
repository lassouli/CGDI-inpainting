#include <iostream>
#include <opencv2/opencv.hpp>

//#include "patch_match.hpp"
#include "inpainting.hpp"

using namespace cv;
using namespace std;

void clean_mask(Mat1b& mask) {
    mask.forEach([](uchar& p, const int[]){ if (p < 255) p = 0;});
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "usage: Inpainting <Source> <Mask>\n";
        return -1;
    }

    Mat3b image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
        cout << "No image data \n";
        return -1;
    }

    Mat1b mask;
    mask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (!mask.data) {
        cout << "No mask data \n";
        return -1;
    }
    clean_mask(mask);

    namedWindow("Inpainting", WINDOW_AUTOSIZE);
    Mat3b target;
    //cvtColor(image, image, CV_BGR2Lab);
    inpaint(image, mask, target);
    //cvtColor(target, target, CV_Lab2BGR);

    cout << "done" << endl;
    imshow("Inpainting", target);
    while (waitKey(10) != 'q') {}
}
