#include <iostream>
#include <opencv2/opencv.hpp>

//#include "patch_match.hpp"
#include "inpainting.hpp"
#include "../GUI/cvui.h"
#include "../GUI/gui_skeleton.hpp"


using namespace cv;
using namespace std;

void clean_mask(Mat1b& mask) {
    mask.forEach([](uchar& p, const int[]){ if (p < 255) p = 0;});
}

int main(int argc, char** argv) {
    int create_mask ;
    if ((argv[1][0] == '-') && (argv[1][1] == 'm')) {
      create_mask = 1 ;
    }
    else {
      create_mask = 0 ;
      if (argc != 3) {
        cout << "usage when using existing mask: Inpainting <Source> <Mask>\n" << "usage when creating mask: Inpainting -m <Source>\n";
        return -1;
      }
    }

    Mat3b image;
    image = imread(argv[1 + create_mask], CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
        cout << "No image data \n";
        return -1;
    }

    Mat1b mask;
    if (create_mask) {
      char *arguments[2] ;
      arguments[0] = "Create_mask" ;
      arguments[1] = argv[2] ;
      mask_from_scratch(2, arguments) ;
      mask = imread("masque.png", CV_LOAD_IMAGE_GRAYSCALE) ;
    }
    else {
      mask = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    }
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
