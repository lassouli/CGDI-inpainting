#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/errors.hpp>

#include "inpainting.hpp"
#include "gui_skeleton.hpp"

using namespace cv;
using namespace std;

void clean_mask(Mat1b& mask) {
    mask.forEach([](uchar& p, const int[]){ if (p < 255) p = 0;});
}

struct Options {
    string image;
    string mask;
    string outputname;
    int patchSize;
    float lambda;
    int AnnIt;
} options;

int main(int argc, char** argv) {
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("image,i", po::value<string>(&options.image), "image name")
        ("mask,m", po::value<string>(&options.mask)->default_value(""), "mask name, if provided")
        ("create,c", po::value<string>(&options.outputname)->default_value(""), "name of the saved mask, if provided")
        ("patchSize,p", po::value<int>(&options.patchSize)->default_value(7), "patch size")
        ("lambda,l", po::value<float>(&options.lambda)->default_value(50.f), "lambda texture feature")
        ("AnnIt,a", po::value<int>(&options.AnnIt)->default_value(10), "number of ANN iterations")
    ;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    po::notify(vm);
    if (vm.count("help")) {
        cout << desc;
        return 0;
    }

    int patchSize = options.patchSize;
    float lambda = options.lambda;
    int AnnIt = options.AnnIt;

    Mat3b image;
    image = imread(options.image, CV_LOAD_IMAGE_COLOR);
    if (!image.data) {
        cout << "No image data \n";
        return -1;
    }

    Mat1b mask;
    if (!options.mask.empty()) {
        mask = imread(options.mask, CV_LOAD_IMAGE_GRAYSCALE);
        if (!mask.data) {
            cout << "No mask data \n";
            return -1;
        }
        clean_mask(mask);
    } else {
        cout << "Press ECHAP when the mask is completed. Use + or - to increase or decrease pencil size." << endl;
        mask_from_scratch(image, mask);
    }

    if (!options.outputname.empty()) {
        imwrite(options.outputname, mask);
    }

    namedWindow("Inpainting", WINDOW_AUTOSIZE);
    Mat3b target;
    //cvtColor(image, image, CV_BGR2Lab);
    inpaint(image, mask, target, patchSize, lambda, AnnIt);
    //cvtColor(target, target, CV_Lab2BGR);

    cout << "done.\nPress ECHAP to quit." << endl;
    imshow("Inpainting", target);
    while (waitKey(10) != 27) {}
}
