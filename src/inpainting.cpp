#include <vector>

#include "inpainting.hpp"
#include "ANNsearch.hpp"
#include "reconstruction.hpp"

using namespace std;
using namespace cv;

namespace {
    void buildTextureFeaturePyramid(Mat3b const& img, vector<Mat2f>& featurePyr, int L) {
        Mat1b src_gray; cvtColor(img, src_gray, CV_BGR2GRAY);
        Mat grad[2];
        Mat1f abs_grad[2];
        Sobel(src_gray, grad[0], CV_32F, 1, 0);
        Sobel(src_gray, grad[1], CV_32F, 0, 1);
        abs_grad[0] = abs(grad[0]);
        abs_grad[1] = abs(grad[1]);
        Mat2f deriv(img.rows, img.cols);
        merge(abs_grad, 2, deriv);
        blur(deriv, deriv, Size2i(1 << L, 1 << L));
        buildPyramid(deriv, featurePyr, L);
    }

    void buildMaskPyramid(Mat1b const& mask, vector<Mat1b>& maskPyr, int L) {
        buildPyramid(mask, maskPyr, L);
        for (auto& pyr : maskPyr) {
            pyr.forEach([](uchar& p, const int[]){ if (p > 0) p = 255;});
        }
    }
}

void computeArea(vector<float>& area, vector<Mat1b> const& maskPyr) {
    for (int l = 0; l < (int)maskPyr.size(); ++l) {
        for (int row = 0; row < maskPyr[l].rows; ++row) {
            for (int col = 0; col < maskPyr[l].cols; ++col) {
                area[l] += 1.f;
            }
        }
    }
}

void inpaint(Mat3b const& image, Mat1b const& mask, Mat3b& output) {
    const int L = 4;
    vector<Mat3b> imgPyr(L+1);
    vector<Mat1b> maskPyr(L+1);
    vector<Mat2f> featurePyr(L+1);
    buildPyramid(image, imgPyr, L);
    buildMaskPyramid(mask, maskPyr, L);
    buildTextureFeaturePyramid(image, featurePyr, L);
    vector<float> area(L+1, 0.f);
    computeArea(area, maskPyr);
    PatchMap pm(maskPyr.back(), 7);
    onionPeelInitialization(imgPyr.back(), featurePyr.back(), maskPyr.back(), pm);
    for (int l = L; l >= 0; --l) {
        PatchDistance patch_distance(imgPyr[l], featurePyr[l], 50.f);
        if (l != L) {
            pm.upSample(maskPyr[l]);
            upSample(imgPyr[l], maskPyr[l], pm);
            upSample(featurePyr[l], maskPyr[l], pm);
            pm.updateDistances(patch_distance);
            reconstruction(imgPyr[l], featurePyr[l], maskPyr[l], pm);
            pm.updateDistances(patch_distance);
            imshow("Inpainting", imgPyr[l]);
            waitKey(1);
        }
        pm.updateDistances(patch_distance);
        for (int k = 0; k < 10; ++k) {
            Mat3b before = imgPyr[l].clone();
            ANNsearch(pm, patch_distance, 10);
            float sigma = reconstruction(imgPyr[l], featurePyr[l], maskPyr[l], pm);
            cout << "[l, k, sigma] " << l << " " << k << " " << sigma << endl;
            pm.updateDistances(patch_distance);
            imshow("Inpainting", imgPyr[l]);
            waitKey(1);
            float e = norm(before, imgPyr[l], NORM_L1);
            e /= (3.f*area[l]);
            if (e < 0.1f) {
                break ;
            }
        }
    }
    finalReconstruction(imgPyr.front(), maskPyr.front(), pm);
    output = imgPyr.front();
}
