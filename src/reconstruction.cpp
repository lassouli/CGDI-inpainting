#include <iostream>
#include <algorithm>
#include <limits>

#include "reconstruction.hpp"

using namespace std;
using namespace cv;

namespace {
    float computeWeights(PatchMap const& pm, Mat1f& s) {
        vector<float> sorted_dists;
        for (int row = 0; row < pm.rows; ++row) {
            for (int col = 0; col < pm.cols; ++col) {
                if (pm.dilated_mask(row, col))
                    sorted_dists.push_back(pm.distances(row, col));
            }
        }
        sort(begin(sorted_dists), end(sorted_dists));
        float sigma = sorted_dists[int(sorted_dists.size() * 0.75)];
        pm.distances.copyTo(s);
        s = -s / (2*sigma*sigma);
        exp(s, s);
        return sigma;
    }
}

void upSample(cv::Mat3b& target, cv::Mat1b const& mask, PatchMap const& pm) {
    target.forEach([&](Vec3b& pixel, const int pos[]){
        if (mask(pos[0], pos[1]))
            pixel = target(pm.offset(pos[0], pos[1]));
    });
}

void upSample(cv::Mat2f& features, cv::Mat1b const& mask, PatchMap const& pm) {
    features.forEach([&](Vec2f& pixel, const int pos[]){
        if (mask(pos[0], pos[1]))
            pixel = features(pm.offset(pos[0], pos[1]));
    });
}

float reconstruction(Mat3b& target, cv::Mat2f& features, Mat1b const& mask, PatchMap const& pm) {
    Mat1f s;
    float sigma = computeWeights(pm, s);
    target.forEach([&](Vec3b& pixel, const int coord[]){
        Vec2i pos(coord[0], coord[1]);
        if (mask(pos) == 0)
            return ;
        Rect origin = pm.get_patch(pos);
        float sum = 0.f;
        Vec3d color(0,0,0);
        Vec2d feature(0,0);
        for (int row = origin.y; row < origin.y+origin.height; ++row) {
            for (int col = origin.x; col < origin.x+origin.width; ++col) {
                Vec2i q(row, col);
                Vec2i alter_ego = pos - q + pm.offset(q);
                if (!pm.is_inside(alter_ego) || mask(alter_ego))
                    continue ;
                double sq = max(double(s(q)), numeric_limits<double>::epsilon());
                color += sq * static_cast<Vec3d>(target(alter_ego));
                feature += sq * features(alter_ego);
                sum += sq;
            }
        }
        color /= sum;
        feature /= sum;
        pixel = static_cast<Vec3b>(color);
        features(pos) = feature;
    });
    return sigma;
}

void finalReconstruction(cv::Mat3b& target, cv::Mat1b const& mask, PatchMap const& pm) {
    target.forEach([&](Vec3b& pixel, const int coord[]){
        Vec2i pos(coord[0], coord[1]);
        if (mask(pos) == 0)
            return ;
        Rect origin = pm.get_patch(pos);
        float best_distance = FLT_MAX;
        for (int row = origin.y; row < origin.y+origin.height; ++row) {
            for (int col = origin.x; col < origin.x+origin.width; ++col) {
                Vec2i q(row, col);
                Vec2i alter_ego = pos - q + pm.offset(q);
                if (!pm.is_inside(alter_ego) || mask(alter_ego))
                    continue ;
                if (pm.distances(q) < best_distance) {
                    best_distance = pm.distances(q);
                    pixel = target(alter_ego);
                }
            }
        }
    });
}

void onionPeelInitialization(Mat3b& target, Mat2f& features, Mat1b const& mask, PatchMap const& pm, PatchDistance const& patch_distance) {
    Mat1b available; bitwise_not(mask, available);
    Mat2i outborder; Mat2i inborder;
    double sigma = 1.;
    vector<double> weights(1, 1.);
    auto const addNeighbours = [&](Vec2i const& pos){
        for (int i = 0; i < 4; ++i) {
            Vec2i v = pos + pm.dirs(i/2, i%2);
            if (pm.is_inside(v) && available(v) == 0) {
                inborder.push_back(v);
                available(v) = available(pos);
            }
        }
    };
    for (int row = 0; row < pm.rows; ++row) {
        for (int col = 0; col < pm.cols; ++col) {
            Vec2i pos(row, col);
            if (mask(pos) != 0)
                continue ;
            addNeighbours(pos);
        }
    }
    auto const swapQueue = [&](){
        outborder = inborder;
        inborder.release();
        sort(begin(weights), end(weights));
        sigma = weights[int(weights.size()*0.75)];
    };
    swapQueue();
    while (!outborder.empty()) {
        for (auto const& pos : outborder) {
            available(pos) = 0;
        }
        for (auto const& pos : outborder) {
            Rect origin = pm.get_patch(pos);
            float sum = 0.f;
            Vec3d color(0,0,0);
            Vec2d feature(0,0);
            for (int row = origin.y; row < origin.y+origin.height; ++row) {
                for (int col = origin.x; col < origin.x+origin.width; ++col) {
                    Vec2i q(row, col);
                    if (available(q) == 0)
                        continue ;
                    Vec2i alter_ego = pos - q + pm.offset(q);
                    if (!pm.is_inside(alter_ego) || available(alter_ego) == 0)
                        continue ;
                    Rect a = pm.get_patch(q);
                    Rect b = pm.get_patch(pm.offset(q));
                    pm.crop(a, q, b, pm.offset(q));
                    double dst = patch_distance(a, b, available);
                    double s = exp(-dst / (2*sigma*sigma));
                    s = max(s, numeric_limits<double>::epsilon());
                    color += s * static_cast<Vec3d>(target(alter_ego));
                    feature += s * static_cast<Vec2d>(features(alter_ego));
                    sum += s;
                }
            }
            color /= sum;
            feature /= sum;
            target(pos) = static_cast<Vec3b>(color);
            features(pos) = static_cast<Vec2f>(feature);
        }
        for (auto const& pos : outborder) {
            Rect a = pm.get_patch(pos);
            Rect b = pm.get_patch(pm.offset(pos));
            pm.crop(a, pos, b, pm.offset(pos));
            float dst = patch_distance(a, b, available);
            weights.push_back(dst);
        }
        for (int i = 0; i < outborder.rows; ++i) {
            available(outborder(i)) = 255;
        }
        for (auto const& pos : outborder) {
            addNeighbours(pos);
        }
        swapQueue();
        imshow("Inpainting", target);
        waitKey(1);
    }
}
