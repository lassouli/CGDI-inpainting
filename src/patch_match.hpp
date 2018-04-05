#ifndef PATCH_MATCH_HPP
#define PATCH_MATCH_HPP

#include <opencv2/opencv.hpp>

struct PatchZone {
    PatchZone(cv::Vec2i center, int patchSize);
    cv::Vec2i center;
    cv::Vec2i bottomLeft;
    cv::Vec2i topRight;
};

class RandomizedPatchMatch {
    static cv::Matx<cv::Vec2i,2,2> const dirs;
    static const int inv_alpha = 2;

    inline bool is_inside(cv::Vec2i const& pos) const {
        return pos[0] >= 0 && pos[0] < rows && pos[1] >= 0 && pos[1] < cols;
    }

    inline bool is_inside(PatchZone const& rect) const {
        return is_inside(rect.bottomLeft) && is_inside(rect.topRight);
    }

    void update(cv::Vec2i pos, cv::Vec2i candidate, float distance);

    cv::Rect get_patch(cv::Vec2i const& p, int half) const;

    float patch_distance(
        cv::Rect const& origin_patch,
        cv::Rect const& candidate_patch
    ) const;

    void patch_match_propagation(int parity);
    void patch_match_search();

    cv::Mat3b const& source;
    cv::Mat1b const& mask;
    cv::Mat3b& target;
    cv::Mat1f distances;
    cv::Mat2i offset;
    int patchSize;
    cv::RNG rng;
    int half;
    int rows;
    int cols;

public:
    RandomizedPatchMatch(
        cv::Mat3b const& source,
        cv::Mat1b const& mask,
        cv::Mat3b& target,
        int patchSize,
        int seed = 178956
    );

    void computeNN(int nbIterations);
    void patch_match_iteration(int iteration);
};

#endif
