#ifndef ANN_SEARCH_HPP
#define ANN_SEARCH_HPP

#include <opencv2/opencv.hpp>

struct PatchDistance {
    cv::Mat3b target;
    cv::Mat2f features;
    cv::Mat1b available;
    float lambda;
    PatchDistance(cv::Mat3b const& target, cv::Mat2f const& features, float lambda);
    float operator()(cv::Rect const& origin_patch, cv::Rect const& candidate_patch) const;
};

class PatchMap {
public:
    int patchSize;
    int half;
    int rows, cols;
    cv::Mat2i offset;
    cv::Mat1f distances;
    cv::Mat1b dilated_mask;

    static cv::Matx<cv::Vec2i,2,2> const dirs;

    inline bool is_inside(int row, int col) const {
        return row >= 0 && row < rows && col >= 0 && col < cols;
    }

    inline bool is_inside(cv::Vec2i const& pos) const {
        return is_inside(pos[0], pos[1]);
    }

    inline bool is_inside(cv::Rect const& rect) const {
        return is_inside(rect.y, rect.x) && is_inside(rect.y+rect.height, rect.x+rect.width);
    }

    void update(cv::Vec2i pos, cv::Vec2i candidate, float distance);
    cv::Rect get_patch(cv::Vec2i const& p) const;
    void crop(cv::Rect const& origin_patch, cv::Vec2i const& pos, cv::Rect& candidate_patch, cv::Vec2i const& candidate) const;

    PatchMap(cv::Mat1b const& mask, int patchSize, int seed = 0x123DAB);
    void updateDistances(PatchDistance const& patch_distance);
    void upSample(cv::Mat1b const& mask);
};

void patch_match_propagation(
    PatchMap& pm,
    PatchDistance const& patch_distance,
    int parity);

void patch_match_search(
    PatchMap& pm,
    PatchDistance const& patch_distance,
    float inv_alpha = 2.f,
    int seed = 0x555EEE);

void ANNsearch(
    PatchMap& pm,
    PatchDistance const& patch_distance,
    int nbIterations);

#endif
