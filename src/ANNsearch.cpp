#include <iostream>
#include <algorithm>

#include "ANNsearch.hpp"

using namespace std;
using namespace cv;

PatchDistance::PatchDistance(cv::Mat3b const& _target, cv::Mat2f const& _features, float _lambda):
target(_target), features(_features), available(target.rows, target.cols, 255), lambda(_lambda) {}

float PatchDistance::operator()(cv::Rect const& a, cv::Rect const& b) const {
    Mat3b target_a(target, a);
    Mat3b target_b(target, b);
    Mat2f features_a(features, a);
    Mat2f features_b(features, b);
    //Mat1b mask(available, b);
    //float s = norm(mask, NORM_L1);
    float s = a.width * a .height;
    float target_diff = norm(target_a, target_b, NORM_L2SQR) / s;
    float features_diff = norm(features_a, features_b, NORM_L2SQR) / s;
    return target_diff + lambda * features_diff;
}

Matx<Vec2i,2,2> const PatchMap::dirs{
    Vec2i(1,0)
    , Vec2i(0,1)
    , Vec2i(-1,0)
    , Vec2i(0,-1)
};

void PatchMap::update(Vec2i pos, Vec2i candidate, float distance) {
    if (distance < distances(pos)) {
        distances(pos) = distance;
        offset(pos) = candidate;
    }
}

void PatchMap::crop(
    cv::Rect const& origin_patch, Vec2i const& origin,
    cv::Rect& candidate_patch, Vec2i const& candidate) const {
    Vec2i delta = origin - Vec2i(origin_patch.y, origin_patch.x);
    Vec2i bottomLeft = candidate - delta;
    candidate_patch.x = bottomLeft[1];
    candidate_patch.y = bottomLeft[0];
    candidate_patch.width = origin_patch.width;
    candidate_patch.height = origin_patch.height;
}

Rect PatchMap::get_patch(Vec2i const& p) const {
    Rect patch(p[1] - half, p[0] - half, 2*half + 1, 2*half + 1);
    patch.x = max(0, patch.x);
    patch.y = max(0, patch.y);
    Vec2i topRight(min(p[0] + half, rows), min(p[1] + half, cols));
    patch.width = topRight[1] - patch.x;
    patch.height = topRight[0] - patch.y;
    return patch;
}

PatchMap::PatchMap(Mat1b const& mask, int _patchSize, int seed)
: patchSize(_patchSize), half(patchSize / 2)
{
    rows = mask.rows;
    cols = mask.cols;
    RNG rng(seed);
    auto dilater = getStructuringElement(MORPH_RECT, Size(patchSize, patchSize));
    dilate(mask, dilated_mask, dilater);
    offset.create(rows, cols);
    offset.forEach([&](Vec2i& pixel, const int pos[]){
        pixel[0] = pos[0];
        pixel[1] = pos[1];
        while (dilated_mask(pixel) != 0) {
            pixel[0] = rng.uniform(half, rows - half);
            pixel[1] = rng.uniform(half, cols - half);
        }
    });
    distances = Mat1f::zeros(rows, cols);
    (Mat1f(rows, cols, FLT_MAX)).copyTo(distances, dilated_mask);
}

void PatchMap::updateDistances(PatchDistance const& patch_distance) {
    distances.forEach([&](float& dst, const int coords[]){
        Vec2i pos(coords[0], coords[1]);
        const Rect origin_patch = get_patch(pos);
        if (dilated_mask(pos) == 0) {
            dst = 0.f;
            return ;
        }
        Rect candidate_patch = get_patch(offset(pos));
        crop(origin_patch, pos, candidate_patch, offset(pos));
        dst = patch_distance(origin_patch, candidate_patch);
    });
}

void PatchMap::upSample(cv::Mat1b const& newMask) {
    rows = newMask.rows;
    cols = newMask.cols;
    auto dilater = getStructuringElement(MORPH_RECT, Size(patchSize, patchSize));
    dilate(newMask, dilated_mask, dilater);
    Mat2i newOffset(rows, cols);
    RNG rng(0x789CCC);
    newOffset.forEach([&](Vec2i& p, const int pos[]){
        if (dilated_mask(pos[0], pos[1]) == 0) {
            p = Vec2i(pos[0], pos[1]);
            return ;
        }
        int subRow = pos[0] / 2;
        int subCol = pos[1] / 2;
        p = offset(pos[0] / 2, pos[1] / 2);
        if (pos[0]%2 && pos[0]+1<rows) {
            p[0] += offset(subRow+1, subCol)[0];
        } else {
            p[0] *= 2;
        }
        if (pos[1]%2 && pos[1]+1<cols) {
            p[1] += offset(subRow, subCol+1)[1];
        } else {
            p[1] *= 2;
        }
        while (dilated_mask(p) != 0) {
            p[0] = rng.uniform(half, rows - half);
            p[1] = rng.uniform(half, cols - half);
        }
    });
    offset = newOffset;
    distances.create(rows, cols);
}

void patch_match_propagation(PatchMap& pm, PatchDistance const& patch_distance, int parity)
{
    int start_row = (parity == 1)? 0 : pm.rows-1;
    int end_row = (parity == 1)? pm.rows : -1;
    int start_col = (parity == 1)? 0 : pm.cols-1;
    int end_col = (parity == 1)? pm.cols : -1;
    int step = (parity == 1)? 1 : -1;
    for (int row = start_row; row != end_row; row += step) {
        for (int col = start_col; col != end_col; col += step) {
            Vec2i pos(row, col);
            if (pm.dilated_mask(pos) == 0)
                continue ;
            Rect patch = pm.get_patch(pos);
            for (int dir = 0; dir < 2; ++dir) {
                Vec2i v = pm.dirs(parity, dir);
                Vec2i neighbor = pos + v;
                Vec2i center = pm.offset(neighbor) - v;
                if (center[0] < pm.half
                    || center[0] >= pm.rows-pm.half
                    || center[1] < pm.half
                    || center[1] >= pm.cols-pm.half)
                    continue ;
                if (pm.dilated_mask(center) != 0)
                    continue ;
                Rect other = pm.get_patch(center);
                pm.crop(patch, pos, other, center);
                float d = patch_distance(patch, other);
                pm.update(pos, center, d);
            }
        }
    }
}

void patch_match_search(PatchMap& pm, PatchDistance const& patch_distance, float inv_alpha, int seed)
{
    RNG rng(seed);
    for (int row = 0; row < pm.rows; ++row) {
        for (int col = 0; col < pm.cols; ++col) {
            Vec2i pos(row, col);
            if (pm.dilated_mask(pos) == 0)
                continue ;
            Rect patch = pm.get_patch(pos);
            int squareSize = max(pm.rows, pm.cols) / inv_alpha;
            Vec2i old_offset = pm.offset(pos);
            while (squareSize > 1) {
                int infrow = max(pm.half, old_offset[0] - squareSize);
                int suprow = min(pm.rows - pm.half, old_offset[0] + squareSize) - 1;
                int infcol = max(pm.half, old_offset[1] - squareSize);
                int supcol = min(pm.cols - pm.half, old_offset[1] + squareSize) - 1;
                int urow = rng.uniform(infrow, suprow + 1);
                int ucol = rng.uniform(infcol, supcol + 1);
                squareSize /= inv_alpha;
                Vec2i center(urow, ucol);
                if (!pm.is_inside(center) || pm.dilated_mask(center))
                    continue ;
                Rect other = pm.get_patch(center);
                pm.crop(patch, pos, other, center);
                float d = patch_distance(patch, other);
                pm.update(pos, center, d);
            }
        }
    }
}

void ANNsearch(PatchMap& pm, PatchDistance const& patch_distance, int nbIterations) {
    for (int k = 1; k <= nbIterations; ++k) {
        int parity = k & 1;
        patch_match_propagation(pm, patch_distance, parity);
        patch_match_search(pm, patch_distance);
    }
}
