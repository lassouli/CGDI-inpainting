#include <iostream>

#include "patch_match.hpp"

using namespace cv;

cv::Matx<cv::Vec2i,2,2> const RandomizedPatchMatch::dirs{
    cv::Vec2i(1,0)
    , cv::Vec2i(0,1)
    , cv::Vec2i(-1,0)
    , cv::Vec2i(0,-1)
};

PatchZone::PatchZone(cv::Vec2i _center, int patchSize):
    center(_center)
{
    int half = patchSize / 2;
    bottomLeft = center - Vec2i(half, half);
    topRight = center + Vec2i(half, half);
}

RandomizedPatchMatch::RandomizedPatchMatch(
    Mat3b const& _source,
    Mat1b const& _mask,
    Mat3b& _target,
    int _patchSize,
    int seed
) : source(_source), mask(_mask), target(_target), patchSize(_patchSize), rng(seed)
{
    half = patchSize / 2;
    rows = source.rows;
    cols = source.cols;
    source.copyTo(target);
    distances = Mat1f::zeros(rows, cols);
    (Mat1f(rows, cols, FLT_MAX)).copyTo(distances, mask);
    offset.create(rows, cols);
    offset.forEach([&](Vec2i& pixel, const int pos[]){
        pixel[0] = pos[0];
        pixel[1] = pos[1];
        while (mask(pixel) != 0) {
            pixel[0] = rng.uniform(half, rows - half);
            pixel[1] = rng.uniform(half, cols - half);
        }
    });
    target.forEach([&](Vec3b& pixel, const int pos[]){
        pixel = source(offset(pos[0],pos[1]));
    });
}

void RandomizedPatchMatch::update(cv::Vec2i pos, cv::Vec2i candidate, float distance) {
    if (distance < distances(pos)) {
        distances(pos) = distance;
        offset(pos) = candidate;
    }
}

Rect RandomizedPatchMatch::get_patch(Vec2i const& p, int half) const {
    return Rect(p[1] - half, p[0] - half, 2*half + 1, 2*half + 1);
}

float RandomizedPatchMatch::patch_distance(
    Rect const& origin_patch,
    Rect const& candidate_patch
) const
{
    Mat3b origin(target, origin_patch);
    Mat3b candidate(target, candidate_patch);
    return norm(origin, candidate, NORM_L2SQR);
}

void RandomizedPatchMatch::patch_match_propagation(int parity)
{
    int start_row = (parity == 1)? half : rows-half-1;
    int end_row = (parity == 1)? rows-half : half-1;
    int start_col = (parity == 1)? half : cols-half-1;
    int end_col = (parity == 1)? cols-half: half-1;
    int step = (parity == 1)? 1 : -1;
    for (int row = start_row; row != end_row; row += step) {
        for (int col = start_col; col != end_col; col += step) {
            Vec2i pos(row, col);
            if (mask(pos) == 0)
                continue ;
            Rect patch = get_patch(pos, half);
            for (int dir = 0; dir < 2; ++dir) {
                Vec2i v = dirs(parity, dir);
                Vec2i neighbor = pos + v;
                Vec2i center = offset(neighbor) - v;
                if (center[0] < half
                    || center[0] >= rows-half
                    || center[1] < half
                    || center[1] >= cols-half)
                    continue ;
                if (mask(center) != 0)
                    continue ;
                Rect other = get_patch(center, half);
                float d = patch_distance(patch, other);
                update(pos, center, d);
            }
            target(pos) = target(offset(pos));
        }
        imshow("Inpainting", target);
        waitKey(1);
    }
}

void RandomizedPatchMatch::patch_match_search()
{
    for (int row = half; row < rows-half; ++row) {
        for (int col = half; col < cols-half; ++col) {
            Vec2i pos(row, col);
            if (mask(pos) == 0)
                continue ;
            Rect patch = get_patch(pos, half);
            int squareSize = std::max(rows, cols) / inv_alpha;
            while (squareSize > 1) {
                int infrow = std::max(half, row - squareSize);
                int suprow = std::min(rows - half, row + squareSize) - 1;
                int infcol = std::max(half, col - squareSize);
                int supcol = std::min(cols - half, col + squareSize) - 1;
                int urow = rng.uniform(infrow, suprow + 1);
                int ucol = rng.uniform(infcol, supcol + 1);
                squareSize /= inv_alpha;
                Vec2i center(urow, ucol);
                if (mask(center) != 0)
                    continue ;
                Rect other = get_patch(center, half);
                float d = patch_distance(patch, other);
                update(pos, center, d);
            }
            target(pos) = target(offset(pos));
        }
    }
}

void RandomizedPatchMatch::computeNN(int nbIterations)
{
    for (int iteration_num = 1; iteration_num <= nbIterations; ++iteration_num) {
        patch_match_iteration(iteration_num);
    }
}

void RandomizedPatchMatch::patch_match_iteration(int iteration)
{
    int parity = iteration & 1;
    patch_match_propagation(parity);
    patch_match_search();
}
