#include <iostream>
#include <algorithm>

#include "patch_match.hpp"

using namespace cv;
using namespace std;

Matx<Vec2i,2,2> const RandomizedPatchMatch::dirs{
    Vec2i(1,0)
    , Vec2i(0,1)
    , Vec2i(-1,0)
    , Vec2i(0,-1)
};

PatchZone::PatchZone(Vec2i _center, int patchSize):
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
        pixel = source(offset(pos[0], pos[1]));
    });
    onionPeelInitialization();
    distances.forEach([&](float& d, const int coords[]){
        Vec2i pos(coords[0], coords[1]);
        PatchZone ptz(pos, patchSize);
        if (mask(pos) != 0 || !is_inside(ptz))
            return ;
        Rect origin_patch = get_patch(pos, half);
        Rect candidate_patch = get_patch(offset(pos), half);
        d = patch_distance(origin_patch, candidate_patch);
    });
}

void RandomizedPatchMatch::final_reconstruction() {
    target.forEach([&](Vec3b& pixel, const int coord[]){
        Vec2i pos(coord[0], coord[1]);
        if (mask(pos) == 0)
            return ;
        Rect origin = get_patch(pos, half);
        float best_distance = FLT_MAX;
        for (int row = origin.y; row < origin.y+origin.height; ++row) {
            for (int col = origin.x; col < origin.x+origin.width; ++col) {
                Vec2i q(row, col);
                Vec2i alter_ego = pos - q + offset(q);
                if (!is_inside(alter_ego)||mask(alter_ego))
                    continue ;
                if (distances(q) < best_distance) {
                    best_distance = distances(q);
                    pixel = target(alter_ego);
                }
            }
        }
    });
}

void RandomizedPatchMatch::reconstruction() {
    vector<float> sorted_dists;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (mask(row, col))
                sorted_dists.push_back(distances(row, col));
        }
    }
    sort(begin(sorted_dists), end(sorted_dists));
    float sigma = sorted_dists[int(sorted_dists.size() * 0.75)];
    Mat1f s; distances.copyTo(s);
    s = -s / (sigma*sigma);
    exp(s, s);
    target.forEach([&](Vec3b& pixel, const int coord[]){
        Vec2i pos(coord[0], coord[1]);
        if (mask(pos) == 0)
            return ;
        Rect origin = get_patch(pos, half);
        float sum = 0.f;
        Vec3f color(0,0,0);
        for (int row = origin.y; row < origin.y+origin.height; ++row) {
            for (int col = origin.x; col < origin.x+origin.width; ++col) {
                Vec2i q(row, col);
                Vec2i alter_ego = pos - q + offset(q);
                if (!is_inside(alter_ego))
                    continue ;
                float sq = s(q);
                color += sq * static_cast<Vec3f>(target(alter_ego));
                sum += sq;
            }
        }
        color /= sum;
        pixel = static_cast<Vec3b>(color);
    });
}

void RandomizedPatchMatch::onionPeelInitialization() {
    Mat1b available; bitwise_not(mask, available);
    Mat2i outborder; Mat2i inborder;
    auto const add_neighbours = [&](Vec2i const& pos){
        for (int i = 0; i < 4; ++i) {
            Vec2i v = pos + dirs(i/2, i%2);
            if (is_inside(v) && available(v) == 0) {
                inborder.push_back(v);
                available(v) = available(pos);
            }
        }
    };
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            Vec2i pos(row, col);
            if (distances(pos) != 0)
                continue ;
            add_neighbours(pos);
        }
    }
    auto const swap_queue = [&](){
        outborder = inborder;
        inborder.release();
    };
    swap_queue();
    while (!outborder.empty()) {
        for (auto const& pos : outborder) {
            available(pos) = 0;
        }
        Mat3b colors;
        for (auto const& pos : outborder) {
            Rect origin = get_patch(pos, half);
            float sum = 0.f;
            Vec3f color(0,0,0);
            for (int row = origin.y; row < origin.y+origin.height; ++row) {
                for (int col = origin.x; col < origin.x+origin.width; ++col) {
                    Vec2i q(row, col);
                    if (available(q) == 0)
                        continue ;
                    Vec2i alter_ego = q;// pos - q + offset(q);
                    if (!is_inside(alter_ego))
                        continue ;
                    color += static_cast<Vec3f>(target(alter_ego));
                    sum += 1.0f;
                }
            }
            color /= sum;
            colors.push_back(Vec3b(color[0], color[1], color[2]));
        }
        for (int i = 0; i < outborder.rows; ++i) {
            available(outborder(i)) = 255;
            target(outborder(i)) = colors(i);
        }
        for (auto const& pos : outborder) {
            add_neighbours(pos);
        }
        swap_queue();
        imshow("Inpainting", target);
        waitKey(20);
    }
}

void RandomizedPatchMatch::update(Vec2i pos, Vec2i candidate, float distance) {
    if (distance < distances(pos)) {
        distances(pos) = distance;
        offset(pos) = candidate;
    }
}

Rect RandomizedPatchMatch::get_patch(Vec2i const& p, int half) const {
    Rect patch(p[1] - half, p[0] - half, 2*half + 1, 2*half + 1);
    patch.x = max(0, patch.x);
    patch.y = max(0, patch.y);
    patch.width = min(cols - patch.x, patch.width);
    patch.height = min(rows - patch.y, patch.height);
    return patch;
}

float RandomizedPatchMatch::patch_distance(
    Rect const& origin_patch,
    Rect const& candidate_patch
) const
{
    Mat3b origin(target, origin_patch);
    Mat3b candidate(target, candidate_patch);
    return norm(origin, candidate, NORM_L2SQR) / (origin_patch.width * origin_patch.height);
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
        }
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
            int squareSize = max(rows, cols) / inv_alpha;
            while (squareSize > 1) {
                int infrow = max(half, row - squareSize);
                int suprow = min(rows - half, row + squareSize) - 1;
                int infcol = max(half, col - squareSize);
                int supcol = min(cols - half, col + squareSize) - 1;
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
        }
    }
}

void RandomizedPatchMatch::computeNN(int nbIterations)
{
    cout << "progression ";
    for (int iteration_num = 1; iteration_num <= nbIterations; ++iteration_num) {
        std::cout << "|" << flush;
        patch_match_iteration(iteration_num);
        reconstruction();
        imshow("Inpainting", target);
        waitKey(2000);
    }
    final_reconstruction();
    cout << endl;
}

void RandomizedPatchMatch::patch_match_iteration(int iteration)
{
    int parity = iteration & 1;
    patch_match_propagation(parity);
    patch_match_search();
}
