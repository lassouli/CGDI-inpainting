#include <iostream>

#include "patch_match.hpp"

using namespace cv;

typedef Rect Patch;

Patch get_patch(Vec2i const& p, int half) {
    return Rect(p[1] - half, p[0] - half, 2*half + 1, 2*half + 1);
}

float patch_distance(
    Mat3b const& target
    , Patch const& origin_patch
    , Patch const& candidate_patch
)
{
    Mat3b origin(target, origin_patch);
    Mat3b candidate(target, candidate_patch);
    return norm(origin, candidate, NORM_L2SQR);
}

void patch_match_propagation(
    Mat1b const& mask
    , Mat3b& target
    , Mat1f& distances
    , Mat2i& offset
    , int patchSize
    , Matx<Vec2i,1,2> const& dirs
)
{
    int half = patchSize / 2;
    int rows = target.rows;
    int cols = target.cols;
    for (int row = half; row < rows-half; ++row) {
        for (int col = half; col < cols-half; ++col) {
            Vec2i pos(row, col);
            if (mask(pos) == 0)
                continue ;
            Patch patch = get_patch(pos, half);
            for (int dir = 0; dir < 2; ++dir) {
                Vec2i v = dirs(0, dir);
                Vec2i neighbor = pos + v;
                Vec2i center = offset(neighbor) - v;
                if (center[0] < half || center[0] >= rows-half || center[1] < half || center[1] >= cols-half)
                    continue ;
                if (mask(center) != 0)
                    continue ;
                Patch other = get_patch(center, half);
                float d = patch_distance(target, patch, other);
                if (d < distances(pos)) {
                    distances(pos) = d;
                    offset(pos) = center;
                    //target(pos) = target(offset(pos));
                }
            }
        }
    }
    target.forEach([&](Vec3b& pixel, const int pos[]){
        pixel = target(offset(pos[0],pos[1]));
    });
}

void
patch_match_iteration(
    Mat1b const& mask
    , Mat3b& target
    , Mat1f& distances
    , Mat2i& offset
    , int patchSize
    , int iteration)
{
    static Matx<Vec2i,2,2> const dirs(Vec2i(-1,0), Vec2i(0,-1), Vec2i(1,0), Vec2i(0,1));
    int parity = iteration & 1;
    patch_match_propagation(mask, target, distances, offset, patchSize, dirs.row(parity));
    //patch_match_search();
}

void
patch_match(
    Mat3b const& source,
    Mat1b const& mask,
    Mat3b& target,
    int patchSize,
    const int nbIterations)
{
    RNG rng( 0xFFFFFFFF );
    namedWindow("Inpainting", WINDOW_AUTOSIZE);
    int half = patchSize / 2;
    int rows = source.rows;
    int cols = source.cols;
    source.copyTo(target);
    Mat1f distances = Mat1f::zeros(rows, cols);
    (Mat1f(rows, cols, FLT_MAX)).copyTo(distances, mask);
    Mat2i offset(rows, cols);
    offset.forEach([&](Vec2i& pixel, const int pos[]){
        pixel[0] = pos[0];
        if (pixel[0] < half)
            pixel[0] = half;
        if (pixel[0] >= rows - half)
            pixel[0] = rows - half - 1;
        pixel[1] = pos[1];
        if (pixel[1] < half)
            pixel[1] = half;
        if (pixel[1] >= cols - half)
            pixel[1] = cols - half - 1;
        while (mask(pixel[0], pixel[1]) != 0) {
            pixel[0] = rng.uniform(half, rows - half);
            pixel[1] = rng.uniform(half, cols - half);
        }
    });
    target.forEach([&](Vec3b& pixel, const int pos[]){
        pixel = source(offset(pos[0],pos[1]));
    });
    imshow("Inpainting", target);
    std::cout << "Wait... " << std::endl;
    waitKey(4000);
    for (int iteration_num = 0; iteration_num < nbIterations; ++iteration_num) {
        patch_match_iteration(mask, target, distances, offset, patchSize, iteration_num);
        imshow("Inpainting", target);
        std::cout << "Wait... " << std::endl;
        waitKey(4000);
    }
}
