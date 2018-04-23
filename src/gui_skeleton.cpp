#include <opencv2/opencv.hpp>
#include <iostream>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include "gui_skeleton.hpp"

#define WINDOW_NAME "CVUI Mask Creator"

using namespace cv;

void mask_from_scratch(Mat3b const& original, Mat1b& mask) {
  Mat image = original.clone();
  Mat frame = image.clone() ;
  int h = original.rows;
  int w = original.cols;
  mask.create(h, w); mask.setTo(0);
  namedWindow(WINDOW_NAME);
  cvui::init(WINDOW_NAME) ;
  int cursor_radius = 2 ;
  std::vector<std::vector<int> > untouched (w, std::vector<int> (h, 1)) ;
  while (true) {
    cvui::image(frame, 0, 0, image);
    if (cvui::mouse(cvui::LEFT_BUTTON, cvui::IS_DOWN)) {
      cv::Point cursor = cvui::mouse() ;
      for (int i = -1*cursor_radius; i <= cursor_radius; i++) {
        for (int j = -1*cursor_radius; j <= cursor_radius; j++) {
          if ((cursor.x + i) < w
          && (cursor.y + j) < h
          && 0 <= (cursor.x + i) && 0 <= (cursor.y + j)
          && untouched[cursor.x + i][cursor.y + j]) {
            untouched[cursor.x + i][cursor.y + j] = 0;
            mask(Point(cursor.x + i,cursor.y + j)) = 255;
            Vec3b color_neg = image.at<Vec3b>(Point(cursor.x + i,cursor.y + j));
            color_neg = Vec3b(255, 255, 255) - color_neg;
            image.at<Vec3b>(Point(cursor.x + i,cursor.y + j)) = color_neg ;
          }
        }
      }
    }
    cvui::update();
    cvui::imshow(WINDOW_NAME, frame);
    char key = cv::waitKey(1);
    if (key == 27) {
      break;
    } else if (key == '+') {
      cursor_radius += 5;
      cursor_radius = std::max(cursor_radius, 0);
    } else if (key == '-') {
      cursor_radius -= 5;
      cursor_radius = std::min(cursor_radius, 100);
    }
  }
}
