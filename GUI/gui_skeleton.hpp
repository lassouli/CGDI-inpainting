#include <opencv2/opencv.hpp>
#include <iostream>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Mask Creator"

using namespace cv;

int mask_from_scratch(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Use: Create_mask <image>" << "\n" ;
    return 1 ;
  }
  Mat image = imread(argv[1], IMREAD_COLOR) ;
  Mat frame = image.clone() ;
  int h = image.rows ;
  int w = image.cols ;
  Mat masque(h, w, CV_8UC3, Scalar(0,0,0)) ;
  namedWindow(WINDOW_NAME);
  cvui::init(WINDOW_NAME) ;
  int cursor_radius = 2 ; //How many pixels are selected when the mask is drawn ?
  Vec3b color = image.at<Vec3b>(Point(0, 0));
  Vec3b color_neg ;
  color[0] = 255 ;
  color[1] = 255 ;
  color[2] = 255 ;
  std::vector<std::vector<int> > untouched (w, std::vector<int> (h, 1)) ;
  while (true) {
    // cvui::window(frame, 10, 50, 180, 180, "Settings");
    cvui::image(frame, 0, 0, image);
    if (cvui::mouse(cvui::LEFT_BUTTON, cvui::IS_DOWN)) {
      cv::Point cursor = cvui::mouse() ;
      for (int i = -1*cursor_radius; i <= cursor_radius; i++) {
        for (int j = -1*cursor_radius; j <= cursor_radius; j++) {
          if ((cursor.x + i) < w && (cursor.y + j) < h && 0 <= (cursor.x + i) && 0 <= (cursor.y + j) && untouched[cursor.x + i][cursor.y + j]) {
            untouched[cursor.x + i][cursor.y + j] = 0 ;
            masque.at<Vec3b>(Point(cursor.x + i,cursor.y + j)) = color ;
            color_neg = image.at<Vec3b>(Point(cursor.x + i,cursor.y + j));
            color_neg[0] = 255 - color_neg[0] ;
            color_neg[1] = 255 - color_neg[1] ;
            color_neg[2] = 255 - color_neg[2] ;
            image.at<Vec3b>(Point(cursor.x + i,cursor.y + j)) = color_neg ;
          }
        }
      }
     }




    cvui::update();
    cvui::imshow(WINDOW_NAME, frame);
    if (cv::waitKey(5) == 27) {
      break;
    }
  }
  imwrite("masque.png", masque) ;
  return 0;
}
