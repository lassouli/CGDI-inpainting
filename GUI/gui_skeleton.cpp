#include <opencv2/opencv.hpp>
#include <iostream>

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Mask Creator"

using namespace cv;

int main(int argc, char const *argv[]) {
  if (argc != 2) {
    std::cout << "Use: Create_mask <image>" << "\n" ;
    return 1 ;
  }
  Mat image = imread(argv[1], IMREAD_COLOR) ;
  Mat frame = image.clone() ;
  int h = image.cols ;
  int w = image.rows ;
  Mat masque(w, h, CV_8UC3, Scalar(0,0,0)) ;
  namedWindow(WINDOW_NAME);
  cvui::init(WINDOW_NAME) ;
  int cursor_radius = 2 ; //How many pixels are selected when the mask is drawn ?
  Vec3b color = image.at<Vec3b>(Point(0, 0));
  color[0] = 255 ;
  color[1] = 255 ;
  color[2] = 255 ;
  while (true) {
    // cvui::window(frame, 10, 50, 180, 180, "Settings");
    cvui::image(frame, 10, 10, image);
    if (cvui::mouse(cvui::LEFT_BUTTON, cvui::IS_DOWN)) {
      cv::Point cursor = cvui::mouse() ;
      for (int i = -1*cursor_radius; i <= cursor_radius; i++) {
        if ((cursor.x + i) <= w && (cursor.y + i) <= h)
          masque.at<Vec3b>(Point(cursor.x + i,cursor.y + i)) = color ;
        if ((cursor.x - i) <= w && (cursor.y + i) <= h)
          masque.at<Vec3b>(Point(cursor.x - i,cursor.y + i)) = color ;

      }
     }




    cvui::update();
    cv::imshow(WINDOW_NAME, frame);
    if (cv::waitKey(30) == 27) {
      break;
    }
  }
  imwrite("masque.png", masque) ;
  return 0;
}
