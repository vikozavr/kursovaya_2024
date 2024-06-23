#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace cv;
using namespace std;

struct GlobalDict {
    Mat clear_frame;
    Mat final_frame;
    bool start_flag = false;
    bool performance_flag = false;
    int vegie_var = 0;
    int vegie_counter = 0;
    int width = 0;
    int height = 0;
    string current_vegie = "none";
};

GlobalDict global_dict;

Mat img_blur(const Mat& img) {
    Mat blurred;
    medianBlur(img, blurred, 15);
    blur(blurred, blurred, Size(5, 5));
    return blurred;
}

Mat img_gray(const Mat& img) {
    Mat grayed;
    cvtColor(img, grayed, COLOR_RGB2GRAY);
    return grayed;
}

Mat img_thresh(const Mat& img) {
    Mat thresh;
    threshold(img, thresh, 160, 255, THRESH_BINARY);
    return thresh;
}

Mat img_edges(const Mat& img) {
    Mat edged;
    Canny(img, edged, 60, 125);
    return edged;
}

void drawing_vegies(vector<int> boxes_lst, string txt_var) {
    if (abs(global_dict.vegie_counter) == 20) {
        cout << "I'm certain that it's a " << txt_var << endl;
        global_dict.vegie_counter = 0;
        global_dict.current_vegie = txt_var;
    }

    if (!boxes_lst.empty()) {
        int left = boxes_lst[2], top = boxes_lst[3];
        int right = boxes_lst[0], bottom = boxes_lst[1];

        rectangle(global_dict.final_frame, Point(left, top), Point(right, bottom), Scalar(255, 0, 0), 2);
        putText(global_dict.final_frame, txt_var, Point(left, bottom), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
    }
}

void image_analysis() {
    Mat starting_img = global_dict.clear_frame.clone();

    if ((global_dict.vegie_var != 0) || (global_dict.performance_flag)) {
        Mat img_blurred = img_blur(starting_img);
        Mat img_grayed = img_gray(img_blurred);
        Mat img2analyse = img_thresh(img_grayed);
        Mat copy4mask = global_dict.clear_frame.clone();

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(img2analyse, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        Mat mask = Mat::zeros(img2analyse.size(), CV_8UC1);
        vector<int> boxes;
        vector<int> mean_clr_lst;

        for (size_t i = 0; i < contours.size(); i++) {
            mask.setTo(Scalar(0));
            Rect bounding_rect = boundingRect(contours[i]);

            if (global_dict.width * 0.2 <= (bounding_rect.x + bounding_rect.width) / 2 &&
                (bounding_rect.x + bounding_rect.width) / 2 <= global_dict.width * 0.8) {
                
                if (abs(bounding_rect.width - bounding_rect.height) >= 15 && bounding_rect.width > 100 &&
                    bounding_rect.height > 100 && bounding_rect.height <= global_dict.height * 0.75) {

                    if (boxes.empty() || (boxes[2] - boxes[0]) < bounding_rect.width) {
                        rectangle(mask, bounding_rect, Scalar(255), -1);
                        Mat masked;
                        copy4mask.copyTo(masked, mask);
                        boxes = { bounding_rect.x, bounding_rect.y, bounding_rect.x + bounding_rect.width, bounding_rect.y + bounding_rect.height };
                        Scalar mean_clr = mean(masked);
                        mean_clr_lst = { static_cast<int>(round(mean_clr[0])), static_cast<int>(round(mean_clr[1])), static_cast<int>(round(mean_clr[2])) };
                    }
                }
            }
        }

        string txt_var = "none";
        if (!mean_clr_lst.empty()) {
            cout << mean_clr_lst[0] << ", " << mean_clr_lst[1] << ", " << mean_clr_lst[2] << endl;
            if (max(mean_clr_lst[0], max(mean_clr_lst[1], mean_clr_lst[2])) == mean_clr_lst[0] && 
                !(mean_clr_lst[1] / mean_clr_lst[0] >= 0.7 || mean_clr_lst[2] / mean_clr_lst[0] >= 0.7)) {
                
                if (global_dict.vegie_var == 1 || global_dict.vegie_var == 3) {
                    txt_var = "Tomato";
                    global_dict.vegie_counter++;
                    drawing_vegies(boxes, txt_var);
                }
            } else {
                if (global_dict.vegie_var == 2 || global_dict.vegie_var == 3) {
                    txt_var = "Eggplant";
                    global_dict.vegie_counter--;
                    drawing_vegies(boxes, txt_var);
                }
            }
        } else {
            global_dict.current_vegie = "none";
        }
    }
}

void image_callback(const Mat& img) {
    Mat clear_frame = img(Range(int(global_dict.height * 0.05), int(global_dict.height * 0.9)), Range::all());
    clear_frame.convertTo(clear_frame, -1, 1.5, 1);
    cvtColor(clear_frame, clear_frame, COLOR_BGR2RGB);

    global_dict.clear_frame = clear_frame;
    global_dict.final_frame = clear_frame.clone();

    image_analysis();

    imshow("Video", global_dict.final_frame);
    waitKey(1);
}

void start() {
    VideoCapture cap("/Users/andreyboriskin/kursach/video.mp4");
    Mat frame;

    cap >> frame;
    global_dict.height = frame.rows;
    global_dict.width = frame.cols;

    namedWindow("Video", WINDOW_AUTOSIZE);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        image_callback(frame);
    }

    destroyAllWindows();
}

int main() {
    start();
    return 0;
}
