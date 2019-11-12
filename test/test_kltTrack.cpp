#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "feature_detector.hpp"
#include "camera.hpp"
#include <vector>

#include <dirent.h>

using namespace cv;
using namespace ssvo;

void DrawEpiLines(const Mat& img_1, const Mat& img_2, std::vector<Point2f>points1, std::vector<Point2f> points2){

    cv::Mat F = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    //首先根据对应点计算出两视图的基础矩阵，基础矩阵包含了两个相机的外参数关系
    std::vector<cv::Vec<float, 3>> epilines1, epilines2;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1);//计算对应点的外极线epilines是一个三元组(a,b,c)，表示点在另一视图中对应的外极线ax+by+c=0;
    cv::computeCorrespondEpilines(points2, 2, F, epilines2);
    //将图片转换为RGB图，画图的时候外极线用彩色绘制
    cv::Mat img1, img2;
    if (img_1.type() == CV_8UC3)
    {
        img_1.copyTo(img1);
        img_2.copyTo(img2);
    }
    else if (img_1.type() == CV_8UC1)
    {
        cvtColor(img_1, img1, COLOR_GRAY2BGR);
        cvtColor(img_2, img2, COLOR_GRAY2BGR);
    }
    else
    {
        std::cout << "unknow img type\n" << std::endl;
        exit(0);
    }

    cv::RNG& rng = theRNG();
    for (int i = 0; i < points2.size(); i++)
    {

        double dist = fabs(points2[i].x * epilines1[i][0] + points2[i].y * epilines1[i][1] + epilines1[i][2]) ;

        Scalar color = Scalar(100*dist/8, 0, 255-100*dist/8);//随机产生颜色

        circle(img2, points2[i], 5, color);//在视图2中把关键点用圆圈画出来，然后再绘制在对应点处的外极线
        line(img2, Point(0, -epilines1[i][2] / epilines1[i][1]), Point(img2.cols, -(epilines1[i][2] + epilines1[i][0] * img2.cols) / epilines1[i][1]), color);



        dist = fabs(points1[i].x * epilines2[i][0] + points1[i].y * epilines2[i][1] + epilines2[i][2]) ;
        color = Scalar(100*dist/8, 0, 255-100*dist/8);//随机产生颜色
        //绘制外极线的时候，选择两个点，一个是x=0处的点，一个是x为图片宽度处
        circle(img1, points1[i], 4, color);
        line(img1, cv::Point(0, -epilines2[i][2] / epilines2[i][1]), cv::Point(img1.cols, -(epilines2[i][2] + epilines2[i][0] * img1.cols) / epilines2[i][1]), color);

        cv::imshow("ep1",img1);
        cv::imshow("ep2",img2);

        cv::imwrite("/home/jh/ep1.png",img1);
        cv::imwrite("/home/jh/ep2.png",img2);


    }
    //cout << " " << endl;
}

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size);

double dist(cv::Point2f point1, cv::Point2f point2)
{
    return sqrt((point1.x-point2.x)*(point1.x-point2.x)+(point1.y-point2.y)*(point1.y-point2.y));
}

int main(int argc, char const *argv[])
{
    if(argc != 4)
    {
        std::cout << "Usge: ./test_feature_detector calib_file config_file path_to_sequence" << std::endl;
        return -1;
    }

    google::InitGoogleLogging(argv[0]);

    std::string dir_name = argv[3];

    ssvo::PinholeCamera::Ptr pinhole_cam = ssvo::PinholeCamera::create(argv[1]);
    Config::file_name_ = std::string(argv[2]);
    int width = pinhole_cam->width();
    int height = pinhole_cam->height();
    int level = Config::imageNLevel()-1;
    int image_border = 8;
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    double fast_min_eigen = Config::fastMinEigen();



    cv::Mat image1 = cv::imread("/home/jh/Datasets/MH_01_easy/cam0/data/1403636582613555456.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (image1.empty()) throw std::runtime_error("Could not open image1 ");

    cv::Mat image2 = cv::imread("/home/jh/Datasets/MH_01_easy/cam0/data/1403636582813555456.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (image2.empty()) throw std::runtime_error("Could not open image2 ");

    std::vector<cv::Mat> image_pyramid;
    computePyramid(image1, image_pyramid, 2, 4, cv::Size(40, 40));

    Corners new_corners, old_corners;

    FastDetector::Ptr fast_detector = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    LOG(WARNING) << "=== This is a FAST corner detector demo ===";
    const int n_trials = 1000;
    double time_accumulator = 0;
    for(int i = 0; i < n_trials; ++i)
    {
        double t = (double)cv::getTickCount();
        fast_detector->detect(image_pyramid, new_corners, old_corners, 100, fast_min_eigen);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
        LOG_EVERY_N(WARNING, n_trials/20) << " i: " << i << ", new_corners: " << new_corners.size();
    }
    LOG(WARNING) << " took " <<  time_accumulator/((double)n_trials)*1000.0
                 << " ms (average over " << n_trials << " trials)." << std::endl;
    cv::Mat kps_img1,kps_img2;
    std::vector<cv::KeyPoint> keypoints1,keypoints2;
    std::vector<cv::Point2f> points1,points2,points2_1;
    std::for_each(new_corners.begin(), new_corners.end(), [&](Corner corner){
        cv::KeyPoint kp(corner.x, corner.y, 0);
        keypoints1.push_back(kp);
        points1.push_back(cv::Point2f(corner.x, corner.y));
    });
    cv::drawKeypoints(image1, keypoints1, kps_img1);


    std::vector<uchar> status1,status2;//特征点跟踪成功标志位
    std::vector<float> errors1,errors2;

    cv::calcOpticalFlowPyrLK(image1,image2,points1,points2,status1,errors1);

    cv::calcOpticalFlowPyrLK(image2,image1,points2,points2_1,status2,errors2);

    std::vector<cv::Point2f> points1_,points2_;


    std::for_each(points2.begin(),points2.end(),[&](Point2f point){
        cv::KeyPoint kp(point.x,point.y,0);
        keypoints2.push_back(kp);
    });

    cv::drawKeypoints(image2, keypoints2, kps_img2);

    cv::Mat image_merge(height,width*2,kps_img2.type());

    Mat ROI1 = image_merge(Rect(0,0,width,height));
    kps_img1.copyTo(ROI1);
    Mat ROI2 = image_merge(Rect(width,0,width,height));
    kps_img2.copyTo(ROI2);


    for (int j = 0; j < status1.size(); ++j) {
        if(status1[j] && status2[j])
        {
            if(dist(points2_1[j],points1[j])<10)
            {
                line(image_merge,points1[j],cv::Point(points2[j].x+width,points2[j].y),cv::Scalar(0,255,0));
                points1_.push_back(points1[j]);
                points2_.push_back(points2[j]);
            }
        }
    }
//    fast_detector->drawGrid(kps_img2, kps_img2);
    cv::imshow("KeyPoints detectByImage1", kps_img1);
    cv::imshow("KeyPoints detectByImage2", kps_img2);
    cv::imshow("KeyPoints merge", image_merge);
//    cv::imwrite("/home/jh/klt_before.png",image_merge);


    DrawEpiLines(image1,image2,points1_,points2_);


    cv::waitKey(0);

    return 0;
}

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size)
{
    LOG_ASSERT(scale_factor > 1.0);
    LOG_ASSERT(!image.empty());

    image_pyramid.resize(level + 1);

    image_pyramid[0] = image.clone();
    for(int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i - 1].cols / scale_factor), round(image_pyramid[i - 1].rows / scale_factor));

        if(size.height < min_size.height || size.width < min_size.width)
        {
            image_pyramid.resize(level);
            return level-1;
        }

        cv::resize(image_pyramid[i - 1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return level;
}