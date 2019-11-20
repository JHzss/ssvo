#ifndef SSVO_IMUDATA_HPP
#define SSVO_IMUDATA_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using Eigen::Matrix3d;
using Eigen::Vector3d;
namespace ssvo
{


class IMAGEData
{
public:
    IMAGEData(const cv::Mat& image,double timestamps);
    cv::Mat image_;
    double timestamps_;
};

class IMUData
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // covariance of measurement
    static Matrix3d _gyrMeasCov;
    static Matrix3d _accMeasCov;
    static Matrix3d getGyrMeasCov(void) {return _gyrMeasCov;}
    static Matrix3d getAccMeasCov(void) {return _accMeasCov;}

    // covariance of bias random walk
    static Matrix3d _gyrBiasRWCov;
    static Matrix3d _accBiasRWCov;
    static Matrix3d getGyrBiasRWCov(void) {return _gyrBiasRWCov;}
    static Matrix3d getAccBiasRWCov(void) {return _accBiasRWCov;}

    // 随机游走的均值
    static double _gyrBiasRw2;
    static double _accBiasRw2;
    static double getGyrBiasRW2(void) {return _gyrBiasRw2;}
    static double getAccBiasRW2(void) {return _accBiasRw2;}


    IMUData(const double& gx, const double& gy, const double& gz,
            const double& ax, const double& ay, const double& az,
            const double& t);
    IMUData(const Vector3d& gyr,const Vector3d& acc,const double& time);

    // Raw data of imu's
    Vector3d _g;    //gyr data
    Vector3d _a;    //acc data
    double _t;      //timestamp
};

}

#endif //SSVO_IMUDATA_HPP
