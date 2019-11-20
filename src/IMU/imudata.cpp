#include "imudata.hpp"
#include "global.hpp"

namespace ssvo
{

    IMAGEData::IMAGEData(const cv::Mat &image, double timestamps):image_(image),timestamps_(timestamps) {}

    // 静态成员变量的定义
    double IMUData::_gyrBiasRw2;
    double IMUData::_accBiasRw2;

    Eigen::Matrix3d IMUData::_gyrMeasCov;
    Eigen::Matrix3d IMUData::_accMeasCov;

    Eigen::Matrix3d IMUData::_gyrBiasRWCov;
    Eigen::Matrix3d IMUData::_accBiasRWCov;

    IMUData::IMUData(const double& gx, const double& gy, const double& gz,
                     const double& ax, const double& ay, const double& az,
                     const double& t) : _g(gx,gy,gz), _a(ax,ay,az), _t(t) {}
    IMUData::IMUData(const Vector3d &gyr, const Vector3d &acc, const double &time) : _g(gyr),_a(acc),_t(time) {}
}