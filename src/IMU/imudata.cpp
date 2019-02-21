#include "imudata.hpp"
#include "global.hpp"

namespace ssvo
{
    double IMUData::_gyrBiasRw2 = 2.0e-5*2.0e-5/**10*/;  //2e-12*1e3
    double IMUData::_accBiasRw2 = 5.0e-3*5.0e-3/**10*/;  //4.5e-8*1e2

    Matrix3d IMUData::_gyrMeasCov = Matrix3d::Identity()*1.7e-4*1.7e-4/0.005/**100*/;       // sigma_g * sigma_g / dt, ~6e-6*10
    Matrix3d IMUData::_accMeasCov = Matrix3d::Identity()*2.0e-3*2.0e-3/0.005*100;       // sigma_a * sigma_a / dt, ~8e-4*10

// covariance of bias random walk
    Matrix3d IMUData::_gyrBiasRWCov = Matrix3d::Identity()*_gyrBiasRw2;     // sigma_gw * sigma_gw * dt, ~2e-12
    Matrix3d IMUData::_accBiasRWCov = Matrix3d::Identity()*_accBiasRw2;     // sigma_aw * sigma_aw * dt, ~4.5e-8

    IMUData::IMUData(const double& gx, const double& gy, const double& gz,
                     const double& ax, const double& ay, const double& az,
                     const double& t) :
            _g(gx,gy,gz), _a(ax,ay,az), _t(t)
    {
    }
    IMUData::IMUData(const Vector3d &gyr, const Vector3d &acc, const double &time) :
    _g(gyr),_a(acc),_t(time)
    {
    }
}