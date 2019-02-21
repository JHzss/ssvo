#ifndef SSVO_NAVSTATE_HPP
#define SSVO_NAVSTATE_HPP

#include "Eigen/Geometry"
#include "sophus/so3.hpp"

namespace ssvo
{

using Eigen::Vector3d;
using Eigen::Matrix3d;
//using namespace g2o;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


    //! 坐标系：Twb（from body to world）
class NavState
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NavState();
    NavState(const NavState& _ns);

    //! from body to world (Twb)
    //Quaterniond Get_qR(){return _qR;}     // rotation
    Sophus::SO3d Get_R() const{return _R;}
    //Matrix3d Get_RotMatrix(){return _qR.toRotationMatrix();}   // rotation matrix
    Matrix3d Get_RotMatrix() const{return _R.matrix();}
    Vector3d Get_P() const{return _P;}         // position
    Vector3d Get_V() const{return _V;}         // velocity
    void Set_Pos(const Vector3d &pos){_P = pos;}
    void Set_Vel(const Vector3d &vel){_V = vel;}
    void Set_Rot(const Matrix3d &rot){_R = Sophus::SO3d(rot);}
    void Set_Rot(const Sophus::SO3d &rot){_R = rot;}

    Vector3d Get_BiasGyr() const{return _BiasGyr;}   // bias of gyroscope, keep unchanged after init and during optimization
    Vector3d Get_BiasAcc() const{return _BiasAcc;}   // bias of accelerometer
    void Set_BiasGyr(const Vector3d &bg){_BiasGyr = bg;}
    void Set_BiasAcc(const Vector3d &ba){_BiasAcc = ba;}

    Vector3d Get_dBias_Gyr() const{return _dBias_g;}  // delta bias of gyroscope, init as 0, change during optimization
    Vector3d Get_dBias_Acc() const{return _dBias_a;}  // delta bias of accelerometer
    void Set_DeltaBiasGyr(const Vector3d &dbg){_dBias_g = dbg;}
    void Set_DeltaBiasAcc(const Vector3d &dba){_dBias_a = dba;}

private:
    /*
     * Note:
     * don't add pointer as member variable.
     * operator = is used in g2o
    */

    Vector3d _P;         // position
    Vector3d _V;         // velocity
    Sophus::SO3d _R;     // rotation

    // keep unchanged during optimization
    Vector3d _BiasGyr;   // bias of gyroscope
    Vector3d _BiasAcc;   // bias of accelerometer

    // update below term during optimization
    Vector3d _dBias_g;  // delta bias of gyroscope, correction term computed in optimization
    Vector3d _dBias_a;  // delta bias of accelerometer

};
}



#endif //SSVO_NAVSTATE_HPP
