#ifndef SSVO_ADD_MATH_HPP
#define SSVO_ADD_MATH_HPP

#include <Eigen/Core>
#include <sophus/so3.hpp>

namespace ssvo
{
const double SMALL_EPS = 1e-10;

class Add_math
{
public:
    static Matrix3d JacobianR(const Vector3d& w)
    {
        Matrix3d Jr = Matrix3d::Identity();
        double theta = w.norm();
        if(theta<0.00001)
        {
            return Jr;// = Matrix3d::Identity();
        }
        else
        {
            Vector3d k = w.normalized();  // k - unit direction vector of w
            Matrix3d K = Sophus::SO3d::hat(k);
            Jr =   Matrix3d::Identity()
                   - (1-cos(theta))/theta*K
                   + (1-sin(theta)/theta)*K*K;
        }
        return Jr;
    }
    static Matrix3d JacobianRInv(const Vector3d& w)
    {
        Matrix3d Jrinv = Matrix3d::Identity();
        double theta = w.norm();

        // very small angle
        if(theta < 0.00001)
        {
            return Jrinv;
        }
        else
        {
            Vector3d k = w.normalized();  // k - unit direction vector of w
            Matrix3d K = Sophus::SO3d::hat(k);
            Jrinv = Matrix3d::Identity()
                    + 0.5*Sophus::SO3d::hat(w)
                    + ( 1.0 - (1.0+cos(theta))*theta / (2.0*sin(theta)) ) *K*K;
        }
        return Jrinv;
    }

};




}

#endif //SSVO_ADD_MATH_HPP
