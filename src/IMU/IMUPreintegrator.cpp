//
// Created by jh on 19-1-16.
//

#include "IMUPreintegrator.hpp"

namespace ssvo
{

IMUPreintegrator::IMUPreintegrator(const IMUPreintegrator& pre):
        _delta_P(pre._delta_P),
        _delta_V(pre._delta_V),
        _delta_R(pre._delta_R),
        _J_P_Biasg(pre._J_P_Biasg),
        _J_P_Biasa(pre._J_P_Biasa),
        _J_V_Biasg(pre._J_V_Biasg),
        _J_V_Biasa(pre._J_V_Biasa),
        _J_R_Biasg(pre._J_R_Biasg),
        _cov_P_V_Phi(pre._cov_P_V_Phi),
        _delta_time(pre._delta_time)
{}

IMUPreintegrator::IMUPreintegrator()
{
    // delta measurements, position/velocity/rotation(matrix)
    _delta_P.setZero();    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    _delta_V.setZero();    // V_k+1 = V_k + R_k*a_k*dt
    _delta_R.setIdentity();    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x

    // jacobian of delta measurements w.r.t bias of gyro/acc
    _J_P_Biasg.setZero();     // position / gyro
    _J_P_Biasa.setZero();     // position / acc
    _J_V_Biasg.setZero();     // velocity / gyro
    _J_V_Biasa.setZero();     // velocity / acc
    _J_R_Biasg.setZero();   // rotation / gyro

    // noise covariance propagation of delta measurements
    _cov_P_V_Phi.setZero();

    _delta_time = 0;
}

void IMUPreintegrator::reset()
{
    // delta measurements, position/velocity/rotation(matrix)
    _delta_P.setZero();    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    _delta_V.setZero();    // V_k+1 = V_k + R_k*a_k*dt
    _delta_R.setIdentity();    // R_k+1 = R_k*exp(w_k*dt).     note: Rwc, Rwc'=Rwc*[w_body]x

    // jacobian of delta measurements w.r.t bias of gyro/acc
    _J_P_Biasg.setZero();     // position / gyro
    _J_P_Biasa.setZero();     // position / acc
    _J_V_Biasg.setZero();     // velocity / gyro
    _J_V_Biasa.setZero();     // velocity / acc
    _J_R_Biasg.setZero();   // rotation / gyro

    // noise covariance propagation of delta measurements
    _cov_P_V_Phi.setZero();

    _delta_time = 0;

}

// incrementally update 1)delta measurements, 2)jacobians, 3)covariance matrix
// acc: acc_measurement - bias_a, last measurement!! not current measurement
// omega: gyro_measurement - bias_g, last measurement!! not current measurement
void IMUPreintegrator::update(const Eigen::Vector3d& omega, const Eigen::Vector3d& acc, const double& dt)
{
    double dt2 = dt*dt;

    Eigen::Matrix3d dR = Expmap(omega*dt);
    Eigen::Matrix3d Jr = JacobianR(omega*dt);

    // noise covariance propagation of delta measurements
    // err_k+1 = A*err_k + B*err_gyro + C*err_acc
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double,9,9> A = Eigen::Matrix<double,9,9>::Identity() ;
    A.block<3,3>(6,6) = dR.transpose();
    A.block<3,3>(3,6) = -_delta_R*skew(acc)*dt;
    A.block<3,3>(0,6) = -0.5*_delta_R*skew(acc)*dt2;
    A.block<3,3>(0,3) = I3x3*dt;
    Eigen::Matrix<double,9,3> Bg = Eigen::Matrix<double,9,3>::Zero();
    Bg.block<3,3>(6,0) = Jr*dt;
    Eigen::Matrix<double,9,3> Ca = Eigen::Matrix<double,9,3>::Zero();
    Ca.block<3,3>(3,0) = _delta_R*dt;
    Ca.block<3,3>(0,0) = 0.5*_delta_R*dt2;
    _cov_P_V_Phi = A*_cov_P_V_Phi*A.transpose() +
                   Bg*IMUData::getGyrMeasCov()*Bg.transpose() +
                   Ca*IMUData::getAccMeasCov()*Ca.transpose();


    // jacobian of delta measurements w.r.t bias of gyro/acc
    // update P first, then V, then R
    _J_P_Biasa += _J_V_Biasa*dt - 0.5*_delta_R*dt2;
    _J_P_Biasg += _J_V_Biasg*dt - 0.5*_delta_R*skew(acc)*_J_R_Biasg*dt2;
    _J_V_Biasa += -_delta_R*dt;
    _J_V_Biasg += -_delta_R*skew(acc)*_J_R_Biasg*dt;
    _J_R_Biasg = dR.transpose()*_J_R_Biasg - Jr*dt;
//    std::cout<<"A:------------------------------------"<<std::endl<<A<<std::endl;
//    std::cout<<"Bg:------------------------------------"<<std::endl<<Bg<<std::endl;
//    std::cout<<"Ca:------------------------------------"<<std::endl<<Ca<<std::endl;
//    std::cout<<"dt:------------------------------------"<<std::endl<<dt<<std::endl;
//    std::cout<<"_delta_time:------------------------------------"<<std::endl<<_delta_time<<std::endl;
//    std::cout<<"dR:------------------------------------"<<std::endl<<dR<<std::endl;
//    std::cout<<"Jr:------------------------------------"<<std::endl<<Jr<<std::endl;
//    std::cout<<"_J_R_Biasg:------------------------------------"<<std::endl<<_J_R_Biasg<<std::endl;
//        std::cout<<"_cov_P_V_Phi:------------------------------------"<<std::endl<<_cov_P_V_Phi<<std::endl;

    // delta measurements, position/velocity/rotation(matrix)
    // update P first, then V, then R. because P's update need V&R's previous state
    _delta_P += _delta_V *dt + 0.5*_delta_R*acc*dt2;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
    _delta_V += _delta_R*acc*dt;
    _delta_R = normalizeRotationM(_delta_R*dR);  // normalize rotation, in case of numerical error accumulation

    // delta time
    _delta_time += dt;
}

}