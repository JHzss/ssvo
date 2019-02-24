#ifndef _OPTIMIZER_HPP_
#define _OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "map_point.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "global.hpp"
#include "add_math.hpp"
//#include "loop_closure.hpp"

namespace ssvo {

//class LoopClosure;

class Optimizer: public noncopyable
{
public:
    typedef std::map<KeyFrame::Ptr,Sophus::Sim3d ,std::less<KeyFrame::Ptr>,
            Eigen::aligned_allocator<std::pair<const KeyFrame::Ptr, Sophus::Sim3d> > > KeyFrameAndPose;

    static void globleBundleAdjustment(const Map::Ptr &map, int max_iters,const uint64_t nLoopKF = 0, bool report=false, bool verbose=false);

    static void motionOnlyBundleAdjustment(const Frame::Ptr &frame, bool use_seeds, bool reject=false, bool report=false, bool verbose=false);

    static void localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size=10, int min_shared_fts=50, bool report=false, bool verbose=false);

    static int optimizeSim3(KeyFrame::Ptr pKF1, KeyFrame::Ptr pKF2, std::vector<MapPoint::Ptr> &vpMatches1,
                                  Sophus::Sim3d &S12, const float th2, const bool bFixScale);
    static void OptimizeEssentialGraph(Map::Ptr pMap, KeyFrame::Ptr pLoopKF, KeyFrame::Ptr pCurKF, KeyFrameAndPose &NonCorrectedSim3, KeyFrameAndPose &CorrectedSim3,
                                             const std::map<KeyFrame::Ptr, std::set<KeyFrame::Ptr> > &LoopConnections, const bool &bFixScale = false);

//    static void localBundleAdjustmentWithInvDepth(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size=10, bool report=false, bool verbose=false);

    static void refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report=false, bool verbose=false);

    template<int nRes>
    static inline Eigen::Matrix<double, nRes, 1> evaluateResidual(const ceres::Problem& problem, ceres::ResidualBlockId id)
    {
        auto cost = problem.GetCostFunctionForResidualBlock(id);
        std::vector<double*> parameterBlocks;
        problem.GetParameterBlocksForResidualBlock(id, &parameterBlocks);
        Eigen::Matrix<double, nRes, 1> residual;
        cost->Evaluate(parameterBlocks.data(), residual.data(), nullptr);
        return residual;
    }

    template<int nRes>
    static inline void reportInfo(const ceres::Problem &problem, const ceres::Solver::Summary summary, bool report = false, bool verbose = false)
    {
        if (!report) return;

        if (!verbose)
        {
            LOG(INFO) << summary.BriefReport();
        }
        else
        {
            LOG(INFO) << summary.FullReport();
            std::vector<ceres::ResidualBlockId> ids;
            problem.GetResidualBlocks(&ids);
            for (size_t i = 0; i < ids.size(); ++i)
            {
                LOG(INFO) << "BlockId: " << std::setw(5) << i << " residual(RMSE): " << evaluateResidual<nRes>(problem, ids[i]).norm();
            }
        }
    }

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
    //! 正常状态下的陀螺仪bias的初始化
    static Vector3d OptimizeInitialGyroBias(const std::vector<SE3d,Eigen::aligned_allocator<SE3d>>& vE_Twc, const std::vector<IMUPreintegrator>& vImuPreInt);
    //! 重定位时对陀螺仪bias的初始化
    static Vector3d OptimizeInitialGyroBias(const std::vector<Frame::Ptr> vFrames);
    //! 全局BA 用于视-惯初始化完成之后和检测到闭环之后
    static void GlobalBundleAdjustmentNavStatePRV(Map::Ptr pMap, const Vector3d& gw, int nIterations, const uint64_t nLoopKF = 0, bool report=false, bool verbose=false);
    //! 滑动窗口的优化
    static void LocalBAPRVIDP(Map::Ptr pMap, const KeyFrame::Ptr keyframe,const std::vector<KeyFrame::Ptr> &actived_keyframes, std::list<MapPoint::Ptr> &bad_mpts, int size,
                              int min_shared_fts,const Vector3d& gw, bool report, bool verbose);
    //! 单帧的位姿优化，用于当前帧的优化
    static int PoseOptimization(Frame::Ptr PFrame, KeyFrame::Ptr pLastKF, const IMUPreintegrator& imupreint, const Vector3d& gw, const bool& bComputeMarg=false);
    static int PoseOptimization(Frame::Ptr pFrame, Frame::Ptr pLastFrame, const IMUPreintegrator& imupreint, const Vector3d& gw, const bool& bComputeMarg=false);


};

namespace ceres_slover {
// https://github.com/strasdat/Sophus/blob/v1.0.0/test/ceres/local_parameterization_se3.hpp
class SE3Parameterization : public ceres::LocalParameterization {
public:
    virtual ~SE3Parameterization() {}

    virtual bool Plus(double const *T_raw, double const *delta_raw,
                      double *T_plus_delta_raw) const {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = Sophus::SE3d::exp(delta) * T;
        return true;
    }

    // Set to Identity, for we have computed in ReprojectionErrorSE3::Evaluate
    virtual bool ComputeJacobian(double const *T_raw,
                                 double *jacobian_raw) const {
        Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
        jacobian.block<6,6>(0, 0).setIdentity();
        jacobian.rightCols<1>().setZero();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const {
        Sophus::SE3<T> pose = Eigen::Map<const Sophus::SE3<T> >(camera);
        Eigen::Matrix<T, 3, 1> p = Eigen::Map<const Eigen::Matrix<T, 3, 1> >(point);

        Eigen::Matrix<T, 3, 1> p1 = pose.rotationMatrix() * p + pose.translation();

        T predicted_x = (T) p1[0] / p1[2];
        T predicted_y = (T) p1[1] / p1[2];
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, Sophus::SE3d::num_parameters, 3>(
            new ReprojectionError(observed_x, observed_y)));
    }

    double observed_x_;
    double observed_y_;
};

class ReprojectionErrorSE3 : public ceres::SizedCostFunction<2, 7, 3>
{
public:

    ReprojectionErrorSE3(double observed_x, double observed_y, double weight)
        : observed_x_(observed_x), observed_y_(observed_y), weight_(weight) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //! In Sophus, stored in the form of [q, t]
        Eigen::Map<const Eigen::Quaterniond> q(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t(parameters[0] + 4);
        Eigen::Map<const Eigen::Vector3d> p(parameters[1]);

        Eigen::Vector3d p1 = q * p + t;

        const double predicted_x =  p1[0] / p1[2];
        const double predicted_y =  p1[1] / p1[2];
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        residuals[0] *= weight_*460;
        residuals[1] *= weight_*460;

//        std::cout<<"ReprojectionErrorSE3 Error: "<< residuals[0] * residuals[0] + residuals[1] * residuals[1] <<std::endl;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian;

        const double z_inv = 1.0 / p1[2];
        const double z_inv2 = z_inv*z_inv;
        jacobian << z_inv, 0.0, -p1[0]*z_inv2,
                    0.0, z_inv, -p1[1]*z_inv2;

        jacobian.array() *= weight_;

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > Jse3(jacobian0);
            Jse3.setZero();
            //! In the order of Sophus::Tangent
            Jse3.block<2,3>(0,0) = jacobian;
            Jse3.block<2,3>(0,3) = jacobian*Sophus::SO3d::hat(-p1);
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobian1);
            Jpoint = jacobian * q.toRotationMatrix();
        }
        return true;
    }

    static inline ceres::CostFunction *Create(const double observed_x, const double observed_y, const double weight = 1.0) {
        return (new ReprojectionErrorSE3(observed_x, observed_y, weight));
    }

private:

    double observed_x_;
    double observed_y_;
    double weight_;

}; // class ReprojectionErrorSE3

class ReprojectionErrorSE3InvDepth : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:

    ReprojectionErrorSE3InvDepth(double observed_x_ref, double observed_y_ref, double observed_x_cur, double observed_y_cur)
        : observed_x_ref_(observed_x_ref), observed_y_ref_(observed_y_ref),
          observed_x_cur_(observed_x_cur), observed_y_cur_(observed_y_cur) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {

        Eigen::Map<const Sophus::SE3d> T_ref(parameters[0]);
        Eigen::Map<const Sophus::SE3d> T_cur(parameters[1]);
        const double inv_z = parameters[2][0];

        const Eigen::Vector3d p_ref(observed_x_ref_/inv_z, observed_y_ref_/inv_z, 1.0/inv_z);
        const Sophus::SE3d T_cur_ref = T_cur * T_ref.inverse();
        const Eigen::Vector3d p_cur = T_cur_ref * p_ref;

        const double predicted_x =  p_cur[0] / p_cur[2];
        const double predicted_y =  p_cur[1] / p_cur[2];
        residuals[0] = predicted_x - observed_x_cur_;
        residuals[1] = predicted_y - observed_y_cur_;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];
        double* jacobian2 = jacobians[2];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> Jproj;

        const double z_inv = 1.0 / p_cur[2];
        const double z_inv2 = z_inv*z_inv;
        Jproj << z_inv, 0.0, -p_cur[0]*z_inv2,
            0.0, z_inv, -p_cur[1]*z_inv2;

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JRse3(jacobian0);
            JRse3.setZero();
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> JRP = Jproj*T_cur_ref.rotationMatrix();
            JRse3.block<2,3>(0,0) = -JRP;
            JRse3.block<2,3>(0,3) = JRP*Sophus::SO3d::hat(p_ref);
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JCse3(jacobian1);
            JCse3.setZero();
            JCse3.block<2,3>(0,0) = Jproj;
            JCse3.block<2,3>(0,3) = Jproj*Sophus::SO3d::hat(-p_cur);
        }
        if(jacobian2 != nullptr)
        {
//            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jp(jacobian2);
//            Eigen::Matrix3d Jpp(T_cur_ref.rotationMatrix());
//            Jpp.col(2) = T_cur_ref.rotationMatrix() * (-p_ref);
//            Jp.noalias() = Jproj * Jpp * p_ref[2];
            Eigen::Map<Eigen::RowVector2d> Jp(jacobian2);
            Jp = Jproj * T_cur_ref.rotationMatrix() * p_ref * (-1.0/inv_z);
        }
        return true;
    }

    static inline ceres::CostFunction *Create(double observed_x_ref, double observed_y_ref,
                                              double observed_x_cur, double observed_y_cur) {
        return (new ReprojectionErrorSE3InvDepth(observed_x_ref, observed_y_ref, observed_x_cur, observed_y_cur));
    }

private:

    double observed_x_ref_;
    double observed_y_ref_;
    double observed_x_cur_;
    double observed_y_cur_;

};

class ReprojectionErrorSE3InvPoint : public ceres::SizedCostFunction<2, 7, 7, 3>
{
public:

    ReprojectionErrorSE3InvPoint(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y){}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Sophus::SE3d> T_ref(parameters[0]);
        Eigen::Map<const Sophus::SE3d> T_cur(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> inv_p(parameters[2]);
        Sophus::SE3d T_cur_ref = T_cur * T_ref.inverse();

        const Eigen::Vector3d p_ref(inv_p[0] / inv_p[2], inv_p[1] / inv_p[2], 1.0 / inv_p[2]);
        const Eigen::Vector3d p_cur = T_cur_ref * p_ref;

        const double predicted_x =  p_cur[0] / p_cur[2];
        const double predicted_y =  p_cur[1] / p_cur[2];
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        if(!jacobians) return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];
        double* jacobian2 = jacobians[2];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> Jproj;

        const double z_inv = 1.0 / p_cur[2];
        const double z_inv2 = z_inv*z_inv;
        Jproj << z_inv, 0.0, -p_cur[0]*z_inv2,
            0.0, z_inv, -p_cur[1]*z_inv2;

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JRse3(jacobian0);
            JRse3.setZero();
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> JRP = Jproj*T_cur_ref.rotationMatrix();
            JRse3.block<2,3>(0,0) = -JRP;
            JRse3.block<2,3>(0,3) = JRP*Sophus::SO3d::hat(p_ref);
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > JCse3(jacobian1);
            JCse3.setZero();
            JCse3.block<2,3>(0,0) = Jproj;
            JCse3.block<2,3>(0,3) = Jproj*Sophus::SO3d::hat(-p_cur);
        }
        if(jacobian2 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jp(jacobian2);
            Eigen::Matrix3d Jpp(T_cur_ref.rotationMatrix());
            Jpp.col(2) = T_cur_ref.rotationMatrix() * (-p_ref);
            Jp.noalias() = Jproj * Jpp * p_ref[2];
        }
        return true;
    }

    static inline ceres::CostFunction *Create(double observed_x, double observed_y) {
        return (new ReprojectionErrorSE3InvPoint(observed_x, observed_y));
    }

private:

    double observed_x_;
    double observed_y_;

};

//!**********************************************************************************************************************************************

class PVRParameterization : public ceres::LocalParameterization
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~PVRParameterization(){}

    virtual bool Plus(double const* T_raw, double const *delta_raw, double *T_plus_delta_raw)
    {
        Eigen::Map<const Eigen::Matrix<double,6,1>,Eigen::RowMajor> pv_raw(T_raw);
        Eigen::Map<const Eigen::Matrix<double,3,1>,Eigen::RowMajor> phi_raw(T_raw+6);
        Eigen::Map<const Eigen::Matrix<double,6,1>,Eigen::RowMajor> pv_delta(delta_raw);
        Eigen::Map<const Eigen::Matrix<double,3,1>,Eigen::RowMajor> phi_delta(delta_raw+6);
        Eigen::Map< Eigen::Matrix<double,6,1>,Eigen::RowMajor> pv_plus(T_plus_delta_raw);
        Eigen::Map< Eigen::Matrix<double,3,1>,Eigen::RowMajor> phi_plus(T_plus_delta_raw+6);

        pv_plus = pv_raw + pv_delta;
        //右乘扰动，因此phi的result并不是直接相加的形式
        phi_plus = (Sophus::SO3d::exp(phi_raw) * Sophus::SO3d::exp(phi_delta)).log();
        return true;
    }
    //在ceres中解析解求出的是是针对delta 量的导数，所以这里直接设置成单位矩阵就可以了
    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 9, 9> > jacobian(jacobian_raw);
        jacobian.setIdentity();
        return true;
    }
    virtual int GlobalSize() const { return 9; }
    virtual int LocalSize() const { return 9; }

};

//bias，没用
/*
//! eigen map 当是X*1大小的矩阵的时候不能加行优先的选项。。。
//class BiasParameterization : public ceres::LocalParameterization
//{
//public:
////    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//
//    virtual ~BiasParameterization(){}
//    virtual bool Plus(double const* T_raw, double const *delta_raw, double *T_plus_delta_raw) const
//    {
//        Sophus::Vector6d bias_raw;
//        bias_raw << T_raw[0],T_raw[1],T_raw[2],T_raw[3],T_raw[4],T_raw[5];
////        Eigen::Map<const Sophus::Vector6d ,Eigen::RowMajor> bias_raw(T_raw);
//        Sophus::Vector6d bias_delta;
//        bias_delta<<delta_raw[0],delta_raw[1],delta_raw[2],delta_raw[3],delta_raw[4],delta_raw[5];
////        Eigen::Map<const Sophus::Vector6d,Eigen::RowMajor> bias_delta(delta_raw);
//        Sophus::Vector6d bias_plus;
//        bias_plus<<T_plus_delta_raw[0],T_plus_delta_raw[1],T_plus_delta_raw[2],T_plus_delta_raw[3],T_plus_delta_raw[4],T_plus_delta_raw[5];
////        Eigen::Map< Sophus::Vector6d,Eigen::RowMajor> bias_plus(T_plus_delta_raw);
//
//        bias_plus = bias_raw + bias_delta;
//
//        return true;
//    }
//    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
//    {
//        Eigen::Map<Eigen::Matrix<double, 6, 6>> jacobian(jacobian_raw);
//        jacobian.setIdentity();
//        return true;
//    }
//    virtual int GlobalSize() const { return 6; }
//    virtual int LocalSize() const { return 6; }
//};
 */

class PRParameterization : public ceres::LocalParameterization
{
public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~PRParameterization(){}

    virtual bool Plus(double const* T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<const Eigen::Vector3d> p_raw(T_raw);
        Eigen::Map<const Eigen::Vector3d> phi_raw(T_raw+3);
//        std::cout<<"p_raw: "<<p_raw.transpose()<<std::endl;

        Eigen::Map<const Eigen::Vector3d> p_delta(delta_raw);
        Eigen::Map<const Eigen::Vector3d> phi_delta(delta_raw+3);
//        std::cout<<"p_delta: "<<p_delta.transpose()<<std::endl;

        Eigen::Map< Eigen::Vector3d> p_plus(T_plus_delta_raw);
        Eigen::Map< Eigen::Vector3d> phi_plus(T_plus_delta_raw+3);

        p_plus = p_raw + p_delta;

//        std::cout<<"p_plus: "<<p_plus.transpose()<<std::endl;
        phi_plus = (Sophus::SO3d::exp(phi_raw) * Sophus::SO3d::exp(phi_delta)).log();

        return true;
    }
    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> jacobian(jacobian_raw);
        jacobian.setIdentity();
        return true;
    }
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }

};

class PRVParameterization : public ceres::LocalParameterization
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual ~PRVParameterization(){}

    virtual bool Plus(double const* T_raw, double const *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<const Eigen::Matrix<double,3,1>> p_raw(T_raw);
        Eigen::Map<const Eigen::Matrix<double,3,1>> v_raw(T_raw+6);
        Eigen::Map<const Eigen::Matrix<double,3,1>> phi_raw(T_raw+3);

        Eigen::Map<const Eigen::Matrix<double,3,1>> p_delta(delta_raw);
        Eigen::Map<const Eigen::Matrix<double,3,1>> v_delta(delta_raw+6);
        Eigen::Map<const Eigen::Matrix<double,3,1>> phi_delta(delta_raw+3);

        Eigen::Map< Eigen::Matrix<double,3,1>> p_plus(T_plus_delta_raw);
        Eigen::Map< Eigen::Matrix<double,3,1>> v_plus(T_plus_delta_raw+6);
        Eigen::Map< Eigen::Matrix<double,3,1>> phi_plus(T_plus_delta_raw+3);

        p_plus = p_raw + p_delta;
        v_plus = v_raw + v_delta;
        //右乘扰动
        phi_plus = (Sophus::SO3d::exp(phi_raw) * Sophus::SO3d::exp(phi_delta)).log();
        return true;
    }
    virtual bool ComputeJacobian(double const *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 9, 9>> jacobian(jacobian_raw);
        jacobian.setIdentity();
        return true;
    }
    virtual int GlobalSize() const { return 9; }
    virtual int LocalSize() const { return 9; }
};



class NavStatePVRError : public ceres::SizedCostFunction<9,9,6,9>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NavStatePVRError(const IMUPreintegrator& M, const Vector3d& GravityVec):
            M_(M),GravityVec_(GravityVec)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Vector3d> Pi(parameters[0]);
        Eigen::Map<const Vector3d> Vi(parameters[0]+3);
        Eigen::Map<const Vector3d> PHIi(parameters[0]+6);
        Sophus::SO3d Ri = Sophus::SO3d::exp(PHIi);

        Eigen::Map<const Vector3d> dBgi(parameters[1]);
        Eigen::Map<const Vector3d> dBai(parameters[0]+3);

        Eigen::Map<const Vector3d> Pj(parameters[2]);
        Eigen::Map<const Vector3d> Vj(parameters[2]+3);
        Eigen::Map<const Vector3d> PHIj(parameters[2]+6);
        Sophus::SO3d Rj = Sophus::SO3d::exp(PHIj);


        //! 测量值
        const double dTij = M_.getDeltaTime();   // Delta Time
        const double dT2 = dTij*dTij;
        const Vector3d dPij = M_.getDeltaP();    // Delta Position pre-integration measurement
        const Vector3d dVij = M_.getDeltaV();    // Delta Velocity pre-integration measurement
        const Sophus::SO3d dRij = Sophus::SO3d(M_.getDeltaR());  // Delta Rotation pre-integration measurement

        // tmp variable, transpose of Ri
        const Sophus::SO3d RiT = Ri.inverse();
        // residual error of Delta Position measurement
        const Vector3d rPij = RiT*(Pj - Pi - Vi*dTij - 0.5*GravityVec_*dT2)
                              - (dPij + M_.getJPBiasg()*dBgi + M_.getJPBiasa()*dBai);   // this line includes correction term of bias change.
        // residual error of Delta Velocity measurement
        const Vector3d rVij = RiT*(Vj - Vi - GravityVec_*dTij)
                              - (dVij + M_.getJVBiasg()*dBgi + M_.getJVBiasa()*dBai);   //this line includes correction term of bias change
        // residual error of Delta Rotation measurement
        const Sophus::SO3d dR_dbg = Sophus::SO3d::exp(M_.getJRBiasg()*dBgi);
        const Sophus::SO3d rRij = (dRij * dR_dbg).inverse() * RiT * Rj;
        const Vector3d rPhiij = rRij.log();

        Eigen::Map<Eigen::Matrix<double,9,1>> residual(residuals);

        residual.block(0,0,3,1) = rPij;
        residual.block(3,0,3,1) = rPhiij;
        residual.block(6,0,3,1) = rVij;


        if(jacobians)
        {
            Matrix3d O3x3 = Matrix3d::Zero();       // 0_3x3
            Matrix3d RiT = Ri.matrix().transpose();          // Ri^T
            Matrix3d RjT = Rj.matrix().transpose();          // Rj^T
            Matrix3d JrInv_rPhi = Add_math::JacobianRInv(rPhiij);    // inverse right jacobian of so3 term #rPhiij#
            Matrix3d J_rPhi_dbg = M_.getJRBiasg();              // jacobian of preintegrated rotation-angle to gyro bias i
            Matrix3d ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
            Matrix3d JrBiasGCorr = Add_math::JacobianR(J_rPhi_dbg*dBgi);

            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double ,9,9, Eigen::RowMajor>> jacobian_r_posei(jacobians[0]);
                jacobian_r_posei.setZero();

                jacobian_r_posei.block<3,3>(0,0) = - RiT;      //J_rP_dpi
                jacobian_r_posei.block<3,3>(0,3) = - RiT*dTij;
                jacobian_r_posei.block<3,3>(0,6) = Sophus::SO3d::hat( RiT*(Pj-Pi-Vi*dTij-0.5*GravityVec_*dT2)  );    //J_rP_dPhi_i
                // J_rPhiij_xxx_i for Vertex_PR_i
                Matrix3d ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
                Matrix3d JrBiasGCorr = Add_math::JacobianR(J_rPhi_dbg*dBgi);


                jacobian_r_posei.block<3,3>(3,0) = O3x3;    //dpi
                jacobian_r_posei.block<3,3>(3,3) = - RiT;
                jacobian_r_posei.block<3,3>(3,6) = Sophus::SO3d::hat(RiT*(Vj-Vi-GravityVec_*dTij));
                // J_rVij_xxx_i for Vertex_PVR_i
                jacobian_r_posei.block<3,3>(6,0) = O3x3;    //dpi
                jacobian_r_posei.block<3,3>(6,3) = O3x3;
                jacobian_r_posei.block<3,3>(6,6) = - JrInv_rPhi * RjT * Ri.matrix();

            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double ,9,6, Eigen::RowMajor>> jacobian_r_biasi(jacobians[1]);

                jacobian_r_biasi.setZero();
                jacobian_r_biasi.block<3,3>(0,0) = - M_.getJPBiasg();     //J_rP_dbgi
                jacobian_r_biasi.block<3,3>(0,3) = - M_.getJPBiasa();     //J_rP_dbai

                // J_rVij_xxx_j for Vertex_Bias_i
                jacobian_r_biasi.block<3,3>(3,0) = - M_.getJVBiasg();    //dbg_i
                jacobian_r_biasi.block<3,3>(3,3) = - M_.getJVBiasa();    //dba_i

                // J_rPhiij_xxx_j for Vertex_Bias_i
                jacobian_r_biasi.block<3,3>(6,0) = - JrInv_rPhi * ExprPhiijTrans * JrBiasGCorr * J_rPhi_dbg;    //dbg_i
                jacobian_r_biasi.block<3,3>(6,3) = O3x3;    //dba_i


            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double ,9,9, Eigen::RowMajor>> jacobian_r_posej(jacobians[2]);

                jacobian_r_posej.setZero();
                // J_rPij_xxx_j for Vertex_PR_j
                jacobian_r_posej.block<3,3>(0,0) = RiT;  //rP_dpj
                jacobian_r_posej.block<3,3>(0,3) = O3x3;
                jacobian_r_posej.block<3,3>(0,6) = O3x3;
                // J_rPhiij_xxx_j for Vertex_PR_j
                // J_rVij_xxx_j for Vertex_PR_j
                jacobian_r_posej.block<3,3>(3,0) = O3x3;    //rV_dpj
                jacobian_r_posej.block<3,3>(3,3) = RiT;
                jacobian_r_posej.block<3,3>(3,6) = O3x3;

                jacobian_r_posej.block<3,3>(6,0) = O3x3;    //rR_dpj
                jacobian_r_posej.block<3,3>(6,3) = O3x3;
                jacobian_r_posej.block<3,3>(6,6) = JrInv_rPhi;    //rR_dphi_j

            }
        }

        return true;
    }

    static inline ceres::CostFunction* Creat(const IMUPreintegrator& M, const Vector3d& GravityVec)
    {
        return (new NavStatePVRError(M,GravityVec));
    }
private:
    IMUPreintegrator M_;
    Vector3d GravityVec_;

};
class GyrBiasError : public ceres::SizedCostFunction<3,3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GyrBiasError(const Matrix3d& dRbij, const Matrix3d& J_dR_bg, const Matrix3d& Rwbi,const Matrix3d& Rwbj):
            dRbij_(dRbij),J_dR_bg_(J_dR_bg),Rwbi_(Rwbi),Rwbj_(Rwbj)
    {}

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);
        Eigen::Map<Eigen::Vector3d> residual(residuals);

        Matrix3d dRbg = Sophus::SO3d::exp(J_dR_bg_ * bg).matrix();
        Sophus::SO3d errR((dRbij_ * dRbg).transpose() * Rwbi_.transpose() * Rwbj_) ;

        residual = errR.log();

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix3d> jacobian_bg(jacobians[0]);
                jacobian_bg.setZero();

                Sophus::SO3d errR ( dRbij_.transpose() * Rwbi_.transpose() * Rwbj_ ); // dRij^T * Riw * Rwj
                Matrix3d Jlinv = Add_math::JacobianRInv(-errR.log());
                jacobian_bg = - Jlinv * J_dR_bg_;
            }
        }

        return true;
    }

    static inline ceres::CostFunction* Creat(const Matrix3d& dRbij, const Matrix3d& J_dR_bg, const Matrix3d& Rwbi,const Matrix3d& Rwbj)
    {
        return (new GyrBiasError(dRbij,J_dR_bg,Rwbi,Rwbj));
    }

private:
    Matrix3d dRbij_;
    Matrix3d J_dR_bg_;
    Matrix3d Rwbi_;
    Matrix3d Rwbj_;

};

class NavStatePRVError : public ceres::SizedCostFunction<9,6,3,6,6,3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    NavStatePRVError(const Matrix9d& CovPRV, const IMUPreintegrator& M, const Vector3d& GravityVec) :
            CovPRV_(CovPRV),M_(M),GravityVec_(GravityVec)
    {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        //第i帧的状态

        Eigen::Map<const Eigen::Vector3d> Pi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Phii(parameters[0]+3);
        Sophus::SO3d Ri = Sophus::SO3d::exp(Phii);

        Eigen::Map<const Eigen::Vector3d> Vi(parameters[1]);

        Eigen::Map<const Eigen::Vector3d> dBgi(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> dBai(parameters[2]+3);
        //第j帧的状态
        Eigen::Map<const Eigen::Vector3d> Pj(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> Phij(parameters[3]+3);
        Sophus::SO3d Rj = Sophus::SO3d::exp(Phij);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[4]);

        //! 测量值
        const double dTij = M_.getDeltaTime();   // Delta Time
//        std::cout<<"dTij:"<<std::endl<<dTij<<std::endl;
        const double dT2 = dTij*dTij;
//        std::cout<<"dT2:"<<std::endl<<dT2<<std::endl;
        const Vector3d dPij = M_.getDeltaP();    // Delta Position pre-integration measurement
        const Vector3d dVij = M_.getDeltaV();    // Delta Velocity pre-integration measurement
        const Sophus::SO3d dRij = Sophus::SO3d(M_.getDeltaR());  // Delta Rotation pre-integration measurement

//        std::cout<<"dPij:"<<std::endl<<dPij<<std::endl;
//        std::cout<<"dVij:"<<std::endl<<dVij<<std::endl;
////        std::cout<<"rPhiij:"<<std::endl<<rPhiij<<std::endl;
//
//        std::cout<<"GravityVec_:"<<std::endl<<GravityVec_<<std::endl;
//        std::cout<<"Vi:"<<std::endl<<Vi<<std::endl;
//        std::cout<<"Vj:"<<std::endl<<Vj<<std::endl;


        // tmp variable, transpose of Ri
        const Sophus::SO3d RiT = Ri.inverse();
        // residual error of Delta Position measurement
        const Vector3d rPij = RiT*(Pj - Pi - Vi*dTij - 0.5*GravityVec_*dT2)
                              - (dPij + M_.getJPBiasg()*dBgi + M_.getJPBiasa()*dBai);   // this line includes correction term of bias change.
        // residual error of Delta Velocity measurement
        const Vector3d rVij = RiT*(Vj - Vi - GravityVec_*dTij)
                              - (dVij + M_.getJVBiasg()*dBgi + M_.getJVBiasa()*dBai);   //this line includes correction term of bias change
        // residual error of Delta Rotation measurement
        const Sophus::SO3d dR_dbg = Sophus::SO3d::exp(M_.getJRBiasg()*dBgi);
        const Sophus::SO3d rRij = (dRij * dR_dbg).inverse() * RiT * Rj;
        const Vector3d rPhiij = rRij.log();

//        std::cout<<"rPij:"<<std::endl<<rPij<<std::endl;
//        std::cout<<"rVij:"<<std::endl<<rVij<<std::endl;
//        std::cout<<"rPhiij:"<<std::endl<<rPhiij<<std::endl;

        Eigen::Map<Eigen::Matrix<double,9,1>> residual(residuals);

        residual.block(0,0,3,1) = rPij;
        residual.block(3,0,3,1) = rPhiij;
        residual.block(6,0,3,1) = rVij;

        Eigen::Matrix<double, 9, 9> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 9, 9>>(CovPRV_.inverse()).matrixL().transpose();//桥列司机分解

//        std::cout<<"NavStatePRVError before： "<<residual.transpose()* residual<<std::endl;

//        std::cout<<"NavStatePRVError CovPRV_.inverse()： "<<std::endl<<CovPRV_.inverse()<<std::endl;

        residual = sqrt_info * residual;

//        std::cout<<"NavStatePRVError after： "<<residual.transpose() * residual<<std::endl;

        if(jacobians)
        {
            Matrix3d O3x3 = Matrix3d::Zero();       // 0_3x3
            Matrix3d RiT = Ri.matrix().transpose();          // Ri^T
            Matrix3d RjT = Rj.matrix().transpose();          // Rj^T
            Matrix3d JrInv_rPhi = Add_math::JacobianRInv(rPhiij);    // inverse right jacobian of so3 term #rPhiij#
            Matrix3d J_rPhi_dbg = M_.getJRBiasg();              // jacobian of preintegrated rotation-angle to gyro bias i
            Matrix3d ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
            Matrix3d JrBiasGCorr = Add_math::JacobianR(J_rPhi_dbg*dBgi);

            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double , 9, 6, Eigen::RowMajor>> jacobian_r_posei(jacobians[0]);
                jacobian_r_posei.setZero();

//                std::cout<<"jacobian_r_posei: "<<jacobian_r_posei<<std::endl;

                jacobian_r_posei.block<3,3>(0,0) = - RiT;      //J_rP_dpi
                jacobian_r_posei.block<3,3>(0,3) = Sophus::SO3d::hat( RiT*(Pj-Pi-Vi*dTij-0.5*GravityVec_*dT2)  );    //J_rP_dPhi_i
                // J_rPhiij_xxx_i for Vertex_PR_i
                Matrix3d ExprPhiijTrans = Sophus::SO3d::exp(rPhiij).inverse().matrix();
                Matrix3d JrBiasGCorr = Add_math::JacobianR(J_rPhi_dbg*dBgi);
                jacobian_r_posei.block<3,3>(3,0) = O3x3;    //dpi
                jacobian_r_posei.block<3,3>(3,3) = - JrInv_rPhi * RjT * Ri.matrix();    //dphi_i
                // J_rVij_xxx_i for Vertex_PVR_i
                jacobian_r_posei.block<3,3>(6,0) = O3x3;    //dpi
                jacobian_r_posei.block<3,3>(6,3) = Sophus::SO3d::hat( RiT*(Vj-Vi-GravityVec_*dTij) );    //dphi_i

                jacobian_r_posei = sqrt_info * jacobian_r_posei;

            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double ,9,3, Eigen::RowMajor>> jacobian_r_vi(jacobians[1]);
                jacobian_r_vi.setZero();

                jacobian_r_vi.block<3,3>(0,0) = - RiT*dTij;  //J_rP_dvi
                jacobian_r_vi.block<3,3>(3,0) = O3x3;    //rR_dvi
                jacobian_r_vi.block<3,3>(6,0) = - RiT;    //rV_dvi

                jacobian_r_vi = sqrt_info * jacobian_r_vi;

            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double ,9,6, Eigen::RowMajor>> jacobian_r_biasi(jacobians[2]);
                jacobian_r_biasi.setZero();

                jacobian_r_biasi.block<3,3>(0,0) = - M_.getJPBiasg();     //J_rP_dbgi
                jacobian_r_biasi.block<3,3>(0,3) = - M_.getJPBiasa();     //J_rP_dbai

                // J_rPhiij_xxx_j for Vertex_Bias_i
                jacobian_r_biasi.block<3,3>(3,0) = - JrInv_rPhi * ExprPhiijTrans * JrBiasGCorr * J_rPhi_dbg;    //dbg_i
                jacobian_r_biasi.block<3,3>(3,3) = O3x3;    //dba_i

                // J_rVij_xxx_j for Vertex_Bias_i
                jacobian_r_biasi.block<3,3>(6,0) = - M_.getJVBiasg();    //dbg_i
                jacobian_r_biasi.block<3,3>(6,3) = - M_.getJVBiasa();    //dba_i

                jacobian_r_biasi = sqrt_info * jacobian_r_biasi;
            }
            if(jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double ,9,6, Eigen::RowMajor>> jacobian_r_posej(jacobians[3]);
                jacobian_r_posej.setZero();

                // J_rPij_xxx_j for Vertex_PR_j
                jacobian_r_posej.block<3,3>(0,0) = RiT;  //rP_dpj
                jacobian_r_posej.block<3,3>(0,3) = O3x3;    //rP_dphi_j
                // J_rPhiij_xxx_j for Vertex_PR_j
                jacobian_r_posej.block<3,3>(3,0) = O3x3;    //rR_dpj
                jacobian_r_posej.block<3,3>(3,3) = JrInv_rPhi;    //rR_dphi_j
                // J_rVij_xxx_j for Vertex_PR_j
                jacobian_r_posej.block<3,3>(6,0) = O3x3;    //rV_dpj
                jacobian_r_posej.block<3,3>(6,3) = O3x3;    //rV_dphi_j

                jacobian_r_posej = sqrt_info * jacobian_r_posej;

            }
            if(jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double ,9,3, Eigen::RowMajor>> jacobian_r_vj(jacobians[4]);
                // For Vertex_V_i, J [dP;dR;dV] / dV1
                jacobian_r_vj.setZero();

                jacobian_r_vj.block<3,3>(0,0) = O3x3;    //rP_dvj
                jacobian_r_vj.block<3,3>(3,0) = O3x3;    //rR_dvj
                jacobian_r_vj.block<3,3>(6,0) = RiT;    //rV_dvj

                jacobian_r_vj = sqrt_info * jacobian_r_vj;

            }
        }
        return true;
    }

    static inline ceres::CostFunction *Create(const Matrix9d& CovPRV, const IMUPreintegrator& M, const Vector3d& GravityVec) {
        return (new NavStatePRVError(CovPRV,M,GravityVec));
    }

private:
    Matrix<double,9,9> CovPRV_;
    IMUPreintegrator M_;
    Vector3d GravityVec_;
};

//! 待优化变量：delta bias
class NavStateBiasError: public ceres::SizedCostFunction<6,6,6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    NavStateBiasError(const Matrix<double,6,6>& Covariance,const Vector3d& bgi,const Vector3d& bai,const Vector3d& bgj,const Vector3d& baj):
            Covariance_(Covariance),bgi_(bgi),bai_(bai),bgj_(bgj),baj_(baj)
    {}

    virtual bool Evaluate(double const* const* parameter, double* residuals, double** jacobians) const
    {

        Eigen::Map<const Vector3d> vdBgi(parameter[0]);
        Eigen::Map<const Vector3d> vdBai(parameter[0]+3);
        Eigen::Map<const Vector3d> vdBgj(parameter[1]);
        Eigen::Map<const Vector3d> vdBaj(parameter[1]+3);
        Eigen::Map<Eigen::Matrix<double,6,1>> resisual(residuals);

        resisual.setZero();
        Vector3d rBiasG = (bgj_ + vdBgj) - (bgi_ + vdBgi);
        Vector3d rBiasA = (baj_ + vdBaj) - (bai_ + vdBai);
        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(Covariance_.inverse()).matrixL().transpose();//桥列司机分解
        resisual.block(0,0,3,1) = rBiasG;
        resisual.block(3,0,3,1) = rBiasA;
        resisual = sqrt_info * resisual;

//        std::cout<<"NavStateBiasError: "<< resisual.transpose()*resisual<<std::endl;

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double,6,6, Eigen::RowMajor>> jacobian_r_dbiasi(jacobians[0]);
                jacobian_r_dbiasi = - Matrix<double,6,6>::Identity();
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double , 6,6, Eigen::RowMajor> > jacobian_r_dbiasj(jacobians[1]);
                jacobian_r_dbiasj = Matrix<double,6,6>::Identity();
            }
        }
        return true;
    }

    static inline ceres::CostFunction* Creat(const Matrix<double,6,6>& Covariance,const Vector3d& bgi,const Vector3d& bai,const Vector3d& bgj,const Vector3d& baj)
    {
        return (new NavStateBiasError(Covariance,bgi,bai,bgj,baj));
    }

private:
    Matrix<double,6,6> Covariance_;
    Vector3d bgi_;
    Vector3d bai_;
    Vector3d bgj_;
    Vector3d baj_;

};

class PointReprojectionError: public ceres::SizedCostFunction<2,6,3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointReprojectionError(const SE3d& Tbc,const double observed_x, const double observed_y, const double weight = 1.0):
            Tbc_(Tbc),observed_x_(observed_x),observed_y_(observed_y),weight_(weight)
    {
    }

    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {

        Eigen::Map<const Eigen::Vector3d> Pwb(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Phiwb(parameters[0]+3);
        Matrix3d Rwb = Sophus::SO3d::exp(Phiwb).matrix();
        SE3d Twb(Rwb,Pwb);
        SE3d Tcw = (Twb * Tbc_).inverse();

        Eigen::Map<const Eigen::Vector3d> Pw(parameters[1]);

        Eigen::Vector3d p1 = Tcw * Pw;

        const double predicted_x =  p1[0] / p1[2];
        const double predicted_y =  p1[1] / p1[2];
        residuals[0] = predicted_x - observed_x_;
        residuals[1] = predicted_y - observed_y_;

        residuals[0] *= weight_;
        residuals[1] *= weight_;

        double info = 460;

        residuals[0] *= info;
        residuals[1] *= info;

//        std::cout<<"PointReprojectionError： "<<0.5*(residuals[0]*residuals[0]+residuals[1]*residuals[1])<<std::endl;

        if(!jacobians)
            return true;
        double* jacobian0 = jacobians[0];
        double* jacobian1 = jacobians[1];

        //! The point observed is in the normalized plane
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian;

        const double z_inv = 1.0 / p1[2];
        const double z_inv2 = z_inv*z_inv;
        jacobian << z_inv, 0.0, -p1[0]*z_inv2,
                0.0, z_inv, -p1[1]*z_inv2;

        jacobian.array() *= weight_;
        jacobian.array() *= info;

        Matrix3d Rcb = Tbc_.rotationMatrix().transpose();
        Vector3d Paux = Rcb * Rwb.transpose()*(Pw - Pwb);

        if(jacobian0 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor> > Jse3(jacobian0);
            Jse3.setZero();
            //! In the order of Sophus::Tangent
            Jse3.block<2,3>(0,0) =  jacobian * (-Rcb * Rwb.transpose());
            Jse3.block<2,3>(0,3) =  jacobian * (Sophus::SO3d::hat(Paux) * Rcb);

//            std::cout<<"Jse3 :"<<std::endl<<Jse3<<std::endl;
        }
        if(jacobian1 != nullptr)
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > Jpoint(jacobian1);
            Jpoint.setZero();
            Jpoint =  jacobian * Rcb * Rwb.transpose();
//            std::cout<<"Jpoint :"<<std::endl<<Jpoint<<std::endl;
        }
        return true;
    }

    static inline ceres::CostFunction* Creat(const SE3d& Tbc,const double observed_x, const double observed_y, const double weight = 1.0)
    {
        return (new PointReprojectionError(Tbc,observed_x,observed_y,weight));
    }

private:
    SE3d Tbc_;
    double observed_x_;
    double observed_y_;
    double weight_;
};

class AlignmentError : public ceres::SizedCostFunction<3,4>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AlignmentError(const Matrix<double,3,4>& A, const Vector3d B):A_(A),B_(B)
    {}
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
    {
        Eigen::Map<const Vector4d> x(parameters[0]);

        Eigen::Map<Vector3d> residual(residuals);

        residual = A_ * x - B_;

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Matrix<double,3,4,Eigen::RowMajor>> jacobian(jacobians[0]);
                jacobian = A_;
            }
        }

        return true;
    }

    static inline ceres::CostFunction* Creat(const Matrix<double,3,4>& A, const Vector3d B)
    {
        return (new AlignmentError(A,B));
    }

private:
    Matrix<double,3,4> A_;
    Vector3d B_;
};

//!**********************************************************************************************************************************************

struct IntrinsicReprojErrorOnlyPose
{
    IntrinsicReprojErrorOnlyPose(double observed_x, double observed_y, double* pos,int level,KeyFrame::Ptr pKF)
    {
        obs_x_ = observed_x;
        obs_y_ = observed_y;
        Mp_Pos_ << pos[0], pos[1], pos[2];
        level_ = level;
        pKF_ = pKF;
    }

    bool operator()(const double* const camera, double* residuals) const
    {
        Eigen::Quaterniond q = Eigen::Quaterniond(camera[3],camera[0],camera[1],camera[2]);
        Vector3d t = Vector3d(camera[4],camera[5],camera[6]);

        //todo 检查这里的问题
        if(q.squaredNorm()<0.001||q.squaredNorm()>1000)
            q.normalize();
//        std::cout<<"q.squaredNorm(): "<<q.squaredNorm()<<std::endl;

        Sophus::Sim3d Sim3_cam(q,t);
        Vector3d Mp_cam = Sim3_cam * Mp_Pos_;
        Vector2d px = pKF_->cam_->project(Mp_cam);

        residuals[0] = (obs_x_ - px[0])/sqrt(1<<level_);
        residuals[1] = (obs_y_ - px[1])/sqrt(1<<level_);
        return true;
    }

    double obs_x_, obs_y_;
    Vector3d Mp_Pos_;
    int level_;
    KeyFrame::Ptr pKF_;
};

struct IntrinsicReprojErrorOnlyPoseInvSim3
{
    IntrinsicReprojErrorOnlyPoseInvSim3(double observed_x, double observed_y, double* pos, int level,KeyFrame::Ptr pKF)
    {
        obs_x_ = observed_x;
        obs_y_ = observed_y;
        Mp_Pos_<< pos[0], pos[1], pos[2];
        level_ = level;
        pKF_ =pKF;
    }

    bool operator()(const double* const camera, double* residuals) const
    {

        Eigen::Quaterniond q = Eigen::Quaterniond(camera[3],camera[0],camera[1],camera[2]);
        Vector3d t = Vector3d(camera[4],camera[5],camera[6]);

        //todo 检查这里的问题
        if(q.squaredNorm()<0.001||q.squaredNorm()>1000)
            q.normalize();
//        std::cout<<"q.squaredNorm(): "<<q.squaredNorm()<<std::endl;

        Sophus::Sim3d Sim3_k12(q,t);

        Vector3d Mp_cam = Mp_Pos_;
        Mp_cam = Sim3_k12.inverse() * Mp_cam;

        Vector2d px = pKF_->cam_->project(Mp_cam);

        //! add weigh
        residuals[0] = (px[0] - obs_x_)/sqrt(1<<level_);
        residuals[1] = (px[1] - obs_y_)/sqrt(1<<level_);

        return true;
    }

    double obs_x_, obs_y_;
    Vector3d Mp_Pos_;
    int level_;
    KeyFrame::Ptr pKF_;
};

struct ReprojErrorOnlyPose
{
    ReprojErrorOnlyPose(double observed_x, double observed_y, double* pos, int level,KeyFrame::Ptr pKF):
            intrinsicReprojErrorOnlyPose_(new ceres::NumericDiffCostFunction<IntrinsicReprojErrorOnlyPose,ceres::CENTRAL,2,7>(
            new IntrinsicReprojErrorOnlyPose(observed_x,observed_y,pos,level,pKF)))
    {}

    template <typename T> bool operator()(const T* const camera, T* residuals) const
    {
        return intrinsicReprojErrorOnlyPose_(camera,residuals);
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y, double* pos, int level,KeyFrame::Ptr pKF) {
        return (new ceres::AutoDiffCostFunction<ReprojErrorOnlyPose, 2, 7>(
                new ReprojErrorOnlyPose(observed_x, observed_y, pos, level,pKF)));
    }

private:
    ceres::CostFunctionToFunctor<2,7> intrinsicReprojErrorOnlyPose_;
};

struct ReprojErrorOnlyPoseInvSim3
{
    ReprojErrorOnlyPoseInvSim3(double observed_x, double observed_y, double* pos, int level,KeyFrame::Ptr pKF):
            intrinsicReprojErrorOnlyPoseInvSim3_(new ceres::NumericDiffCostFunction<IntrinsicReprojErrorOnlyPoseInvSim3,ceres::CENTRAL,2,7>(
            new IntrinsicReprojErrorOnlyPoseInvSim3(observed_x, observed_y, pos, level,pKF)))
    {}

    template <typename T> bool operator()(const T* const camera, T* residuals) const
    {
        return intrinsicReprojErrorOnlyPoseInvSim3_(camera,residuals);
    }

    static ceres::CostFunction* Create(double observed_x, double observed_y, double* pos, int level,KeyFrame::Ptr pKF) {
        return (new ceres::AutoDiffCostFunction<ReprojErrorOnlyPoseInvSim3, 2, 7>(
                new ReprojErrorOnlyPoseInvSim3(observed_x, observed_y, pos, level,pKF)));
    }

private:
    ceres::CostFunctionToFunctor<2,7> intrinsicReprojErrorOnlyPoseInvSim3_;
};

struct IntrinsicRelativeSim3Error
{
    IntrinsicRelativeSim3Error(double* obs)
    {
        for (int i = 0; i < 7; ++i)
        {
            mObs[i] = obs[i];
        }
    }

    bool operator()(const double* const camera1, const double* const camera2, double* residual) const
    {
        double camera21[7];
        for (int i = 0; i < 7; ++i)
            camera21[i] = mObs[i];

        Eigen::Quaterniond qk1 = Eigen::Quaterniond(camera1[3],camera1[0],camera1[1],camera1[2]);
        Vector3d tk1 = Vector3d(camera1[4],camera1[5],camera1[6]);

        Sophus::Sim3d Sim3_k1(qk1,tk1);

        Eigen::Quaterniond qk2 = Eigen::Quaterniond(camera2[3],camera2[0],camera2[1],camera2[2]);
        Vector3d tk2 = Vector3d(camera2[4],camera2[5],camera2[6]);

        Sophus::Sim3d Sim3_k2(qk2,tk2);

        Eigen::Quaterniond qk21 = Eigen::Quaterniond(camera21[3],camera21[0],camera21[1],camera21[2]);
        Vector3d tk21 = Vector3d(camera21[4],camera21[5],camera21[6]);

        Sophus::Sim3d Sim3_k21(qk21,tk21);

        Sophus::Sim3d result = Sim3_k21*Sim3_k1*(Sim3_k2.inverse());

        //! S21*S1w*Sw2
        double* tResiduals = result.log().data();

        for (int j = 0; j < 7; ++j)
            residual[j] = tResiduals[j];
        return true;
    }

    double mObs[7];

};

struct RelativeSim3Error
{
    RelativeSim3Error(double* obs):
            intrinsicRelativeSim3Error_(new ceres::NumericDiffCostFunction<IntrinsicRelativeSim3Error,ceres::CENTRAL,7,7,7>(
                    new IntrinsicRelativeSim3Error(obs)))
    {}

    template <typename T> bool operator()(const T* const camera1, const T* const camera2, T* residuals) const
    {
        return intrinsicRelativeSim3Error_(camera1,camera2,residuals);
    }

    static ceres::CostFunction* Create(double* obs)
    {
        return (new ceres::AutoDiffCostFunction<RelativeSim3Error, 7, 7, 7>(new RelativeSim3Error(obs)));
    }

private:
    ceres::CostFunctionToFunctor<7,7,7> intrinsicRelativeSim3Error_;

};

}//! namespace ceres

}//! namespace ssvo

#endif