#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "keyframe.hpp"
#include "utils.hpp"
#include "config.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;
const cv::Size Frame::optical_win_size_ = cv::Size(21,21);
float Frame::light_affine_a_ = 1.0f;
float Frame::light_affine_b_ = 0.0f;

Frame::Frame(const cv::Mat &img, const double timestamp, const AbstractCamera::Ptr &cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), max_level_(Config::imageNLevel()-1)
{
    Tcw_ = SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

//    utils::createPyramid(img, img_pyr_, nlevels_);
    //! create pyramid for optical flow
    cv::buildOpticalFlowPyramid(img, optical_pyr_, optical_win_size_, max_level_, false);
    LOG_ASSERT(max_level_ == (int) optical_pyr_.size()-1) << "The pyramid level is unsuitable! maxlevel should be " << optical_pyr_.size()-1;

    //! copy to image pyramid
    img_pyr_.resize(optical_pyr_.size());
    for(size_t i = 0; i < optical_pyr_.size(); i++)
        optical_pyr_[i].copyTo(img_pyr_[i]);

}

Frame::Frame(const cv::Mat &img, const double timestamp, const AbstractCamera::Ptr &cam, const std::vector<ssvo::IMUData> &vimu) :
        id_(next_id_++), timestamp_(timestamp), cam_(cam), max_level_(Config::imageNLevel()-1)
{
    //保存两帧之间的imu数据,这个绝对是正确的
    mvIMUDataSinceLastFrame = vimu;

    Tcw_ = SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

//    utils::createPyramid(img, img_pyr_, nlevels_);
    //! create pyramid for optical flow
    cv::buildOpticalFlowPyramid(img, optical_pyr_, optical_win_size_, max_level_, false);
    LOG_ASSERT(max_level_ == (int) optical_pyr_.size()-1) << "The pyramid level is unsuitable! maxlevel should be " << optical_pyr_.size()-1;

    //! copy to image pyramid
    img_pyr_.resize(optical_pyr_.size());
    for(size_t i = 0; i < optical_pyr_.size(); i++)
        optical_pyr_[i].copyTo(img_pyr_[i]);
}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, const AbstractCamera::Ptr &cam) :
    id_(id), timestamp_(timestamp), cam_(cam), max_level_(Config::imageNLevel()-1), img_pyr_(img_pyr),
    Tcw_(SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

const ImgPyr Frame::images() const
{
    return img_pyr_;
}

const ImgPyr Frame::opticalImages() const
{
    return optical_pyr_;
}

const cv::Mat Frame::getImage(int level) const
{
    LOG_ASSERT(level < (int) img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}

SE3d Frame::Tcw()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Tcw_;
}

SE3d Frame::Twc()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

SE3d Frame::pose()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

Vector3d Frame::ray()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Dw_;
}

void Frame::setPose(const SE3d& pose)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = pose;
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    SE3d Tcb(ImuConfigParam::GetEigTcb());
    Twb_ = Twc_ * Tcb;
    Tbw_ = Twb_.inverse();

//    UpdateNavStatePVRFromTcw(Tcw_,Tcb.inverse());
}

void Frame::setPose(const Matrix3d& R, const Vector3d& t)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = SE3d(R, t);
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    SE3d Tcb(ImuConfigParam::GetEigTcb());
    Twb_ = Twc_ * Tcb;
    Tbw_ = Twb_.inverse();

//    UpdateNavStatePVRFromTcw(Tcw_,Tcb.inverse());

}

void Frame::setTcw(const SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Tcw_ = Tcw;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

    SE3d Tcb(ImuConfigParam::GetEigTcb());
    Twb_ = Twc_ * Tcb;
    Tbw_ = Twb_.inverse();

//    UpdateNavStatePRFromTcw(Tcw_,Tcb.inverse());
}

void Frame::setTwb(const SE3d &Twb)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);

    Twb_ = Twb;
    Tbw_ = Twb_.inverse();
    SE3d Tcb(ImuConfigParam::GetEigTcb());
    Tcw_ = Tcb * Tbw_;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

//    UpdateNavStatePVRFromTcw(Tcw_,Tcb.inverse());
}

bool Frame::isVisiable(const Vector3d &xyz_w, const int border)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    const Vector3d xyz_c = Tcw * xyz_w;
    if(xyz_c[2] < 0.0f)
        return false;

    Vector2d px = cam_->project(xyz_c);
    return cam_->isInFrame(px.cast<int>(), border);
}

std::unordered_map<MapPoint::Ptr, Feature::Ptr> Frame::features()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return mpt_fts_;
}

int Frame::featureNumber()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (int)mpt_fts_.size();
}

std::vector<Feature::Ptr> Frame::getFeatures()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    std::vector<Feature::Ptr> fts;
    fts.reserve(mpt_fts_.size());
    for(const auto &it : mpt_fts_)
        fts.push_back(it.second);

    return fts;
}

std::vector<MapPoint::Ptr> Frame::getMapPoints()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    std::vector<MapPoint::Ptr> mpts;
    mpts.reserve(mpt_fts_.size());
    for(const auto &it : mpt_fts_)
        mpts.push_back(it.first);

    return mpts;
}

void Frame::getFeaturesAndMapPoints(std::vector<Feature::Ptr> &features, std::vector<MapPoint::Ptr> &mappoints)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    for(const auto &it : mpt_fts_)
    {
        features.push_back(it.second);
        mappoints.push_back(it.first);
    }
}

bool Frame::addFeature(const Feature::Ptr &ft)
{
    LOG_ASSERT(ft->mpt_ != nullptr) << " The feature is invalid with empty mappoint!";
    std::lock_guard<std::mutex> lock(mutex_feature_);
    if(mpt_fts_.count(ft->mpt_))
    {
        LOG(ERROR) << " The mappoint is already be observed! Frame: " << id_ << " Mpt: " << ft->mpt_->id_
            << ", px: " << mpt_fts_.find(ft->mpt_)->second->px_.transpose() << ", " << ft->px_.transpose();
        return false;
    }

    mpt_fts_.emplace(ft->mpt_, ft);

    return true;
}

bool Frame::removeFeature(const Feature::Ptr &ft)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (bool)mpt_fts_.erase(ft->mpt_);
}

bool Frame::removeMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return (bool)mpt_fts_.erase(mpt);
}

Feature::Ptr Frame::getFeatureByMapPoint(const MapPoint::Ptr &mpt)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    const auto it = mpt_fts_.find(mpt);
    if(it != mpt_fts_.end())
        return it->second;
    else
        return nullptr;
}

int Frame::seedNumber()
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (int)seed_fts_.size();
}

std::vector<Feature::Ptr> Frame::getSeeds()
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    std::vector<Feature::Ptr> fts;
    fts.reserve(seed_fts_.size());
    for(const auto &it : seed_fts_)
        fts.push_back(it.second);

    return fts;
}

std::vector<Seed::Ptr> Frame::getTrueSeeds()
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    std::vector<Seed::Ptr> seeds;
    seeds.reserve(seed_fts_.size());
    for(const auto &it : seed_fts_)
        seeds.push_back(it.first);

    return seeds;
}

bool Frame::addSeed(const Feature::Ptr &ft)
{
    LOG_ASSERT(ft->seed_ != nullptr) << " The feature is invalid with empty mappoint!";

    {
        std::lock_guard<std::mutex> lock(mutex_seed_);
        if(seed_fts_.count(ft->seed_))
        {
            LOG(ERROR) << " The seed is already exited ! Frame: " << id_ << " Seed: " << ft->seed_->id;
            return false;
        }

        seed_fts_.emplace(ft->seed_, ft);
    }

    return true;
}

bool Frame::removeSeed(const Seed::Ptr &seed)
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (bool) seed_fts_.erase(seed);
}

bool Frame::hasSeed(const Seed::Ptr &seed)
{
    std::lock_guard<std::mutex> lock(mutex_seed_);
    return (bool) seed_fts_.count(seed);
}

bool Frame::getSceneDepth(double &depth_mean, double &depth_min)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        for(const auto &it : mpt_fts_)
            fts.push_back(it.second);
    }

    std::vector<double> depth_vec;
    depth_vec.reserve(fts.size());

    depth_min = std::numeric_limits<double>::max();
    for(const Feature::Ptr &ft : fts)
    {
        if(ft->mpt_ == nullptr)
            continue;

        const Vector3d p =  Tcw * ft->mpt_->pose();
        depth_vec.push_back(p[2]);
        depth_min = fmin(depth_min, p[2]);
    }

    if(depth_vec.empty())
        return false;

    depth_mean = utils::getMedian(depth_vec);
    return true;
}

std::map<KeyFrame::Ptr, int> Frame::getOverLapKeyFrames()
{
    std::vector<MapPoint::Ptr> mpts = getMapPoints();

    std::map<KeyFrame::Ptr, int> overlap_kfs;

    for(const MapPoint::Ptr &mpt : mpts)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            auto it = overlap_kfs.find(item.first);
            if(it != overlap_kfs.end())
                it->second++;
            else
                overlap_kfs.insert(std::make_pair(item.first, 1));
        }
    }

    return overlap_kfs;
}

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

void Frame::ComputeIMUPreIntSinceLastFrame(const Frame::Ptr pLastF, IMUPreintegrator& IMUPreInt) const
{
    // Reset pre-integrator first
    IMUPreInt.reset();

    const std::vector<IMUData>& vIMUSInceLastFrame = mvIMUDataSinceLastFrame;

    Vector3d bg = pLastF->GetNavState().Get_BiasGyr();
    Vector3d ba = pLastF->GetNavState().Get_BiasAcc();

    // remember to consider the gap between the last KF and the first IMU
    {
        const IMUData& imu = vIMUSInceLastFrame.front();
        double dt = imu._t - pLastF->timestamp_;
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log
        if(dt < 0)
        {
            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this frame vs last imu time: "<<pLastF->timestamp_<<" vs "<<imu._t<<std::endl;
            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
        }
    }
    // integrate each imu
    for(size_t i=0; i<vIMUSInceLastFrame.size(); i++)
    {
        const IMUData& imu = vIMUSInceLastFrame[i];
        double nextt;
        if(i==vIMUSInceLastFrame.size()-1)
            nextt = timestamp_;         // last IMU, next is this KeyFrame
        else
            nextt = vIMUSInceLastFrame[i+1]._t;  // regular condition, next is imu data

        // delta time
        double dt = nextt - imu._t;

        LOG_ASSERT(dt>-1e-5)<<"dt is '-', please check";

        // update pre-integrator
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log

//        if(dt <= -1e-6)
//        {
//            LOG_ASSERT()
//            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu._t<<" vs "<<nextt<<std::endl;
//            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
//        }
    }
}

void Frame::UpdatePoseFromNS(const Matrix4d &Tbc)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Matrix3d Rbc = Tbc.block(0,0,3,3);
    Vector3d Pbc = Tbc.block(0,3,3,1);

    Matrix3d Rwb = mNavState.Get_RotMatrix();
    Vector3d Pwb = mNavState.Get_P();
    SE3d Twb(Rwb,Pwb);
    Twb_ = Twb;
    Tbw_ = Twb_.inverse();
    SE3d Tcb(ImuConfigParam::GetEigTcb());
    Tcw_ = Tcb * Tbw_;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);
//    std::lock_guard<std::mutex> lock1(mutex_pose_);
}

void Frame::UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw)
{
    Matrix3d dR = imupreint.getDeltaR();
    Vector3d dP = imupreint.getDeltaP();
    Vector3d dV = imupreint.getDeltaV();
    double dt = imupreint.getDeltaTime();

    Vector3d Pwbpre = mNavState.Get_P();
    Matrix3d Rwbpre = mNavState.Get_RotMatrix();
    Vector3d Vwbpre = mNavState.Get_V();

    Matrix3d Rwb = Rwbpre * dR;
    Vector3d Pwb = Pwbpre + Vwbpre*dt + 0.5*gw*dt*dt   + Rwbpre*dP;
    Vector3d Vwb = Vwbpre + gw*dt + Rwbpre*dV;

    // Here assume that the pre-integration is re-computed after bias updated, so the bias term is ignored
    mNavState.Set_Pos(Pwb);
    mNavState.Set_Vel(Vwb);
    mNavState.Set_Rot(Rwb);

}

const NavState& Frame::GetNavState(void) const
{
    return mNavState;
}

void Frame::SetInitialNavStateAndBias(const NavState& ns)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState = ns;
    // Set bias as bias+delta_bias, and reset the delta_bias term
    mNavState.Set_BiasGyr(ns.Get_BiasGyr()+ns.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(ns.Get_BiasAcc()+ns.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}


void Frame::SetNavStateBiasGyr(const Vector3d &bg)
{
    mNavState.Set_BiasGyr(bg);
}

void Frame::SetNavStateBiasAcc(const Vector3d &ba)
{
    mNavState.Set_BiasAcc(ba);
}

void Frame::SetNavState(const NavState& ns)
{
    mNavState = ns;

}

void Frame::UpdateNavStatePVRFromTcw(const SE3d &Tbc)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    SE3d Twb = (Tbc*Tcw_).inverse();
    Matrix3d Rwb = Twb.rotationMatrix();
    Vector3d Pwb = Twb.translation();

    Matrix3d Rw1 = mNavState.Get_RotMatrix();
    Vector3d Vw1 = mNavState.Get_V();
    Vector3d Vw2 = Rwb * Rw1.transpose()*Vw1;   // bV1 = bV2 ==> Rwb1^T*wV1 = Rwb2^T*wV2 ==> wV2 = Rwb2*Rwb1^T*wV1

    mNavState.Set_Pos(Pwb);
    mNavState.Set_Rot(Rwb);
    mNavState.Set_Vel(Vw2);
}
void Frame::UpdateNavStatePRFromTcw(const SE3d &Tcw,const SE3d &Tbc)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    SE3d Twb = (Tbc*Tcw).inverse();
    Matrix3d Rwb = Twb.rotationMatrix();
    Vector3d Pwb = Twb.translation();

    mNavState.Set_Pos(Pwb);
    mNavState.Set_Rot(Rwb);
}

//! 从Navstate设置待优化变量的状态，即赋初值
void Frame::setOptimizationState()
{
    Vector3d Pwb = mNavState.Get_P();
    Sophus::SO3d Rwb = mNavState.Get_R();

    optimal_PR_[0] = Pwb[0];
    optimal_PR_[1] = Pwb[1];
    optimal_PR_[2] = Pwb[2];

    Vector3d phi = Rwb.log();

    optimal_PR_[3] = phi[0];
    optimal_PR_[4] = phi[1];
    optimal_PR_[5] = phi[2];

    optimal_v_ = mNavState.Get_V();

    optimal_detla_bias_.segment(0,3) = mNavState.Get_dBias_Gyr();
    optimal_detla_bias_.segment(3,3) = mNavState.Get_dBias_Acc();

}

const IMUPreintegrator &Frame::GetIMUPreInt(void)
{
    std::unique_lock<std::mutex> lock(mMutexIMUData);
    return mIMUPreInt;
}

}