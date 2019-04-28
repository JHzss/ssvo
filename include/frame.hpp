#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include "global.hpp"
#include "camera.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "seed.hpp"
#include "feature_detector.hpp"
#include "src/IMU/imudata.hpp"
#include "src/IMU/NavState.hpp"
#include "src/IMU/IMUPreintegrator.hpp"

namespace ssvo{

class KeyFrame;

class Frame : public noncopyable
{
public:

    virtual ~Frame() {};

    typedef std::shared_ptr<Frame> Ptr;

    const ImgPyr images() const;

    const ImgPyr opticalImages() const;

    const cv::Mat getImage(int level) const;

    //! Transform (c)amera from (w)orld
    SE3d Tcw();

    //! Transform (w)orld from (c)amera
    SE3d Twc();

    //! Transform (w)orld from (c)amera
    SE3d pose();

    //! Principal ray in world frame
    Vector3d ray();

    //! Set pose in world frame
    void setPose(const SE3d& pose);

    //! Set pose in world frame
    void setPose(const Matrix3d& R, const Vector3d& t);

    //! Set Extrinsic Matrix
    void setTcw(const SE3d& Tcw);

    void setTwb(const SE3d& Twb);

    bool isVisiable(const Vector3d &xyz_w, const int border = 0);

    //! Feature created by MapPoint
    int featureNumber();

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> features();

    std::vector<Feature::Ptr> getFeatures();


    std::vector<MapPoint::Ptr> getMapPoints();

    //! keep the fts and mappoints order right
    void getFeaturesAndMapPoints(std::vector<Feature::Ptr> &features, std::vector<MapPoint::Ptr> &mappoints);

    bool addFeature(const Feature::Ptr &ft);

    bool removeFeature(const Feature::Ptr &ft);

    bool removeMapPoint(const MapPoint::Ptr &mpt);

    Feature::Ptr getFeatureByMapPoint(const MapPoint::Ptr &mpt);

    //! Feature created by Seed
    int seedNumber();

    std::vector<Feature::Ptr> getSeeds();

    std::vector<Seed::Ptr> getTrueSeeds();

    bool addSeed(const Feature::Ptr &ft);

    bool removeSeed(const Seed::Ptr &seed);

    bool hasSeed(const Seed::Ptr &seed);

    bool getSceneDepth(double &depth_mean, double &depth_min);

    std::map<std::shared_ptr<KeyFrame>, int> getOverLapKeyFrames();

    inline void setRefKeyFrame(const std::shared_ptr<KeyFrame> &kf) {ref_keyframe_ = kf;}

    inline std::shared_ptr<KeyFrame> getRefKeyFrame() const {return ref_keyframe_;}

    inline static Ptr create(const cv::Mat& img, const double timestamp, AbstractCamera::Ptr cam)
    { return Ptr(new Frame(img, timestamp, cam)); }

    inline static Ptr create(const cv::Mat& img, const double timestamp, AbstractCamera::Ptr cam, const std::vector<ssvo::IMUData> &vimu)
    { return Ptr(new Frame(img, timestamp, cam, vimu)); }

protected:

    Frame(const cv::Mat& img, const double timestamp, const AbstractCamera::Ptr &cam);

    Frame(const cv::Mat& img, const double timestamp, const AbstractCamera::Ptr &cam, const std::vector<ssvo::IMUData> &vimu);

    Frame(const ImgPyr& img_pyr, const uint64_t id, const double timestamp, const AbstractCamera::Ptr &cam);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static uint64_t next_id_;
    const uint64_t id_;
    const double timestamp_;

    AbstractCamera::Ptr cam_;

    const int max_level_;
    static const cv::Size optical_win_size_;
    static float light_affine_a_;
    static float light_affine_b_;

    Sophus::SE3d optimal_Tcw_;//! for optimization


    //! for vi optimization
    Sophus::SE3d optimal_Twb_;//! for optimization
    Eigen::Vector3d optimal_v_;//! for optimization
    Sophus::Vector6d optimal_detla_bias_;//! for optimization
    Sophus::Vector6d optimal_PR_;//! for optimization
//    Sophus::Vector6d optimal_PRV_;//! for optimization
//    Sophus::Vector6d optimal_PVR_;//! for optimization



    double disparity_;//! for depth filter

    SE3d beforeGBA_Tcw_;
    SE3d beforeUpdate_Tcw_;

protected:

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts_;

    std::unordered_map<Seed::Ptr, Feature::Ptr> seed_fts_;

    ImgPyr img_pyr_;

    SE3d Tcw_;
    SE3d Twc_;
    Vector3d Dw_;

    //!for imu
    SE3d Tbw_;
    SE3d Twb_;

    std::shared_ptr<KeyFrame> ref_keyframe_;

    std::mutex mutex_pose_;
    std::mutex mutex_feature_;
    std::mutex mutex_seed_;

private:

    ImgPyr optical_pyr_;



//! imu--------------------
public:

    void ComputeIMUPreIntSinceLastFrame(const Frame::Ptr pLastF, IMUPreintegrator& imupreint) const;
    void UpdatePoseFromNS(const Matrix4d &Tbc);
    void SetInitialNavStateAndBias(const NavState& ns);
    void UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw);
    const NavState& GetNavState(void) const;
    void SetNavState(const NavState& ns);
    void SetNavStateBiasGyr(const Vector3d &bg);
    void SetNavStateBiasAcc(const Vector3d &ba);

    void UpdateNavStatePVRFromTcw(const SE3d &Tbc);
    void UpdateNavStatePRFromTcw(const SE3d &Tcw,const SE3d &Tbc);
    void setOptimizationState();
        const IMUPreintegrator & GetIMUPreInt(void);

    // IMU Data from last Frame to this Frame 构造帧的时候赋值
    std::vector<IMUData> mvIMUDataSinceLastFrame;

    // For pose optimization, use as prior and prior information(inverse covariance)
    Matrix<double,15,15> mMargCovInv;
    NavState mNavStatePrior;

protected:
    std::mutex mMutexNavState;
    NavState mNavState;

    // IMU Data from lask Frame to this Frame
    std::mutex mMutexIMUData;
    IMUPreintegrator mIMUPreInt;
};

}

#endif