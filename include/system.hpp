#ifndef _SSVO_SYSTEM_HPP_
#define _SSVO_SYSTEM_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "initializer.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "local_mapping.hpp"
#include "depth_filter.hpp"
#include "viewer.hpp"
#include "src/IMU/imudata.hpp"

#ifdef SSVO_DBOW_ENABLE
#include "loop_closure.hpp"
#endif
namespace ssvo {

class System: public noncopyable
{
public:
    enum Stage{
        STAGE_INITALIZE,
        STAGE_NORMAL_FRAME,
        STAGE_RELOCALIZING
    };

    enum Status {
        STATUS_INITAL_RESET,
        STATUS_INITAL_PROCESS,
        STATUS_INITAL_SUCCEED,
        STATUS_TRACKING_BAD,
        STATUS_TRACKING_GOOD,
    };

    System(std::string config_file, std::string calib_flie);

    void saveTrajectoryTUM(const std::string &file_name);

    ~System();

    void process(const cv::Mat& image, const double timestamp);

    void process(const cv::Mat& image, const double timestamp,const std::vector<ssvo::IMUData> &vimu );

private:

    void processFrame();

    Status tracking();

    Status trackingVIO();

    Status initialize();

    Status relocalize();

    bool createNewKeyFrame();

    void finishFrame();

    void calcLightAffine();

    void drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst);

private:

    struct Option{
        double min_kf_disparity;
        double min_ref_track_rate;

    } options_;

    Stage stage_;
    Status status_;

    AbstractCamera::Ptr camera_;
    FastDetector::Ptr fast_detector_;
    FeatureTracker::Ptr feature_tracker_;
    Initializer::Ptr initializer_;
    DepthFilter::Ptr depth_filter_;
    LocalMapper::Ptr mapper_;

#ifdef SSVO_DBOW_ENABLE
    LoopClosure::Ptr loop_closure_;
#endif
    Viewer::Ptr viewer_;

    std::thread viewer_thread_;

    cv::Mat rgb_;
    Frame::Ptr last_frame_;
    Frame::Ptr current_frame_;
    KeyFrame::Ptr reference_keyframe_;
    KeyFrame::Ptr last_keyframe_;

    double time_;
    uint64_t loopId_;

    std::list<double > frame_timestamp_buffer_;
    std::list<Sophus::SE3d> frame_pose_buffer_;
    std::list<KeyFrame::Ptr> reference_keyframe_buffer_;

//! imu ---------------------------------------------
public:
    /**
     * 1.初始化完成后清空
     * 2.创建完关键帧后清空
     * 3.初始化重置时清空
     * 4.每一帧来的时候插入
     */
    bool mbCreateNewKFAfterReloc;
    bool mbRelocBiasPrepare;
    void RecomputeIMUBiasAndCurrentNavstate(NavState& nscur);
    // 20 Frames are used to compute bias
    std::vector<Frame::Ptr> mv20FramesReloc;

    // Predict the NavState of Current Frame by IMU
    void PredictNavStateByIMU(bool bMapUpdated);
    IMUPreintegrator mIMUPreIntInTrack;

    bool TrackWithIMU(bool bMapUpdated=false);
    bool TrackLocalMapWithIMU(bool bMapUpdated=false);

    ImuConfigParam* mpParams;
    cv::Mat GrabImageMonoVI(const cv::Mat &im, const std::vector<IMUData> &vimu, const double &timestamp);
    // IMU Data since last KF. Append when new data is provided
    // Should be cleared in 1. initialization beginning, 2. new keyframe created.
    std::vector<IMUData> mvIMUSinceLastKF;
    IMUPreintegrator GetIMUPreIntSinceLastKF(Frame::Ptr pCurF, KeyFrame::Ptr pLastKF, const std::vector<IMUData>& vIMUSInceLastKF);
    IMUPreintegrator GetIMUPreIntSinceLastFrame(Frame::Ptr pCurF, Frame::Ptr pLastF);

};

}// namespce ssvo

#endif //SSVO_SYSTEM_HPP
