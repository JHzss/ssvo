#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "DBoW3/DBoW3.h"

#include <mutex>

namespace ssvo
{

class Map;

class KeyFrame: public Frame, public std::enable_shared_from_this<KeyFrame>
{
public:

    typedef std::shared_ptr<KeyFrame> Ptr;

    void updateConnections();

    void setBad();

    bool isBad();

    std::set<KeyFrame::Ptr> getConnectedKeyFrames(int num=-1, int min_fts = 0);

    std::set<KeyFrame::Ptr> getSubConnectedKeyFrames(int num=-1);

    std::set<KeyFrame::Ptr> getOrderedSubConnectedKeyFrames();

    //! roughly, only can be used in loopclosure
    std::vector<int > getFeaturesInArea(const double &x, const double &y, const double &r);

    const ImgPyr opticalImages() const = delete;    //! disable this function

    inline static KeyFrame::Ptr create(const Frame::Ptr frame)
    { return Ptr(new KeyFrame(frame)); }

    inline static KeyFrame::Ptr create(const Frame::Ptr frame, std::vector<IMUData> vIMUData, KeyFrame::Ptr pPrevKF)
    { return Ptr(new KeyFrame(frame, vIMUData, pPrevKF)); }

    void setNotErase();

    void setErase();

    void addLoopEdge(KeyFrame::Ptr pKF);

    int getWight(KeyFrame::Ptr pKF);

    KeyFrame::Ptr getParent();

    std::set<KeyFrame::Ptr> getLoopEdges();

private:

    KeyFrame(const Frame::Ptr frame);

    KeyFrame(const Frame::Ptr frame, std::vector<IMUData> vIMUData, KeyFrame::Ptr pPrevKF);

    void addConnection(const KeyFrame::Ptr &kf, const int weight);

    void updateOrderedConnections();

    void removeConnection(const KeyFrame::Ptr &kf);

public:

    static uint64_t next_id_;

    const uint64_t frame_id_;

    std::vector<Feature::Ptr> dbow_fts_;
    cv::Mat descriptors_;
    unsigned int dbow_Id_;

    DBoW3::BowVector bow_vec_;

    DBoW3::FeatureVector feat_vec_;


    //todo 可能有特征点融合的问题
    std::unordered_map<uint64_t , cv::Mat> mptId_des;

    std::unordered_map<uint64_t , uint> mptId_nodeId;

    std::unordered_map<uint64_t , uint > mptId_wordId;

    uint64_t loop_query_;

    uint64_t GBA_KF_;

    std::vector<cv::KeyPoint> KeyPoints;

    //解决mpt和feature无序的问题
    std::vector<Feature::Ptr> featuresInBow;
    std::vector<MapPoint::Ptr> mapPointsInBow;

    // Variables used by the local mapping
    uint64_t mnBALocalForKF;
    uint64_t mnBAFixedForKF;

private:

    std::map<KeyFrame::Ptr, int> connectedKeyFrames_;

    //todo remove from database
    DBoW3::Database* mpDatabase_;

    std::multimap<int, KeyFrame::Ptr> orderedConnectedKeyFrames_;

    bool isBad_;

    std::mutex mutex_connection_;

    bool notErase;

    bool toBeErase;

    std::set<KeyFrame::Ptr> loopEdges_;

    //todo 删除（bad）的时候记着改
    KeyFrame::Ptr parent_;

//! imu-------------------------------------------------
//! imu-------------------------------------------------
//! imu-------------------------------------------------
protected:

    std::mutex mMutexPrevKF;
    std::mutex mMutexNextKF;
    KeyFrame::Ptr mpPrevKeyFrame;
    KeyFrame::Ptr mpNextKeyFrame;

    // P, V, R, bg, ba, delta_bg, delta_ba (delta_bx is for optimization update)
    //todo 注意位姿更新的时候这个也要更新
//    NavState mNavState;

    // IMU Data from last KeyFrame to this KeyFrame ,构造关键帧的时候用到
    std::mutex mMutexIMUData;
    std::vector<IMUData> mvIMUData;
    // 关键帧的IMU预积分，是与上一个关键帧之间的预积分。在创建关键帧的时候就会直接积分。基类Frame中也有一个mIMUPreInt，积分的是与上一帧普通帧之间的IMU信息
    IMUPreintegrator mIMUPreInt; // 关键帧的IMU预积分，是与上一个关键帧之间的预积分，基类Frame中也有一个mIMUPreInt，积分的是与上一帧普通帧之间的IMU信息
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrame::Ptr GetPrevKeyFrame(void);
    KeyFrame::Ptr GetNextKeyFrame(void);
    void SetPrevKeyFrame(KeyFrame::Ptr pKF);
    void SetNextKeyFrame(KeyFrame::Ptr pKF);
    std::vector<IMUData> GetVectorIMUData(void);
    void AppendIMUDataToFront(KeyFrame::Ptr pPrevKF);
    void ComputePreInt(void);

    const IMUPreintegrator & GetIMUPreInt(void);



    void UpdatePoseFromNS(const Eigen::Matrix4d &Tbc);
    void UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw);
    void SetNavState(const NavState& ns);
    const NavState& GetNavState(void);
    void SetNavStateVel(const Vector3d &vel);
    void SetNavStatePos(const Vector3d &pos);
    void SetNavStateRot(const Matrix3d &rot);
    void SetNavStateRot(const Sophus::SO3d &rot);
    void SetNavStateBiasGyr(const Vector3d &bg);
    void SetNavStateBiasAcc(const Vector3d &ba);
    void SetNavStateDeltaBg(const Vector3d &dbg);
    void SetNavStateDeltaBa(const Vector3d &dba);



    // Variables used by loop closing
    NavState mNavStateGBA;       //mTcwGBA
    NavState mNavStateBefGBA;    //mTcwBefGBA


};

}

#endif