#ifndef _SSVO_LOCAL_MAPPING_HPP_
#define _SSVO_LOCAL_MAPPING_HPP_

#include <future>
#include "global.hpp"
#include "feature_detector.hpp"
#include "brief.hpp"
#include "map.hpp"
#include "src/IMU/imudata.hpp"
#include "config.hpp"

#ifdef SSVO_DBOW_ENABLE
#include <DBoW3/DBoW3.h>
#include "loop_closure.hpp"
#endif

namespace ssvo{

class LoopClosure;

class LocalMapper : public noncopyable
{
public:

    typedef std::shared_ptr<LocalMapper> Ptr;

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur);

    void createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, std::vector<IMUData> &mvIMUSinceLastKF);

    void insertKeyFrame(const KeyFrame::Ptr &keyframe);

    void startMainThread();

    void stopMainThread();

    void addOptimalizeMapPoint(const MapPoint::Ptr &mpt);

    int refineMapPoints(const int max_optimalize_num = -1, const double outlier_thr = 2.0/480.0);

    void createFeatureFromSeed(const Seed::Ptr &seed);

    KeyFrame::Ptr relocalizeByDBoW(const Frame::Ptr &frame, const Corners &corners);

    static LocalMapper::Ptr create(const FastDetector::Ptr fast, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(fast, report, verbose));}

#ifdef SSVO_DBOW_ENABLE
    static LocalMapper::Ptr create(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database, const FastDetector::Ptr fast, ImuConfigParam* pParams, bool report = false, bool verbose = false)
    { return LocalMapper::Ptr(new LocalMapper(vocabulary, database, fast, pParams ,report, verbose));}
        
    void setLoopCloser(std::shared_ptr<LoopClosure> loop_closure);
#endif

    void setStop();

    bool isRequiredStop();

    void release();

    bool finish_once();
private:

    LocalMapper(const FastDetector::Ptr fast, bool report, bool verbose);

#ifdef SSVO_DBOW_ENABLE
    LocalMapper(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database,const FastDetector::Ptr fast,ImuConfigParam* pParams, bool report, bool verbose );

    std::shared_ptr<LoopClosure> loop_closure_;
#endif
    void run();

    KeyFrame::Ptr checkNewKeyFrame();

    void finishLastKeyFrame();

    int createFeatureFromSeedFeature(const KeyFrame::Ptr &keyframe);

    int createFeatureFromLocalMap(const KeyFrame::Ptr &keyframe, const int num = 5);

    void checkCulling(const KeyFrame::Ptr &keyframe);

    void addToDatabase(const KeyFrame::Ptr &keyframe);

    void RunGlobalBundleAdjustment(uint64_t nLoopKF,const bool vi = false );

public:

    Map::Ptr map_;

    std::mutex update_finish_mutex_;
    bool update_finish_;

    uint64_t init_kfID_;

private:

    struct Option{
        double min_disparity;
        int min_redundant_observations;
        int max_features;
        int num_reproject_kfs;
        int num_local_ba_kfs;
        int min_local_ba_connected_fts;
        int num_align_iter;
        double max_align_epsilon;
        double max_align_error2;
        double min_found_ratio_;
    } options_;

    FastDetector::Ptr fast_detector_;

    BRIEF::Ptr brief_;

    std::deque<KeyFrame::Ptr> keyframes_buffer_;
    KeyFrame::Ptr keyframe_last_;

#ifdef SSVO_DBOW_ENABLE
    DBoW3::Vocabulary* vocabulary_;
    DBoW3::Database* database_;

#endif

    const bool report_;
    const bool verbose_;

    std::shared_ptr<std::thread> mapping_thread_;

    std::list<MapPoint::Ptr> optimalize_candidate_mpts_;

    bool stop_require_;
    bool finish_once_;
    std::mutex mutex_stop_;
    std::mutex mutex_keyframe_;
    std::mutex mutex_optimalize_mpts_;
    std::condition_variable cond_process_;


    // Variables related to Global Bundle Adjustment
    bool RunningGBA_;
    bool FinishedGBA_;
    //todo 还要在迭代的时候设置是否停止的标志位
    bool StopGBA_;
    std::mutex mutex_GBA_;
    std::thread* thread_GBA_;
    bool FullBAIdx_;

    //! 通过最小得分计算的闭环次数，仅用于输出信息
    int loop_time_;



    //! imu--------------------------
public:
    ImuConfigParam* mpParams;

    // KeyFrames in Local Window, for Local BA
    // Insert in ProcessNewKeyFrame()
    void AddToLocalWindow(KeyFrame::Ptr pKF);
    void DeleteBadInLocalWindow(void);

    //实时的时候用到的
    void VINSInitThread(void);


    bool TryInitVIO();
    bool GetVINSInited(void);
    void SetVINSInited(bool flag);

    bool GetFirstVINSInited(void);
    void SetFirstVINSInited(bool flag);

    Vector3d GetGravityVec(void);
    Eigen::Matrix3d GetRwiInit(void);

    bool GetMapUpdateFlagForTracking();
    void SetMapUpdateFlagForTracking(bool bflag);
    KeyFrame::Ptr GetMapUpdateKF();

    const KeyFrame::Ptr GetCurrentKF(void) const {return mpCurrentKeyFrame;}

    std::mutex mMutexUpdatingInitPoses;
    bool GetUpdatingInitPoses(void);
    void SetUpdatingInitPoses(bool flag);

    std::mutex mMutexInitGBAFinish;
    bool mbInitGBAFinish;
    bool GetFlagInitGBAFinish() { std::unique_lock<std::mutex> lock(mMutexInitGBAFinish); return mbInitGBAFinish; }
    void SetFlagInitGBAFinish(bool flag) { std::unique_lock<std::mutex> lock(mMutexInitGBAFinish); mbInitGBAFinish = flag; }


protected:
    double mnStartTime;
    bool mbFirstTry;

    //估计尺度的结果
    double mnVINSInitScale;

    Vector3d mGravityVec;
    Eigen::Matrix3d mRwiInit;

    std::mutex mMutexVINSInitFlag;
    bool mbVINSInited;

    std::mutex mMutexFirstVINSInitFlag;
    bool mbFirstVINSInited;

    unsigned int mnLocalWindowSize;
    std::list<KeyFrame::Ptr> mlLocalKeyFrames;

    std::mutex mMutexMapUpdateFlag;
    bool mbMapUpdateFlagForTracking;
    KeyFrame::Ptr mpMapUpdateKF;
    KeyFrame::Ptr mpCurrentKeyFrame; //local mapper 正在处理的关键帧

    bool mbUpdatingInitPoses;

//    std::mutex mMutexCopyInitKFs;

    //实时的时候用到的
//    bool mbCopyInitKFs;
//    bool GetFlagCopyInitKFs() { std::unique_lock<std::mutex> lock(mMutexCopyInitKFs); return mbCopyInitKFs; }
//    void SetFlagCopyInitKFs(bool flag) { std::unique_lock<std::mutex> lock(mMutexCopyInitKFs); mbCopyInitKFs = flag; }

};

class KeyFrameInit
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<KeyFrameInit> Ptr;

    KeyFrameInit(KeyFrame::Ptr kf):
            mTimeStamp(kf->timestamp_), mpPrevKeyFrame(NULL), Twc(kf->Twc()),
            mIMUPreInt(kf->GetIMUPreInt()), mvIMUData(kf->GetVectorIMUData()), bg(0,0,0)
    {}

    inline static Ptr Creat(KeyFrame::Ptr kf)
    {
        return Ptr(new KeyFrameInit(kf));
    }
    void ComputePreInt(void)
    {
        if(mpPrevKeyFrame == NULL)
        {
            return;
        }
        else
        {
            // Reset pre-integrator first
            mIMUPreInt.reset();

            if(mvIMUData.empty())
                return;

            // remember to consider the gap between the last KF and the first IMU
            {
                const IMUData& imu = mvIMUData.front();
                double dt = std::max(0., imu._t - mpPrevKeyFrame->mTimeStamp);
                mIMUPreInt.update(imu._g - bg,imu._a ,dt);  // Acc bias not considered here
            }
            // integrate each imu
            for(size_t i=0; i<mvIMUData.size(); i++)
            {
                const IMUData& imu = mvIMUData[i];
                double nextt;
                if(i==mvIMUData.size()-1)
                    nextt = mTimeStamp;         // last IMU, next is this KeyFrame
                else
                    nextt = mvIMUData[i+1]._t;  // regular condition, next is imu data

                // delta time
                double dt = std::max(0., nextt - imu._t);
                // update pre-integrator
                mIMUPreInt.update(imu._g - bg,imu._a ,dt);
            }
        }
    }

public:
    double mTimeStamp;
    KeyFrameInit::Ptr mpPrevKeyFrame;
    SE3d Twc;
    IMUPreintegrator mIMUPreInt;
    std::vector<IMUData> mvIMUData;
    Vector3d bg;


};
}

#endif //_SSVO_LOCAL_MAPPING_HPP_
