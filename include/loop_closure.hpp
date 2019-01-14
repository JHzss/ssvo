//
// Created by jh on 18-11-29.
//

#ifndef SSVO_LOOP_CLOSURE_HPP
#define SSVO_LOOP_CLOSURE_HPP

#include "global.hpp"
#include "DBoW3/DBoW3.h"
#include "DBoW3/DescManip.h"
#include "keyframe.hpp"
#include "sim3_solver.hpp"
#include "brief.hpp"
#include <eigen3/Eigen/Dense>
#include "local_mapping.hpp"

namespace ssvo{

class LocalMapper;

class LoopClosure
{

public:
    typedef std::pair<std::set<KeyFrame::Ptr>,int> ConsistentGroup;

    typedef std::map<KeyFrame::Ptr,Sophus::Sim3d ,std::less<KeyFrame::Ptr>,
            Eigen::aligned_allocator<std::pair<const KeyFrame::Ptr, Sophus::Sim3d> > > KeyFrameAndPose;

    typedef std::shared_ptr<LoopClosure> Ptr;

    void startMainThread();

    void stopMainThread();

    void insertKeyFrame(KeyFrame::Ptr kf);

    void run();

    inline static Ptr creat(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database)
    { return Ptr(new LoopClosure(vocabulary, database));}

    void setLocalMapper(std::shared_ptr<LocalMapper> local_mapper);

    bool isRunningGBA(){
        std::unique_lock<std::mutex> lock(mutex_GBA_);
        return RunningGBA_;
    }
    bool isFinishedGBA(){
        std::unique_lock<std::mutex> lock(mutex_GBA_);
        return FinishedGBA_;
    }

public:
    std::list<KeyFrame::Ptr> keyFramesList_; //list-快速的插入和删除，可以在两端进行操作

    std::vector<KeyFrame::Ptr> KeyFrames;

    std::mutex mutex_database_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    uint64_t LastLoopKFid_;

    bool update_finish_;


private:
    LoopClosure(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database_);

    bool CheckNewKeyFrames();

    bool DetectLoop();

    bool ComputeSim3();

    void CorrectLoop();

    int SearchByBoW(KeyFrame::Ptr loopKeyFrame, std::vector<MapPoint::Ptr> &Matches12, std::vector<int > &bestidx);

    int SearchBySim3(KeyFrame::Ptr loopKeyFrame,
                     std::unordered_map<uint64_t,uint64_t > &matches_1_2,std::unordered_map<uint64_t,uint64_t > &matches_2_1,
                     double &s12, const Matrix3d &R12, const Vector3d &t12, float th, std::vector<MapPoint::Ptr> &Matches12,  std::vector<int > &bestidx);

    int SearchByProjection(int th );

    void searchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

    int fuse(KeyFrame::Ptr pKF,const Sophus::Sim3d& Scw, float th, std::vector<MapPoint::Ptr > &vpReplacePoint);

    void RunGlobalBundleAdjustment(uint64_t nLoopKF);

private:
    std::vector<KeyFrame::Ptr> DetectLoopCandidates(double minScore);

    DBoW3::Vocabulary* vocabulary_;
    DBoW3::Database* database_;

    KeyFrame::Ptr curKeyFrame_;
    KeyFrame::Ptr MatchedKeyFrame_;
    std::vector<ConsistentGroup> mvConsistentGroups;
    std::vector<KeyFrame::Ptr> mvpEnoughConsistentCandidates;
    std::vector<KeyFrame::Ptr> mvpCurrentConnectedKFs;
    std::vector<MapPoint::Ptr> CurrentMatchedPoints; //当前匹配的mappoint
//    *   [mpt1_1  mpt_a_2 ]  一一匹配
//    *   [mpt2_1  mpt_b_2 ]
//    *   [mpt3_1  mpt_c_2 ]
//    *   [mpt4_1  mpt_d_2 ]
//    *   [mpt5_1  mpt_e_m ]
//    *   [mpt6_1  mpt_f_n ]
//    *   [mpt7_1  mpt_g_n ]
//    *   [mpt8_1  mpt_h_k ]
    std::vector<MapPoint::Ptr> LoopMapPoints; //闭环帧及其共视帧的所有观测到的mpt，设置为set，不包含相同的元素



    Sophus::Sim3d sim3_cw;
    SE3d T_cw; //通过闭环帧求得的当前帧的位姿，而不是自身的Tcw

    std::shared_ptr<std::thread> loop_closure_thread_;

    std::mutex mutex_keyFramesList_;


    double mnCovisibilityConsistencyTh; //连续性检测的阈值

    bool ifFinished;

    std::shared_ptr<LocalMapper> local_mapper_;


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

    std::condition_variable cond_process_;
};

}



#endif //SSVO_LOOP_CLOSURE_HPP
