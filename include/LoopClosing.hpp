//
// Created by jh on 18-11-17.
//
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef SSVO_LOOPCLOSING_HPP
#define SSVO_LOOPCLOSING_HPP

#include "global.hpp"
#include "keyframe.hpp"
//#include "local_mapping.hpp"
#include "map.hpp"
//#include "ORBVocabulary.h"
//#include "feature_tracker.hpp"
#include "depth_filter.hpp"
//
//#include "KeyFrameDatabase.h"
//
//#include <thread>
//#include <mutex>


namespace ssvo {

class LoopClosing {
public:
    typedef std::shared_ptr<LoopClosing> Ptr;

    typedef std::pair<std::set<KeyFrame::Ptr>,int> ConsistentGroup;
//    typedef map<KeyFrame*,g2o::Sim3,std::less<KeyFrame*>,
//        Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;

        static LoopClosing::Ptr creat(Map::Ptr pMap, DBoW3::Database pDB, DBoW3::Vocabulary pVoc,const bool bFixScale)
        { return LoopClosing::Ptr(new LoopClosing(pMap, pDB, pVoc, bFixScale));}

        //    void SetTracker(Tracking* pTracker);
//    void SetTracker(FeatureTracker::Ptr pTracker);

//    void SetLocalMapper(LocalMapping* pLocalMapper);
//    void SetLocalMapper(LocalMapper::Ptr pLocalMapper);


        void startMainThread();

        void stopMainThread();

        // Main function
        void Run();

//    void InsertKeyFrame(KeyFrame *pKF);
        void InsertKeyFrame(KeyFrame::Ptr pKF);

        void RequestReset();

        // This function will run in a separate thread
        void RunGlobalBundleAdjustment(unsigned long nLoopKF);

        bool isRunningGBA(){
            std::unique_lock<std::mutex> lock(mMutexGBA);
            return mbRunningGBA;
        }
        bool isFinishedGBA(){
            std::unique_lock<std::mutex> lock(mMutexGBA);
            return mbFinishedGBA;
        }

        void RequestFinish();

        bool isFinished();

        void setStop();

        bool isRequiredStop();

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        LoopClosing(Map::Ptr pMap, DBoW3::Database pDB, DBoW3::Vocabulary pVoc,const bool bFixScale);

    protected:

        bool CheckNewKeyFrames();

        bool DetectLoop();

        bool ComputeSim3();

//    void SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap);

        void CorrectLoop();

        void ResetIfRequested();
        bool mbResetRequested;
        std::mutex mMutexReset;

        bool CheckFinish();
        void SetFinish();
        bool mbFinishRequested;
        bool mbFinished;
        std::mutex mMutexFinish;


        std::shared_ptr<std::thread> loop_closing_thread_;

//    Map* mpMap;
        Map::Ptr mpMap;
//    Tracking* mpTracker;
//    FeatureTracker::Ptr mpTracker;

//    KeyFrameDatabase* mpKeyFrameDB;
        DBoW3::Database mpKeyFrameDB;
//    ORBVocabulary* mpORBVocabulary;
        DBoW3::Vocabulary mpORBVocabulary;

//    LocalMapping *mpLocalMapper;
//    LocalMapper::Ptr mpLocalMapper;

//    std::list<KeyFrame*> mlpLoopKeyFrameQueue;
        //! 在localmapping中插入，在检测的环节删除
        std::list<KeyFrame::Ptr> mlpLoopKeyFrameQueue;

        std::mutex mMutexLoopQueue;

        // Loop detector parameters
        float mnCovisibilityConsistencyTh;

        // Loop detector variables
//        KeyFrame* mpCurrentKF;
//        KeyFrame* mpMatchedKF;
        KeyFrame::Ptr mpCurrentKF;
        KeyFrame::Ptr mpMatchedKF;
        std::vector<ConsistentGroup> mvConsistentGroups;
        std::vector<KeyFrame::Ptr> mvpEnoughConsistentCandidates;
        std::vector<KeyFrame::Ptr> mvpCurrentConnectedKFs;
        std::vector<MapPoint::Ptr> mvpCurrentMatchedPoints;
        std::vector<MapPoint::Ptr> mvpLoopMapPoints;
//        std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
//        std::vector<KeyFrame*> mvpCurrentConnectedKFs;
//        std::vector<MapPoint*> mvpCurrentMatchedPoints;
//        std::vector<MapPoint*> mvpLoopMapPoints;
        cv::Mat mScw;

//    g2o::Sim3 mg2oScw;

        long unsigned int mLastLoopKFid;

        // Variables related to Global Bundle Adjustment
        bool mbRunningGBA;
        bool mbFinishedGBA;
        bool mbStopGBA;
        std::mutex mMutexGBA;
        std::thread* mpThreadGBA;

        // Fix scale in the stereo/RGB-D case
        bool mbFixScale;

        bool mnFullBAIdx;

        bool stop_require_;
        std::mutex mutex_stop_;
    };

};
#endif //SSVO_LOOPCLOSING_HPP
