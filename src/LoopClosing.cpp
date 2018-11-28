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

#include "LoopClosing.hpp"

//#include "LoopClosing.h"
//
//#include "Sim3Solver.h"
//
//#include "Converter.h"
//
//#include "Optimizer.h"
//
//#include "ORBmatcher.h"

//#include "Sim3Solver.h"

//#include "Converter.h"

#include "optimizer.hpp"

//#include "ORBmatcher.h"

#include<mutex>
#include<thread>
#include <include/LoopClosing.hpp>

using namespace std;

namespace ssvo
{

LoopClosing::LoopClosing(Map::Ptr pMap,DBoW3::Vocabulary *mpVocabulary_, DBoW3::Database *mpDatabase_, const bool bFixScale):
        mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
        mpVocabulary_(mpVocabulary_),mpDatabase_(mpDatabase_), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
        mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0),
        loop_closing_thread_(nullptr)
{
    mnCovisibilityConsistencyTh = 3;

    const int nlevel = Config::imageTopLevel() + 1;
    const int cols = Config::imageWidth();
    const int rows = Config::imageHeight();
    border_tl_.resize(nlevel);
    border_br_.resize(nlevel);

    for(int i = 0; i < nlevel; i++)
    {
        border_tl_[i].x = BRIEF::EDGE_THRESHOLD;
        border_tl_[i].y = BRIEF::EDGE_THRESHOLD;
        border_br_[i].x = cols/(1<<i) - BRIEF::EDGE_THRESHOLD;
        border_br_[i].y = rows/(1<<i) - BRIEF::EDGE_THRESHOLD;
    }
}

//void LoopClosing::SetTracker(FeatureTracker::Ptr pTracker)
//{
//    mpTracker=pTracker;
//}

//void LoopClosing::SetLocalMapper(LocalMapper::Ptr pLocalMapper)
//{
//    mpLocalMapper=pLocalMapper;
//}


void LoopClosing::startMainThread()
{
    if(loop_closing_thread_ == nullptr)
        loop_closing_thread_ = std::make_shared<std::thread>(std::bind(&LoopClosing::Run, this));
}

void LoopClosing::stopMainThread()
{
    setStop();
    if(loop_closing_thread_)
    {
        if(loop_closing_thread_->joinable())
            loop_closing_thread_->join();
        loop_closing_thread_.reset();
    }
}

void LoopClosing::setStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    stop_require_ = true;
}
bool LoopClosing::isRequiredStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    return stop_require_;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())
            {
                cout<<"-----------------------------DetectLoop Successfully------------------------------"<<endl;
                cv::waitKey(0);
                // Compute similarity transformation [sR|t]
                // In the stereo/RGBD case s=1
                if(ComputeSim3())
                {
                    // Perform loop fusion and pose graph optimization
                    CorrectLoop();
                }
            }
        }

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame::Ptr pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
//    if(pKF->id_!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}



    std::string to_binary(const cv::Mat &desp)
    {
        assert(desp.cols == 1 || desp.rows == 1);
        assert(desp.type() == CV_8UC1);
        const int size = desp.cols * desp.rows;
        const uchar* pt = desp.ptr<uchar>(0);
        std::string out;
        for(int i = 0; i < size; i++, pt++)
        {
            for(int j = 0; j < 8; ++j)
            {
                out += *pt & (0x01<<j) ? '1' : '0';
            }

            if(i < size-1)
                out += " ";
        }
        return out;
    }

bool LoopClosing::DetectLoop()
{

    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        //todo 这里需要考虑关键帧之间的关联关系不能动
        //mpCurrentKF->SetNotErase();
    }

    addToDatabase(mpCurrentKF);

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->id_<mLastLoopKFid+10)
    {
//        addToDatabase(mpCurrentKF);
//        mpCurrentKF->SetErase();
        cout<<"------------------------------------less keyframe-------------------------------------"<<endl;
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    //! 获取所有的共视帧
    const set<KeyFrame::Ptr> vpConnectedKeyFrames = mpCurrentKF->getConnectedKeyFrames(-1);
    cout<<"time :"<<std::fixed <<std::setprecision(6)<<mpCurrentKF->timestamp_<<endl;
    cout<<"KeyFrame "<<mpCurrentKF->id_<<" frame id "<<mpCurrentKF->frame_id_<<" has ConnectedKeyFrames size: "<<vpConnectedKeyFrames.size()<<endl;
    cout<<"KeyFrame "<<mpCurrentKF->id_<<" bowvec size: "<<mpCurrentKF->bow_vec_.size() <<": "<<mpCurrentKF->bow_vec_<<endl<<endl;

    //! 词袋向量bow_vec_ 在localmapping的时候计算
    const DBoW3::BowVector &CurrentBowVec = mpCurrentKF->bow_vec_;

    //! 计算共视帧中的词袋向量的最小得分，因为共视帧都是最近观测的，所以不能拿来做闭环检测，需要计算一个最小得分然后从数据库中找
    double minScore = 1;
    for(KeyFrame::Ptr pKF:vpConnectedKeyFrames)
    {
//        KeyFrame::Ptr pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW3::BowVector &BowVec = pKF->bow_vec_;
//        std::cout<< pKF->bow_vec_<<std::endl;

        double score = (*mpVocabulary_).score(CurrentBowVec, BowVec);

        cout<<"time :"<<std::fixed <<std::setprecision(6)<<pKF->timestamp_<<endl;
        cout<<"KeyFrame "<<pKF->id_<<" frame id "<<pKF->frame_id_<<" score: "<<score<<endl;
        cout<<"KeyFrame "<<pKF->id_<<" bowvec: "<<pKF->bow_vec_<<endl<<endl;

        if(score<minScore)
            minScore = score;
    }

    cout<<"minScore: "<<minScore<<endl;

    // Query the database imposing the minimum score
    //todo 候选帧的筛选，参考orb
    vector<KeyFrame::Ptr> vpCandidateKFs = DetectLoopCandidates(mpCurrentKF,minScore);

    cout<<"vpCandidateKFs size: "<<vpCandidateKFs.size()<<endl;

    //todo 策略调整
    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        cout<<"------------------------------------no vpCandidateKFs-------------------------------------"<<endl;
//        addToDatabase(mpCurrentKF);
        mvConsistentGroups.clear();
//        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame::Ptr pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame::Ptr> spCandidateGroup = pCandidateKF->getConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame::Ptr> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame::Ptr>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                //todo 修改
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    // todo
//    addToDatabase(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
//        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

//    mpCurrentKF->SetErase();
    return false;
}

bool LoopClosing::ComputeSim3()
{
    /*
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty())
            {
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true;
                    mpMatchedKF = pKF;
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }
*/
}

    void LoopClosing::CorrectLoop()
    {
        /*
        cout << "Loop detected!" << endl;

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        mpLocalMapper->RequestStop();

        // If a Global Bundle Adjustment is running, abort it
        if(isRunningGBA())
        {
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;

            mnFullBAIdx++;

            if(mpThreadGBA)
            {
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Ensure current keyframe is updated
        mpCurrentKF->UpdateConnections();

        // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        CorrectedSim3[mpCurrentKF]=mg2oScw;
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();


        {
            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
            {
                KeyFrame* pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();

                if(pKFi!=mpCurrentKF)
                {
                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                    cv::Mat tic = Tic.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                    g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                    //Pose corrected with the Sim3 of the loop closure
                    CorrectedSim3[pKFi]=g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                cv::Mat tiw = Tiw.rowRange(0,3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
                //Pose without correction
                NonCorrectedSim3[pKFi]=g2oSiw;
            }

            // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
            for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;
                g2o::Sim3 g2oCorrectedSiw = mit->second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

                vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
                for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
                {
                    MapPoint* pMPi = vpMPsi[iMP];
                    if(!pMPi)
                        continue;
                    if(pMPi->isBad())
                        continue;
                    if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateNormalAndDepth();
                }

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                eigt *=(1./s); //[R t/s;0 1]

                cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

                pKFi->SetPose(correctedTiw);

                // Make sure connections are updated
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
            {
                if(mvpCurrentMatchedPoints[i])
                {
                    MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                    MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                    if(pCurMP)
                        pCurMP->Replace(pLoopMP);
                    else
                    {
                        mpCurrentKF->AddMapPoint(pLoopMP,i);
                        pLoopMP->AddObservation(mpCurrentKF,i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }

        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        SearchAndFuse(CorrectedSim3);


        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        map<KeyFrame*, set<KeyFrame*> > LoopConnections;

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            pKFi->UpdateConnections();
            LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
            for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev);
            }
            for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // Optimize graph
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

        mpMap->InformNewBigChange();

        // Add loop edge
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

        // Loop closed. Release Local Mapping.
        mpLocalMapper->Release();

        mLastLoopKFid = mpCurrentKF->mnId;
         */
    }

//void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
//{
//    ORBmatcher matcher(0.8);
//
//    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
//    {
//        KeyFrame* pKF = mit->first;
//
//        g2o::Sim3 g2oScw = mit->second;
//        cv::Mat cvScw = Converter::toCvMat(g2oScw);
//
//        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
//        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);
//
//        // Get Map Mutex
//        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
//        const int nLP = mvpLoopMapPoints.size();
//        for(int i=0; i<nLP;i++)
//        {
//            MapPoint* pRep = vpReplacePoints[i];
//            if(pRep)
//            {
//                pRep->Replace(mvpLoopMapPoints[i]);
//            }
//        }
//    }
//}


    void LoopClosing::RequestReset()
    {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while(1)
        {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if(!mbResetRequested)
                    break;
            }
            usleep(5000);
        }
    }

    void LoopClosing::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            mlpLoopKeyFrameQueue.clear();
            mLastLoopKFid=0;
            mbResetRequested=false;
        }
    }

    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        /*
        cout << "Starting Global Bundle Adjustment" << endl;

        int idx =  mnFullBAIdx;
        Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that there might be new keyframes
        // not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        {
            unique_lock<mutex> lock(mMutexGBA);
            if(idx!=mnFullBAIdx)
                return;

            if(!mbStopGBA)
            {
                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;
                mpLocalMapper->RequestStop();
                // Wait until Local Mapping has effectively stopped

                while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
                {
                    usleep(1000);
                }

                // Get Map Mutex
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

                while(!lpKFtoCheck.empty())
                {
                    KeyFrame* pKF = lpKFtoCheck.front();
                    const set<KeyFrame*> sChilds = pKF->GetChilds();
                    cv::Mat Twc = pKF->GetPoseInverse();
                    for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                    {
                        KeyFrame* pChild = *sit;
                        if(pChild->mnBAGlobalForKF!=nLoopKF)
                        {
                            cv::Mat Tchildc = pChild->GetPose()*Twc;
                            pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                            pChild->mnBAGlobalForKF=nLoopKF;

                        }
                        lpKFtoCheck.push_back(pChild);
                    }

                    pKF->mTcwBefGBA = pKF->GetPose();
                    pKF->SetPose(pKF->mTcwGBA);
                    lpKFtoCheck.pop_front();
                }

                // Correct MapPoints
                const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

                for(size_t i=0; i<vpMPs.size(); i++)
                {
                    MapPoint* pMP = vpMPs[i];

                    if(pMP->isBad())
                        continue;

                    if(pMP->mnBAGlobalForKF==nLoopKF)
                    {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // Update according to the correction of its reference keyframe
                        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                        if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                            continue;

                        // Map to non-corrected camera
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                        cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                        // Backproject using corrected camera
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                        cv::Mat twc = Twc.rowRange(0,3).col(3);

                        pMP->SetWorldPos(Rwc*Xc+twc);
                    }
                }

                mpMap->InformNewBigChange();

                mpLocalMapper->Release();

                cout << "Map updated!" << endl;
            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
         */
    }

    void LoopClosing::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool LoopClosing::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void LoopClosing::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool LoopClosing::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }


    std::vector<KeyFrame::Ptr> LoopClosing::DetectLoopCandidates(KeyFrame::Ptr pKF, double minScore)
    {
        set<KeyFrame::Ptr> spConnectedKeyFrames = pKF->getConnectedKeyFrames(-1);
        list<KeyFrame::Ptr> lKFsSharingWords;

        // Search all keyframes that share a word with current keyframes
        // Discard keyframes connected to the query keyframe
        {
//            unique_lock<mutex> lock(mMutex);

            // 遍历该关键帧的单词
            for(DBoW3::BowVector::const_iterator vit=pKF->bow_vec_.begin(), vend=pKF->bow_vec_.end(); vit != vend; vit++)
            {
                std::vector<DBoW3::EntryId> EntryIds = mpDatabase_->getEntryIdFromWord(vit->first);

                list<KeyFrame::Ptr> lKFs;
                for(auto entryId:EntryIds)
                {
                    lKFs.push_back(KeyFrames[entryId]);
                }

                for(list<KeyFrame::Ptr>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
                {
                    KeyFrame::Ptr pKFi=*lit;
                    if(pKFi->mnLoopQuery!=pKF->id_)
                    {
                        pKFi->mnLoopWords=0;
                        if(!spConnectedKeyFrames.count(pKFi))
                        {
                            pKFi->mnLoopQuery=pKF->id_;
                            lKFsSharingWords.push_back(pKFi);
                        }
                    }
                    pKFi->mnLoopWords++;
                }
            }
        }

        if(lKFsSharingWords.empty())
            return vector<KeyFrame::Ptr>();

        list<pair<float,KeyFrame::Ptr> > lScoreAndMatch;

        // Only compare against those keyframes that share enough words
        int maxCommonWords=0;
        for(list<KeyFrame::Ptr>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
        {
            if((*lit)->mnLoopWords>maxCommonWords)
                maxCommonWords=(*lit)->mnLoopWords;
        }

        int minCommonWords = maxCommonWords*0.8f;

        int nscores=0;

        // Compute similarity score. Retain the matches whose score is higher than minScore
        for(list<KeyFrame::Ptr>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
        {
            KeyFrame::Ptr pKFi = *lit;

            if(pKFi->mnLoopWords>minCommonWords)
            {
                nscores++;

                float si = mpVocabulary_->score(pKF->bow_vec_,pKFi->bow_vec_);

                pKFi->mLoopScore = si;
                if(si>=minScore)
                    lScoreAndMatch.push_back(make_pair(si,pKFi));
            }
        }

        if(lScoreAndMatch.empty())
            return vector<KeyFrame::Ptr>();

        list<pair<float,KeyFrame::Ptr> > lAccScoreAndMatch;
        float bestAccScore = minScore;

        // Lets now accumulate score by covisibility
        for(list<pair<float,KeyFrame::Ptr> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
        {
            KeyFrame::Ptr pKFi = it->second;
            std::set<KeyFrame::Ptr> vpNeighs_set = pKFi->getConnectedKeyFrames(10);
            std::vector<KeyFrame::Ptr> vpNeighs;
            vpNeighs.assign(vpNeighs_set.begin(),vpNeighs_set.end());

            float bestScore = it->first;
            float accScore = it->first;
            KeyFrame::Ptr pBestKF = pKFi;
            for(vector<KeyFrame::Ptr>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
            {
                KeyFrame::Ptr pKF2 = *vit;
                if(pKF2->mnLoopQuery==pKF->id_ && pKF2->mnLoopWords>minCommonWords)
                {
                    accScore+=pKF2->mLoopScore;
                    if(pKF2->mLoopScore>bestScore)
                    {
                        pBestKF=pKF2;
                        bestScore = pKF2->mLoopScore;
                    }
                }
            }

            lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
            if(accScore>bestAccScore)
                bestAccScore=accScore;
        }

        // Return all those keyframes with a score higher than 0.75*bestScore
        float minScoreToRetain = 0.75f*bestAccScore;

        set<KeyFrame::Ptr> spAlreadyAddedKF;
        vector<KeyFrame::Ptr> vpLoopCandidates;
        vpLoopCandidates.reserve(lAccScoreAndMatch.size());

        for(list<pair<float,KeyFrame::Ptr> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
        {
            if(it->first>minScoreToRetain)
            {
                KeyFrame::Ptr pKFi = it->second;
                if(!spAlreadyAddedKF.count(pKFi))
                {
                    vpLoopCandidates.push_back(pKFi);
                    spAlreadyAddedKF.insert(pKFi);
                }
            }
        }

        return vpLoopCandidates;
    }

    template <>
    inline size_t Grid<Feature::Ptr>::getIndex(const Feature::Ptr &element)
    {
        const Vector2d &px = element->px_;
        return static_cast<size_t>(px[1]/grid_size_)*grid_n_cols_
               + static_cast<size_t>(px[0]/grid_size_);
    }

    void LoopClosing::addToDatabase(const KeyFrame::Ptr &keyframe)
    {
        KeyFrames.push_back(keyframe);
        LOG_ASSERT(KeyFrames.size() == (keyframe->id_ + 1))<<"KeyFrames size() is wrong, please check!"<<endl;

        keyframe->getFeatures(keyframe->dbow_fts_);
        const int cols = keyframe->cam_->width();
        const int rows = keyframe->cam_->height();
        const int N = Config::minCornersPerKeyFrame();;

        Grid<Feature::Ptr> grid(cols, rows, 30);

        for(const Feature::Ptr &ft : keyframe->dbow_fts_)
        {
            if(ft->px_[0] <= border_tl_[ft->level_].x ||
               ft->px_[1] <= border_tl_[ft->level_].y ||
               ft->px_[0] >= border_br_[ft->level_].x ||
               ft->px_[1] >= border_br_[ft->level_].y)
                continue;
            grid.insert(ft);
        }

        resetGridAdaptive(grid, N, 20);
        grid.sort();
        grid.getBestElement(keyframe->dbow_fts_);

        std::vector<cv::KeyPoint> kps;
        for(const Feature::Ptr &ft : keyframe->dbow_fts_)
            kps.emplace_back(cv::KeyPoint(ft->px_[0], ft->px_[1], 31, -1, 0, ft->level_));

        BRIEF::Ptr brief = BRIEF::create();

        brief->compute(keyframe->images(), kps, keyframe->descriptors_); //3

//        cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 2, 1);
//        std::vector<cv::KeyPoint> kps;
//        orb->detect(keyframe->images()[0], kps);
//        orb->compute(keyframe->images()[0], kps, keyframe->descriptors_);

        /* 检查图像金字塔是不是正确的
        cv::imshow("0-0",keyframe->getImage(0));
        cv::imshow("0-1",keyframe->getImage(1));
        cv::imshow("0-2",keyframe->getImage(2));
        cv::imshow("0-3",keyframe->getImage(3));

        cv::imshow("1-0",imgPyr[0]);
        cv::imshow("1-1",imgPyr[1]);
        cv::imshow("1-2",imgPyr[2]);
        cv::imshow("1-3",imgPyr[3]);

        for(int i = 0 ;i<4;i++)
        {
            LOG_ASSERT(imgPyr[i].rows == keyframe->getImage(i).rows)<<"wrong size"<<endl;
            LOG_ASSERT(imgPyr[i].cols == keyframe->getImage(i).cols)<<"wrong size"<<endl;
        }
        cv::imshow("r-0",imgPyr[0]-keyframe->getImage(0));
        cv::imshow("r-1",imgPyr[1]-keyframe->getImage(1));
        cv::imshow("r-2",imgPyr[2]-keyframe->getImage(2));
        cv::imshow("r-3",imgPyr[3]-keyframe->getImage(3));
        cv::waitKey(0);
         */

        /*
//        brief->compute(imgPyr, kps, desps1);

//        cv::waitKey(0);

//        for(int j = 0; j < kps.size(); ++j)
//        {
//            std::cout << "orb  2: " << desps1.row(j) << std::endl;
//            std::cout << "ssvo 3: " << keyframe->descriptors_.row(j) << std::endl;
//
//        }
//        std::abort();

//        cout<<"keyframe->descriptors_:"<<endl<<keyframe->descriptors_<<endl;

//        keyframe->computeBoW(mpVocabulary_);
//        std::cout<< keyframe->bow_vec_<<std::endl;

//        std::cout << "=========" << std::endl;
//        std::cout << "Voc and DB info:" << std::endl;
//        std::cout << *mpVocabulary_ << std::endl;
//        std::cout << *mpDatabase_ << std::endl;
         */

        keyframe->dbow_Id_ = (*mpDatabase_).add(keyframe->descriptors_, &keyframe->bow_vec_, &keyframe->feat_vec_);

        //! 每一个关键帧都添加到database中
        LOG_ASSERT(keyframe->dbow_Id_ == keyframe->id_) << "DBoW Id(" << keyframe->dbow_Id_ << ") is not match the keyframe's Id(" << keyframe->id_ << ")!";

    }

}
