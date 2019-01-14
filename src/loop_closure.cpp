//
// Created by jh on 18-11-29.
//

#include "loop_closure.hpp"
#include "optimizer.hpp"

namespace ssvo{

const int TH_HIGH = 100;
const int TH_LOW = 50;
const int HISTO_LENGTH = 30;
const float fNNratio =0.75;
const bool CheckOrientation = true;

void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

cv::Mat computeDistinctiveDescriptors(std::vector<cv::Mat> &descriptors)
{
    const size_t N = descriptors.size();

    std::vector<std::vector<double > > Distances;
    Distances.resize(N, std::vector<double >(N, 0));
    for (size_t i = 0; i<N; i++)
    {
        Distances[i][i]=0;
        for(size_t j = i+1; j < N;j++)
        {
            double distij = DBoW3::DescManip::distance(descriptors[i],descriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        // 第i个描述子到其它所有所有描述子之间的距离
        //vector<int> vDists(Distances[i],Distances[i]+N);
        std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
        std::sort(vDists.begin(), vDists.end());

        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    return descriptors[BestIdx];

}

cv::Mat showMatch(const cv::Mat& img1,const cv::Mat& img2,const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2)
{
    cv::Mat img_show;
    std::vector<cv::Point2f> points1_copy,points2_copy;
    points1_copy.assign(points1.begin(),points1.end());
    points2_copy.assign(points2.begin(),points2.end());
    for(auto iter2=points2_copy.begin();iter2!=points2_copy.end();)
    {
        iter2->x+=img1.cols;
        iter2++;
    }
    cv::RNG rng(time(0));
    hconcat(img1,img2,img_show);
    std::vector<cv::Point2f>::iterator iter1,iter2;
    for(iter1=points1_copy.begin(),iter2=points2_copy.begin();iter1!=points1_copy.end();iter1++,iter2++)
    {
        line(img_show,*iter1,*iter2,cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255)),1);
        circle(img_show,*iter1,1,0,2);
        circle(img_show,*iter2,1,0,2);
    }
    return img_show;
}

cv::Mat showFeatures(const cv::Mat& src, const std::vector<Feature::Ptr> &features)
{
    cv::Mat img = src.clone();

    for(auto item:features)
    {
        cv::circle(img,cv::Point2f(item->px_[0],item->px_[1]),1,0,2);
    }

    return img;
}

LoopClosure::LoopClosure(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database_):
        vocabulary_(vocabulary), database_(database_),LastLoopKFid_(0),
        ifFinished(true),RunningGBA_(false), FinishedGBA_(true), StopGBA_(false), thread_GBA_(NULL),
        FullBAIdx_(0),update_finish_(false),loop_time_(0)
{
    sim3_cw = Sophus::Sim3d();
    mnCovisibilityConsistencyTh = 3;
}
void LoopClosure::startMainThread()
{
    if(loop_closure_thread_ == nullptr)
    {
        loop_closure_thread_ = std::make_shared<std::thread>(std::bind(&LoopClosure::run,this));
    }
}

void LoopClosure::stopMainThread()
{
    if(loop_closure_thread_)
    {
        if(loop_closure_thread_->joinable())
            loop_closure_thread_->join();
        loop_closure_thread_.reset();
    }
}

void LoopClosure::insertKeyFrame(KeyFrame::Ptr kf)
{
    std::unique_lock<std::mutex> lock(mutex_keyFramesList_);
    keyFramesList_.push_back(kf);
    cond_process_.notify_one();

    KeyFrames.push_back(kf);
}

void LoopClosure::run()
{
    ifFinished = false;
    while(1)
    {
        if(CheckNewKeyFrames())
        {
            if(DetectLoop())
            {
                if(ComputeSim3())
                {
                    CorrectLoop();
                }
            }
        }
    }
    ifFinished = true;
}

bool LoopClosure::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mutex_keyFramesList_);
    cond_process_.wait_for(lock, std::chrono::microseconds(5));
    return (!keyFramesList_.empty());
}

/**
 * @brief 检测闭环
 *      1. 计算与当前帧的共视关键帧的闭环向量的最小得分，作为筛选闭环帧的阈值
 *      2. 检测闭环候选帧
 *          A. 从database中查找有共同单词的关键帧
 *          B. 闭环关键帧需要满足共同单词数和得分
 *          C. 闭环关键帧的共视帧也需要有良好的闭环因素，所以要检查闭环关键帧的共视帧
 *      3. 检测闭环关键帧的连续性（时间上）
 * @return 是否有满足条件的闭环候选帧
 * @attention 没有用到mappoint或feature 只是用计算过的词袋向量和特征向量
 */
bool LoopClosure::DetectLoop()
{
    {
        std::unique_lock<std::mutex> lock(mutex_keyFramesList_);
        curKeyFrame_ = keyFramesList_.front();
        keyFramesList_.pop_front();
        curKeyFrame_->setNotErase();
    }

    if(curKeyFrame_->id_<LastLoopKFid_+10)
    {
        curKeyFrame_->setErase();
        LOG(WARNING) << "[LoopClosure] Less than 10 keyframe to last loop or too little keyframe in the map.";
        return false;
    }

    //! step 1
    const std::set<KeyFrame::Ptr> ConnectedKeyFrames = curKeyFrame_->getConnectedKeyFrames(-1);

    const DBoW3::BowVector &currentBowVec = curKeyFrame_->bow_vec_;

    bool test_shareword = false;

    if(test_shareword)
    {
        DBoW3::QueryResults results;
        database_->query(curKeyFrame_->bow_vec_,results,-1,-1);
        for(const DBoW3::Result ret : results)
        {
            if(ret.Id != curKeyFrame_->id_)
            {
                DBoW3::BowVector::iterator bv1_iter = curKeyFrame_->bow_vec_.begin();
                DBoW3::BowVector::iterator bv1_end = curKeyFrame_->bow_vec_.end();

                LOG_ASSERT(KeyFrames[ret.Id]->id_ == ret.Id);
                DBoW3::BowVector::iterator bv2_iter = KeyFrames[ret.Id]->bow_vec_.begin();
                DBoW3::BowVector::iterator bv2_end = KeyFrames[ret.Id]->bow_vec_.end();

                int count = 0;
                while(bv1_iter != bv1_end && bv2_iter != bv2_end)
                {

                    if(bv1_iter->first == bv2_iter->first)
                    {
                        count++;
                        bv2_iter++;
                        bv1_iter++;
                    }
                    else if(bv1_iter->first > bv2_iter->first)
                        bv2_iter = KeyFrames[ret.Id]->bow_vec_.lower_bound(bv1_iter->first);
                    else
                        bv1_iter = curKeyFrame_->bow_vec_.lower_bound(bv2_iter->first);
                }

                std::cout << "share words <" << curKeyFrame_->id_ << ", " << ret.Id << "> : " << count << std::endl;
            }
        }
    }

    //! step 2 Compute reference BoW similarity score
    double minScore = 1;
    double maxScore = 0;
    for(KeyFrame::Ptr pKF:ConnectedKeyFrames)
    {
        if(pKF->isBad())
            continue;
        const DBoW3::BowVector &BowVec = pKF->bow_vec_;

        int same_word = 0;
        for (auto wordId:currentBowVec)
        {
            if(BowVec.count(wordId.first))
            {
                same_word ++;
            }
        }
        double score = (*vocabulary_).score(currentBowVec, BowVec);

        bool check_score = false;
        if(check_score)
        {
            std::cout<<"time :"<<std::fixed <<std::setprecision(6)<<pKF->timestamp_<<std::endl;
            std::cout<<"KeyFrame "<<pKF->id_<<" frame id "<<pKF->frame_id_<<" score:["<<score<<"] have ["<<same_word<<"] word."<<std::endl;
            std::cout<<"======================================"<<std::endl;
        }

        if(score<minScore && score!=0)
            minScore = score;
        if(score>maxScore)
            maxScore = score;

    }

    LOG(WARNING) << "[LoopClosure] <minScore,maxScore>: <"<<minScore<<","<<maxScore<<">";

    //! step 3 Query the database imposing the minimum score
    std::vector<KeyFrame::Ptr> vpCandidateKFs = DetectLoopCandidates(minScore);

    LOG(WARNING) << "[LoopClosure] CandidateKFs number after DetectLoopCandidates: "<<vpCandidateKFs.size();

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        LOG(WARNING) << "[LoopClosure] No vpCandidateKFs";
        mvConsistentGroups.clear();
        curKeyFrame_->setErase();
        return false;
    }
    else
    {
        LOG(WARNING) << "[LoopClosure] Loop detect successfully! curKeyFrame<id,frameid>: <"<<curKeyFrame_->id_<<","<<curKeyFrame_->frame_id_<<">";

        bool save_loop = false;
        if(save_loop)
        {
            std::string filename_;
            filename_ = std::to_string(loop_time_)+"_cur_" +std::to_string(curKeyFrame_->id_)+ ".png";
            cv::imwrite(filename_, curKeyFrame_->getImage(0));
            for(auto kf:vpCandidateKFs)
            {
                std::string filename;
                filename =std::to_string(loop_time_)+"_loop_" +std::to_string(kf->id_)+ ".png";
                cv::imwrite(filename, kf->getImage(0));
            }
        }
        loop_time_++;
    }

    //! step 4
    mvpEnoughConsistentCandidates.clear();

    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame::Ptr pCandidateKF = vpCandidateKFs[i];

        std::set<KeyFrame::Ptr> spCandidateGroup = pCandidateKF->getConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            std::set<KeyFrame::Ptr> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(std::set<KeyFrame::Ptr>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
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
                if(nCurrentConsistency >=  mnCovisibilityConsistencyTh && !bEnoughConsistent)
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


    if(mvpEnoughConsistentCandidates.empty())
    {
        curKeyFrame_->setErase();
        return false;
    }
    else
    {
        return true;
    }
}

bool LoopClosure::ComputeSim3()
{

    LOG(WARNING) << "[LoopClosure] The loop keyframe fit all condition and then we ComputeSim3!!!";
    const int InitialCandidates = mvpEnoughConsistentCandidates.size();
    LOG(WARNING) << "[LoopClosure] satisfy the score / common words / consistent kf number: "<< InitialCandidates;

    std::vector<Sim3Solver* > Sim3Solvers;
    Sim3Solvers.resize(InitialCandidates);

    std::vector<std::vector<MapPoint::Ptr>> MapPointMatches_globa;
    MapPointMatches_globa.resize(InitialCandidates);

    std::vector<bool > Discarded;
    Discarded.resize(InitialCandidates);

    std::vector<Feature::Ptr> features_1 = curKeyFrame_->featuresInBow;

    //candidates with enough matches
    int Candidates=0;

    //记录每一帧闭环帧与当前帧之间的匹配关系[第i帧回环帧][-1 || 与当前帧特征点匹配的特征点的索引]
    std::vector<std::vector<int >> bestidx;
    bestidx.resize(InitialCandidates);

    for (int i = 0; i < InitialCandidates; ++i)
    {
        // 步骤1：从筛选的闭环候选帧中取出一帧关键帧pKF
        KeyFrame::Ptr loopKeyFrame = mvpEnoughConsistentCandidates[i];

        loopKeyFrame->setNotErase();

        if(loopKeyFrame->isBad())
        {
            Discarded[i] = true;
            continue;
        }

        // 步骤2：将当前帧mpCurrentKF与闭环候选关键帧pKF匹配
        // 通过bow加速得到mpCurrentKF与pKF之间的匹配特征点
        // bestidx[i] 记录第i帧候选帧中与当前帧mpt（按照顺序）的匹配的mpt的idx
        int nmatches = SearchByBoW(loopKeyFrame,MapPointMatches_globa[i],bestidx[i]);

        LOG(WARNING) << "[LoopClosure]  Matches SearchByBoW: "<<nmatches;

        bool test_SearchByBoW = false;

        if(test_SearchByBoW)
        {
            cv::Mat img0 = curKeyFrame_->getImage(0).clone();
            cv::Mat img1 = loopKeyFrame->getImage(0).clone();

            std::vector<cv::Point2f> points1,points2;

            for (int j = 0; j < MapPointMatches_globa[i].size(); ++j)
            {
                if(MapPointMatches_globa[i][j])
                {
                    LOG_ASSERT(bestidx[i][j] != -1);
                    points1.push_back(cv::Point2f(curKeyFrame_->featuresInBow[j]->px_[0],curKeyFrame_->featuresInBow[j]->px_[1]));
                    points2.push_back(cv::Point2f(loopKeyFrame->featuresInBow[bestidx[i][j]]->px_[0],loopKeyFrame->featuresInBow[bestidx[i][j]]->px_[1]));
                }
            }

            cv::Mat image_show = showMatch(img0,img1,points1,points2);
            cv::imshow("SearchByBoW",image_show);
            cv::imwrite("SearchByBoW.png",image_show);
        }

        if(nmatches < 20)
        {
            LOG(WARNING) << "[LoopClosure] Too little matches SearchByBoW!"<<std::endl;
            Discarded[i] = true;
            continue;
        }
        else
        {
            std::vector<Feature::Ptr> fts_1_match,fts_2_match;
            std::vector<Feature::Ptr> features_2 = loopKeyFrame->featuresInBow;
            LOG_ASSERT(features_2.size() == loopKeyFrame->mptId_des.size());
            for (int j = 0; j < MapPointMatches_globa[i].size(); ++j)
            {
                if(bestidx[i][j] == -1)
                    continue;
                LOG_ASSERT(bestidx[i][j] <= loopKeyFrame->mptId_des.size())<<bestidx[i][j]<<"------"<<loopKeyFrame->mptId_des.size()<<std::endl;
                fts_1_match.emplace_back(features_1[j]);
                fts_2_match.emplace_back(features_2[bestidx[i][j]]);
            }

            Sim3Solver* pSolver = new Sim3Solver(curKeyFrame_->Tcw(),loopKeyFrame->Tcw(),fts_1_match,fts_2_match,false);
            int liner_th = fts_1_match.size()*0.1>30?fts_1_match.size()*0.1:30;
            pSolver->SetRansacParameters(0.99,liner_th,300);
            Sim3Solvers[i] = pSolver;
        }
        Candidates++;
    }

    bool Match = false;

    while(Candidates>0 && !Match)
    {
        for(int i = 0; i < InitialCandidates; ++i)
        {
            if(Discarded[i])
                continue;
            KeyFrame::Ptr loopKeyFrame = mvpEnoughConsistentCandidates[i];

            if((curKeyFrame_->frame_id_-loopKeyFrame->frame_id_)<1000)
            {
                Discarded[i] = true;
                continue;
            }
            bool success = false;
            Sim3Solver* pSolver = Sim3Solvers[i];
            success = pSolver->runRANSAC(50);

            if(pSolver->noMore_)
            {
                Discarded[i]=true;
                Candidates--;
            }

            if(success)
            {
                std::vector<MapPoint::Ptr > MapPointMatches_local(MapPointMatches_globa[i].size(), static_cast<MapPoint::Ptr>(NULL));

                std::unordered_map<uint64_t,uint64_t > matches_1_2;
                std::unordered_map<uint64_t,uint64_t > matches_2_1;
                // [sR t;0 1]
                Matrix3d R;
                Vector3d t;
                double s;

                // inlier 是 fts_1_match，fts_2_match 的索引，而不是整个MapPointMatches_globa[i]
                std::vector<bool > inliers;
                int good_match = 0;

                pSolver->getEstimateSim3(R,t,s,inliers);

                std::cout<<"==================EstimateSim3====================="<<std::endl;
                std::cout<<" R ======== <<"<<std::endl<< R <<std::endl;
                std::cout<<" t ======== <<"<<std::endl<< t.transpose() <<std::endl;
                std::cout<<" s ======== <<"<<std::endl<< s <<std::endl;
                std::cout<<" Good Match size ======== <<"<<std::endl<< inliers.size() <<std::endl;

                std::vector<int > bestidx_local; //表示当前帧第[ ]个特征点有匹配点
                //【0,5,6,9,12,15,48】表示当前帧的第0,5，……，48有特征点匹配
                for (int k = 0; k < bestidx[i].size(); ++k) {
                    if(bestidx[i][k]!=-1)
                        bestidx_local.push_back(k);
                }

                for(size_t j=0, jend = bestidx_local.size(); j<jend; j++)
                {
                    if(MapPointMatches_globa[i][bestidx_local[j]])
                    {
                        LOG_ASSERT(bestidx[i][bestidx_local[j]] != -1);

                        if(j>inliers.size())
                            break;
                        if(inliers[j])
                        {
                            MapPointMatches_local[bestidx_local[j]] = MapPointMatches_globa[i][bestidx_local[j]];

                            LOG_ASSERT(features_1[bestidx_local[j]]->mpt_!= nullptr);
                            LOG_ASSERT(MapPointMatches_globa[i][bestidx_local[j]]!= nullptr);

                            matches_1_2.insert(std::make_pair(features_1[bestidx_local[j]]->mpt_->id_,MapPointMatches_globa[i][bestidx_local[j]]->id_));
                            matches_2_1.insert(std::make_pair(MapPointMatches_globa[i][bestidx_local[j]]->id_,features_1[bestidx_local[j]]->mpt_->id_));
                            good_match++;
                        }
                    }

                }

                int newFound = SearchBySim3(loopKeyFrame,matches_1_2,matches_2_1,s,R,t,7.5,MapPointMatches_local,bestidx[i]);

                LOG(WARNING) << "[LoopClosure] SearchBySim3 creat new matches number: "<< newFound;

                if((good_match+newFound)<30)
                {
                    LOG(WARNING) << "[LoopClosure] Too little matches after SearchBySim3!!!";
                    continue;
                }

                bool test_SearchBySim3 = false;

                if(test_SearchBySim3)
                {
                    cv::Mat img0 = curKeyFrame_->getImage(0).clone();
                    cv::Mat img1 = loopKeyFrame->getImage(0).clone();

                    std::vector<cv::Point2f> points1,points2;

                    for (int j = 0; j < bestidx[i].size(); ++j)
                    {
                        if( bestidx[i][j]!=-1)
                        {
                            LOG_ASSERT(bestidx[i][j]!=-1);
                            points1.push_back(cv::Point2f(curKeyFrame_->featuresInBow[j]->px_[0],curKeyFrame_->featuresInBow[j]->px_[1]));
                            points2.push_back(cv::Point2f(loopKeyFrame->featuresInBow[bestidx[i][j]]->px_[0],loopKeyFrame->featuresInBow[bestidx[i][j]]->px_[1]));
                        }
                    }

                    cv::Mat image_show = showMatch(img0,img1,points1,points2);
                    cv::imwrite("SearchBySim3.png",image_show);
                }
//                Matrix4d temp;
//                temp.topLeftCorner(3,3) = s * R;
//                temp.topRightCorner(3,1) = t;
//                Sophus::Sim3d sim3_lkf2cur(temp);

                Matrix3d sR_ = s * R;
                Eigen::Quaterniond q_sr(sR_);
                Sophus::Sim3d sim3_lkf2cur(q_sr,t);

                int nInliers = 0;

                nInliers = Optimizer::optimizeSim3(curKeyFrame_, loopKeyFrame, MapPointMatches_local, sim3_lkf2cur, 10, true);// 卡方chi2检验阈值

                if(nInliers >= 20)
                {
                    std::cout<<"==================OptimizerSim3 Successfully!!!====================="<<std::endl;
                    std::cout<<" R ======== <<"<<std::endl<< sim3_lkf2cur.rotationMatrix() <<std::endl;
                    std::cout<<" t ======== <<"<<std::endl<< sim3_lkf2cur.translation().transpose() <<std::endl;
                    std::cout<<" s ======== <<"<<std::endl<< sim3_lkf2cur.scale() <<std::endl;
                    std::cout<<"optimizeSim3 nInliers: "<<nInliers<<std::endl;

                    Match = true;
                    MatchedKeyFrame_ = loopKeyFrame;

                    Matrix4d temp_w2lkf;
                    temp_w2lkf.topLeftCorner(3,3) = loopKeyFrame->Tcw().rotationMatrix();
                    temp_w2lkf.topRightCorner(3,1) = loopKeyFrame->Tcw().translation();
                    Sophus::Sim3d sim3_w2lkf(temp_w2lkf);

                    sim3_cw = sim3_lkf2cur * sim3_w2lkf;
                    T_cw = SE3d(sim3_cw.scale()*sim3_cw.rotationMatrix(),sim3_cw.translation());
                    CurrentMatchedPoints = MapPointMatches_local;

                    break;
                }
                else
                {
                    LOG(WARNING) << "[LoopClosure] No enough inliers after optimizeSim3!";
                }
            }
        }
    }

    if(!Match)
    {
        for(int i=0; i<InitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->setErase();
        curKeyFrame_->setErase();
        return false;
    }

    std::set<KeyFrame::Ptr> LoopConnectedKFs = MatchedKeyFrame_->getConnectedKeyFrames();
    LoopConnectedKFs.insert(MatchedKeyFrame_);
    LoopMapPoints.clear();
    for(std::set<KeyFrame::Ptr>::iterator vit = LoopConnectedKFs.begin(); vit != LoopConnectedKFs.end(); vit++)
    {
        KeyFrame::Ptr pKF = *vit;
        std::vector<MapPoint::Ptr> mpts = pKF->mapPointsInBow;
        for(size_t i = 0; i<mpts.size(); i++)
        {
            MapPoint::Ptr pMP = mpts[i];

            if(pMP->isBad())
                continue;
            if(pMP->loop_id_ != curKeyFrame_->id_)
            {
                LoopMapPoints.push_back(pMP);
                pMP->loop_id_=curKeyFrame_->id_;
            }

        }
    }

    int nTotalMatches = 0;
    for(size_t i = 0; i < CurrentMatchedPoints.size(); i++)
    {
        if(CurrentMatchedPoints[i])
            nTotalMatches++;
    }
    int foundByProjection = SearchByProjection(10);
    LOG(WARNING) << "[LoopClosure] New matches SearchByProjection "<<foundByProjection;

    int nTotalMatches1 = 0;
    for(size_t i = 0; i < CurrentMatchedPoints.size(); i++)
    {
        if(CurrentMatchedPoints[i])
            nTotalMatches1++;
    }

    LOG(WARNING) << "[LoopClosure] CurrentKF features size===>"<<curKeyFrame_->mapPointsInBow.size();
    LOG(WARNING) << "[LoopClosure] Total Matches After SearchBySim3 and SearchByProjection===>"<<nTotalMatches1;


    if(nTotalMatches1 >= 40)
    {
        for(int i=0; i<InitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i] != MatchedKeyFrame_)
                mvpEnoughConsistentCandidates[i]->setErase();
        return true;
    }
    else
    {
        for(int i=0; i<InitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->setErase();
        curKeyFrame_->setErase();
        return false;
    }
}

/**
 * @brief 校正闭环
 * 1. 通过求解的Sim3以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的MapPoints的位置（相连关键帧---当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的MapPoints和与当前帧相连的关键帧的点进行匹配（相连关键帧+当前帧---闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 */
void LoopClosure::CorrectLoop()
{
    LOG(WARNING) << "[LoopClosure] Loop detected, Begin to CorrectLoop!!!";
    LOG(WARNING) << "[LoopClosure] <Loop,cur>:<"<<MatchedKeyFrame_->id_<<"("<<MatchedKeyFrame_->frame_id_<<"),"<<curKeyFrame_->id_<<"("<<curKeyFrame_->frame_id_<<")>";

    //! 1.保存未校正的轨迹
    std::vector<KeyFrame::Ptr > all_kfs = local_mapper_->map_->getAllKeyFrames();
    std::sort(all_kfs.begin(),all_kfs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});

    bool traj_beforeEss = true;
    if(traj_beforeEss)
    {
        std::string trajBeforeEss = "traj_beforeEss.txt";
        std::ofstream f_trajBeforeEss;
        f_trajBeforeEss.open(trajBeforeEss.c_str());
        f_trajBeforeEss << std::fixed;

        for(KeyFrame::Ptr kf:all_kfs)
        {
            Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();

            f_trajBeforeEss << std::setprecision(6) << kf->timestamp_ << " "
                            << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        f_trajBeforeEss.close();
        std::cout<<"traj_beforeEss saved!"<<std::endl;
    }

    local_mapper_->setStop();
    LOG(WARNING) << "[LoopClosure] local_mapper_ require stop!";

    if(isRunningGBA())
    {
        // 如果正在进行globalBA的话。如果优化过程已经完成了，正在更新位姿，就等位姿更新完成，如果优化还没有结束，就直接结束globaBA的线程，将内存资源回收
        std::unique_lock<std::mutex> lock(mutex_GBA_);

        //TODO 要把这个flag加到globa BA中，能够随时停止迭代
        StopGBA_ = true;

        FullBAIdx_++;

        if(thread_GBA_)
        {
            thread_GBA_->detach();
            delete thread_GBA_;
        }
        LOG(WARNING) << "[LoopClosure] RunningGBA stop!";
    }

    //todo 等到localmapping进程结束，地图中的特征点和关键帧都不再变化(就可以不考虑你深度滤波线程了)
    while(!local_mapper_->isRequiredStop() || !local_mapper_->finish_once())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    LOG(WARNING) << "[LoopClosure] local_mapper_ stop!";

    curKeyFrame_->updateConnections();

    for(KeyFrame::Ptr kf:all_kfs)
    {
        kf->beforeUpdate_Tcw_ = kf->Tcw();
    }

    std::set<KeyFrame::Ptr> CurrentConnectedKFs_set = curKeyFrame_->getConnectedKeyFrames();
    mvpCurrentConnectedKFs.assign(CurrentConnectedKFs_set.begin(),CurrentConnectedKFs_set.end());
    mvpCurrentConnectedKFs.push_back(curKeyFrame_);

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;

    CorrectedSim3[curKeyFrame_] = sim3_cw;

    LOG(WARNING) << "[LoopClosure] ======curKeyFrame_ sim3======";
    std::cout<<" R ======== <<"<<std::endl<< sim3_cw.rotationMatrix() <<std::endl;
    std::cout<<" t ======== <<"<<std::endl<< sim3_cw.translation().transpose() <<std::endl;
    std::cout<<" s ======== <<"<<std::endl<< sim3_cw.scale() <<std::endl;

    {
        std::unique_lock<std::mutex > lock(local_mapper_->map_->mutex_update_);
        for(std::vector<KeyFrame::Ptr>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
        {
            KeyFrame::Ptr pKFi = *vit;

            SE3d Tiw = pKFi->Tcw();

            if(pKFi != curKeyFrame_)
            {
                SE3d Tic = Tiw * curKeyFrame_->Twc();
                Matrix3d Ric = Tic.rotationMatrix();
                Vector3d tic = Tic.translation();
                Matrix4d temp;
                temp.topLeftCorner(3,3) = Ric;
                temp.topRightCorner(3,1) = tic;
                Sophus::Sim3d sim3_Sic(temp);
                Sophus::Sim3d CorrectedSiw = sim3_Sic * sim3_cw;
                CorrectedSim3[pKFi] = CorrectedSiw;
            }
            Matrix3d Riw = Tiw.rotationMatrix();
            Vector3d tiw = Tiw.translation();
            Matrix4d temp;
            temp.topLeftCorner(3,3) = Riw;
            temp.topRightCorner(3,1) = tiw;
            Sophus::Sim3d sim3_Siw(temp);
            NonCorrectedSim3[pKFi] = sim3_Siw;
        }

        LOG(WARNING) << "[LoopClosure] Loop->update mpt and kf pose by sim3.";

        for(KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame::Ptr pKFi = mit->first;
            Sophus::Sim3d CorrectedSiw = mit->second;
            Sophus::Sim3d CorrectedSwi = CorrectedSiw.inverse();
            Sophus::Sim3d Siw = NonCorrectedSim3[pKFi];
            std::vector<MapPoint::Ptr > vpMPsi = pKFi->getMapPoints();
            Sophus::Sim3d sim3_correct_pose = CorrectedSwi * Siw;

            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
                MapPoint::Ptr pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
                if(pMPi->mnCorrectedByKF == curKeyFrame_->id_)
                    continue;
                Vector3d P3Dw = pMPi->pose();
                Vector3d P3Dw_NonCorrected = Siw * P3Dw;
                Vector3d P3Dw_Corrected = CorrectedSwi * P3Dw_NonCorrected;
                pMPi->setPose(P3Dw_Corrected);
                pMPi->mnCorrectedByKF = curKeyFrame_->id_;
                pMPi->mnCorrectedReference = pKFi->id_;
                pMPi->updateViewAndDepth();
            }

            Eigen::Matrix3d eigR = CorrectedSiw.rotationMatrix();
            Eigen::Vector3d eigt = CorrectedSiw.translation();
            double s = CorrectedSiw.scale();

            eigt *=(1.0/s); //[R t/s;0 1]
            SE3d correctedTiw = SE3d(eigR,eigt);
            pKFi->setTcw(correctedTiw);
            pKFi->updateConnections();
        }

        LOG(WARNING) << "[LoopClosure] Loop->Finish update mpt and kf pose by sim3.";


        bool traj_afterSim3 = true;
        if(traj_afterSim3)
        {
            std::string trajAftersim3 = "traj_afterSim3.txt";
            std::ofstream f_trajAftersim3;
            f_trajAftersim3.open(trajAftersim3.c_str());
            f_trajAftersim3 << std::fixed;

            for(KeyFrame::Ptr kf:all_kfs)
            {
                Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
                Vector3d t = frame_pose.translation();
                Quaterniond q = frame_pose.unit_quaternion();

                f_trajAftersim3 << std::setprecision(6) << kf->timestamp_ << " "
                                << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            }
            f_trajAftersim3.close();
            std::cout<<"traj_afterSim3 saved!"<<std::endl;
        }

        std::set<KeyFrame::Ptr> kf_toUpdate;
        int replace_num = 0;
        int match_num = 0;
        for(size_t i=0; i<CurrentMatchedPoints.size(); i++)
        {
            if(CurrentMatchedPoints[i])
            {
                match_num++;
                MapPoint::Ptr pLoopMP = CurrentMatchedPoints[i];
                if(pLoopMP->isBad())
                    continue;

                MapPoint::Ptr pCurMP = curKeyFrame_->mapPointsInBow[i];
                //todo 没有填补的功能
                if(pCurMP != pLoopMP)
                {
                    pLoopMP->fusion(pCurMP,true);
                    std::map<KeyFrame::Ptr, Feature::Ptr> obs = pCurMP->getObservations();

                    for(auto &item:obs)
                    {
                        KeyFrame::Ptr kf = item.first;
                        Feature::Ptr ft = kf->getFeatureByMapPoint(pCurMP);
                        if(!ft)
                            continue;
                        ft->mpt_ = pLoopMP;
                        kf->removeMapPoint(pCurMP);
                        kf->addFeature(ft);
                        kf_toUpdate.insert(kf);
                    }
                    replace_num++;
                }
            }
        }
        update_finish_ = true;
    }

    searchAndFuse(CorrectedSim3);
    std::map<KeyFrame::Ptr, std::set<KeyFrame::Ptr> > LoopConnections;

    for(std::vector<KeyFrame::Ptr>::iterator vit = mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame::Ptr pKFi = *vit;
        std::set<KeyFrame::Ptr > vpPreviousNeighbors_set = pKFi->getConnectedKeyFrames();
        std::vector<KeyFrame::Ptr > vpPreviousNeighbors;
        vpPreviousNeighbors.assign(vpPreviousNeighbors_set.begin(),vpPreviousNeighbors_set.end());
        pKFi->updateConnections();
        LoopConnections[pKFi]=pKFi->getConnectedKeyFrames();

        for(std::vector<KeyFrame::Ptr >::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev);
        }

        for(std::vector<KeyFrame::Ptr >::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    LOG(WARNING) << "[LoopClosure] LoopConnections(闭环产生) size:"<<LoopConnections.size();

    Optimizer::OptimizeEssentialGraph(local_mapper_->map_, MatchedKeyFrame_, curKeyFrame_, NonCorrectedSim3, CorrectedSim3, LoopConnections, false);

    update_finish_ = true;

    bool traj_afterEss = true;

    if(traj_afterEss)
    {
        std::string trajAfterEss = "traj_afterEss.txt";
        std::ofstream f_trajAfterEss;
        f_trajAfterEss.open(trajAfterEss.c_str());
        f_trajAfterEss << std::fixed;

        all_kfs = local_mapper_->map_->getAllKeyFrames();
        std::sort(all_kfs.begin(),all_kfs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});

        for(KeyFrame::Ptr kf:all_kfs)
        {
            Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();

            f_trajAfterEss << std::setprecision(6) << kf->timestamp_ << " "
                           << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        f_trajAfterEss.close();
        std::cout<<"traj_afterEss saved!"<<std::endl;
    }

    // 添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）因为这条边的优化没有意义，误差总是0，所以在优化之后再添加
    MatchedKeyFrame_->addLoopEdge(curKeyFrame_);
    curKeyFrame_->addLoopEdge(MatchedKeyFrame_);

    // Launch a new thread to perform Global Bundle Adjustment
    RunningGBA_ = true;
    FinishedGBA_ = false;
    StopGBA_ = false;
    thread_GBA_ = new std::thread(&LoopClosure::RunGlobalBundleAdjustment,this,curKeyFrame_->id_);

    local_mapper_->release();
    LOG(WARNING) << "[LoopClosure] Loop Closed!" << std::endl;

    LastLoopKFid_ = curKeyFrame_->id_;
}


std::vector<KeyFrame::Ptr> LoopClosure::DetectLoopCandidates(double minScore)
{
    std::set<KeyFrame::Ptr> connectedKeyFrames = curKeyFrame_->getConnectedKeyFrames();
    std::list<KeyFrame::Ptr> sharingWordsKF;

    std::map<KeyFrame::Ptr,int> kf_words;
    std::map<KeyFrame::Ptr,double> kf_score;
    int maxCommonWords = 0;

    // 步骤1：找出和当前帧具有公共单词的所有关键帧（不包括与当前帧链接的关键帧）
    DBoW3::QueryResults results;
    {
        std::unique_lock<std::mutex> lock(mutex_database_);
        database_->query(curKeyFrame_->bow_vec_,results,-1,-1);
    }

    for(auto &ret:results)
    {
        if(connectedKeyFrames.count(KeyFrames[ret.Id]) || KeyFrames[ret.Id]==curKeyFrame_)
            continue;
        sharingWordsKF.push_back(KeyFrames[ret.Id]);
        KeyFrames[ret.Id]->loop_query_ = curKeyFrame_->id_;
        kf_words.insert(std::pair<KeyFrame::Ptr,int>(KeyFrames[ret.Id],0));
    }
    //! nwords,  words in common , is filled only by Bhatt score!
    for(auto &k_w:kf_words)
    {
        const DBoW3::BowVector &BowVec = k_w.first->bow_vec_;
        for (auto word_id:curKeyFrame_->bow_vec_)
        {
            if(BowVec.count(word_id.first))
            {
                k_w.second ++;
            }
        }
        maxCommonWords = k_w.second>maxCommonWords?k_w.second:maxCommonWords;
    }

    LOG(WARNING) << "[LoopClosure] Loop KFs(no ConnectedKeyFrames) which have common word : "<< sharingWordsKF.size();
    LOG(WARNING) << "[LoopClosure] In the loop KFs, the max commonWords is"<< maxCommonWords;

    if(sharingWordsKF.empty())
    {
        LOG(WARNING) << "[LoopClosure] There is no KF sharing same Words with current kf.";
        return std::vector<KeyFrame::Ptr>();
    }

    //! satisfy score and word limit
    std::list<std::pair<double ,KeyFrame::Ptr> > scoreAndMatch;
    // Only compare against those keyframes that share enough words

    //todo test
    int minCommonWords = maxCommonWords*0.8f ;

    LOG(WARNING) << "[LoopClosure] The min commonWords it should have : "<<minCommonWords;


    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(std::list<KeyFrame::Ptr>::iterator lit = sharingWordsKF.begin(); lit != sharingWordsKF.end(); lit++)
    {
        KeyFrame::Ptr pKF = *lit;
        double si = vocabulary_->score(curKeyFrame_->bow_vec_,pKF->bow_vec_);
        kf_score.insert(std::pair<KeyFrame::Ptr,double>(pKF,si));

        if(kf_words[pKF]>minCommonWords && si >= minScore)
            scoreAndMatch.push_back(make_pair(si,pKF));
    }

    LOG(WARNING) << "[LoopClosure] The KeyFrame have enough score and common words number: "<<scoreAndMatch.size();

    if(scoreAndMatch.empty())
    {
        LOG(WARNING) << "[LoopClosure] There is no KF sharing enough same Words and score with current kf";
        return std::vector<KeyFrame::Ptr>();
    }

    std::list<std::pair<double ,KeyFrame::Ptr> > accScoreAndMatch;
    float bestAccScore = minScore;

    // accumulate score by covisibility
    for(std::list<std::pair<double ,KeyFrame::Ptr> >::iterator it = scoreAndMatch.begin(); it != scoreAndMatch.end(); it++)
    {
        KeyFrame::Ptr pKF = it->second;
        std::set<KeyFrame::Ptr> vpNeighs_set = pKF->getConnectedKeyFrames(10);
        std::vector<KeyFrame::Ptr> vpNeighs;
        vpNeighs.assign(vpNeighs_set.begin(),vpNeighs_set.end());

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame::Ptr pBestKF = pKF;
        for(std::vector<KeyFrame::Ptr>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame::Ptr pKF2 = *vit;

            //! 如果有共同单词，之前一定检测到了，同时计算了得分
            if(kf_words.find(pKF2)==kf_words.end())
                continue;

            if(pKF2->loop_query_==curKeyFrame_->id_ && kf_words[pKF2]>minCommonWords)
            {
                accScore += kf_score[pKF2];
                if(kf_score[pKF2]>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = kf_score[pKF2];
                }
            }
        }

        accScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    for(auto &it:accScoreAndMatch)
    {
        LOG(WARNING) << "[LoopClosure] The best score in the scoreAndMatch group，<best kfid,score>=====: <"<<it.second->id_<<","<<it.first<<">";
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    LOG(WARNING) << "[LoopClosure] The min score the keyframe should have:  "<<minScoreToRetain;

    std::set<KeyFrame::Ptr> spAlreadyAddedKF;
    std::vector<KeyFrame::Ptr> vpLoopCandidates;
    vpLoopCandidates.reserve(accScoreAndMatch.size());

    for(std::list<std::pair<double,KeyFrame::Ptr> >::iterator it=accScoreAndMatch.begin(), itend=accScoreAndMatch.end(); it!=itend; it++)
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
    LOG(WARNING) << "[LoopClosure] The keyframes fit all conditions：  "<<vpLoopCandidates.size();

    return vpLoopCandidates;
}

/**
 * @brief get matches between curFeyFrame and loopKeyFrame by bow
 * @param curFeyFrame  [in]
 * @param loopKeyFrame [in]
 * @param Matches12 [out] mappoints in loopkeyframe ,和curkf中的特征点是一一对应的。
 *          [mpt1_1  mpt_a_2 ]  the first mpt in curkf is match to the ath mpt in loopkf
 *          [mpt2_1  mpt_b_2 ]
 *          [mpt3_1  mpt_c_2 ]
 *          [mpt4_1  mpt_d_2 ]
 *          [mpt5_1  mpt_e_2 ]
 * @param [out] bestidx idx of matches mpt in loopkeyframe
 * @return
 */
int LoopClosure::SearchByBoW(KeyFrame::Ptr loopKeyFrame, std::vector<MapPoint::Ptr> &Matches12, std::vector<int > &bestidx)
{
    std::vector<MapPoint::Ptr > mpts_1 = curKeyFrame_->mapPointsInBow;
    std::vector<Feature::Ptr > fts_1 = curKeyFrame_->featuresInBow;

    //! todo 要注意关键帧的mpt是随着时间的推移在增加的，但是新增加的mpt并没有计算描述子
    LOG_ASSERT(mpts_1.size() == curKeyFrame_->mptId_des.size());
    LOG_ASSERT(fts_1.size() == curKeyFrame_->mptId_des.size());


    std::vector<MapPoint::Ptr > mpts_2 = loopKeyFrame->mapPointsInBow;
    std::vector<Feature::Ptr > fts_2 = loopKeyFrame->featuresInBow;
    LOG_ASSERT(mpts_2.size() == loopKeyFrame->mptId_des.size());
    LOG_ASSERT(fts_2.size() == loopKeyFrame->mptId_des.size());

    //! 需要注意，关键帧的描述子和MapPoint的对应关系

    Matches12 = std::vector<MapPoint::Ptr>(mpts_1.size(), static_cast<MapPoint::Ptr>(NULL));
    bestidx = std::vector<int >(mpts_1.size(),-1);
    //把角度平均分成30份
    std::vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(200);

    const float factor = 1.0f/HISTO_LENGTH;
    int nmatches = 0;

    //! 遍历currentkf中的mpt
    for(int i1 = 0; i1<mpts_1.size();i1++)
    {
        //! 可能存在mpt没有node
        if(curKeyFrame_->mptId_nodeId.find(mpts_1[i1]->id_)==curKeyFrame_->mptId_nodeId.end())
            continue;

        //!currentkf 中mpt的nodeid 需要找出loopkey中属于同一个
        DBoW3::NodeId nodeId_1 = curKeyFrame_->mptId_nodeId[mpts_1[i1]->id_];
        cv::Mat des_1 = curKeyFrame_->mptId_des[mpts_1[i1]->id_];

        //! 与mpts_1[i1]属于同一个node的mpt 索引（不是mpt id）
        std::vector<int > mptsId_2_1;

        for (int i2 = 0; i2 < mpts_2.size(); ++i2)
        {
            if(loopKeyFrame->mptId_nodeId.find(mpts_2[i2]->id_) == loopKeyFrame->mptId_nodeId.end())
                continue;

            if( bestidx[i1] != -1 || mpts_2[i2]->isBad())
                continue;
//            if(fts_1[i1]->level_!=fts_2[i2]->level_)
//                continue;
            DBoW3::NodeId nodeId_2 = loopKeyFrame->mptId_nodeId[mpts_2[i2]->id_];
            if(nodeId_1 == nodeId_2)
                mptsId_2_1.emplace_back(i2);
        }

        if(mptsId_2_1.empty())
            continue;

        //! bestDist1 bestDist2 分别记录前两名的距离
        int bestDist1 = 256;
        int bestDist2 = 256;
        int bestIdx2 = -1 ;

        for (int i21 = 0; i21 < mptsId_2_1.size(); ++i21)
        {
            cv::Mat des_2 = loopKeyFrame->mptId_des[mpts_2[mptsId_2_1[i21]]->id_];
            int dist = DBoW3::DescManip::distance(des_1,des_2);

            if(dist<bestDist1)
            {
                bestDist2 = bestDist1;
                bestDist1 = dist;
                bestIdx2 = mptsId_2_1[i21];
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }
        if(bestDist1<TH_LOW)
        {
            if(static_cast<double >(bestDist1)<fNNratio*static_cast<double >(bestDist2))
            {
                //!设置匹配状态并标记
                Matches12[i1] = mpts_2[bestIdx2];
                bestidx[i1] = bestIdx2;

                if(CheckOrientation)
                {
                    LOG_ASSERT(fts_1[i1]->angle>0)<<"the fts-mpt don't have angle , this should not happen."<<std::endl;
                    LOG_ASSERT(fts_2[bestIdx2]->angle>0)<<"the fts-mpt don't have angle , this should not happen."<<std::endl;
                    double rot = fts_1[i1]->angle-fts_2[bestIdx2]->angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
                nmatches++;
            }
        }
    }

    if(CheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                Matches12[rotHist[i][j]]=static_cast<MapPoint::Ptr>(NULL);
                bestidx[rotHist[i][j]] = -1;
                nmatches--;
            }
        }
    }
    return nmatches;
}

/**
 * @brief 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
          通过Sim3变换，确定pKF1的特征点在pKF2中的大致区域，同理，确定pKF2的特征点在pKF1中的大致区域
          在该区域内通过描述子进行匹配捕获pKF1和pKF2之前漏匹配的特征点，更新匹配vpMapPointMatches
 * @param loopKeyFrame [in] 闭环的关键帧
 * @param matches_1_2  [in,out] 记录匹配关系 方便判断，找到匹配之后更新
 * @param matches_2_1  [in,out] 记录匹配关系方便判断，找到匹配之后更新
 * @param s12 [in] Sim3参数，可以考虑把这个换成Sim3的类型减少参数个数
 * @param R12 [in] Sim3参数
 * @param t12 [in] Sim3参数
 * @param th  [in] Sim3参数
 * @param Matches12 [in,out] 维护的 当前帧-闭环帧（localmap） 匹配关系，
 * @return
 */
int LoopClosure::SearchBySim3(KeyFrame::Ptr loopKeyFrame, std::unordered_map<uint64_t,uint64_t > &matches_1_2,std::unordered_map<uint64_t,uint64_t > &matches_2_1,
                              double &s12,const Matrix3d &R12,const Vector3d &t12, float th, std::vector<MapPoint::Ptr> &Matches12, std::vector<int > &bestidx)
{
    std::vector<Feature::Ptr> fts_1 = curKeyFrame_->featuresInBow;
    const int Num_1 = fts_1.size();
    LOG_ASSERT(Num_1 == curKeyFrame_->mptId_des.size());
    std::vector<Feature::Ptr> fts_2 = loopKeyFrame->featuresInBow;
    const int Num_2 = fts_2.size();
    LOG_ASSERT(Num_2 == loopKeyFrame->mptId_des.size());

    Matrix3d sR12 = s12 * R12;
    Matrix3d sR21 = (1.0/s12) * R12.transpose();
    Vector3d t21 = - sR21*t12;

    std::vector<int> vnMatch1(Num_1,-1);
    std::vector<int> vnMatch2(Num_2,-1);

    for (int i1 = 0; i1 < Num_1; ++i1)
    {
        if(matches_1_2.find(fts_1[i1]->mpt_->id_) != matches_1_2.end())
            continue;

        Vector3d mptPose_world = fts_1[i1]->mpt_->pose();
        Vector3d mptPose_cam1 = curKeyFrame_->Tcw() * mptPose_world;
        Vector3d mptPose_cam2 = sR21 * mptPose_cam1 + t21;

        if(mptPose_cam2[2]<0.0)
            continue;

        Vector2d mptPx_cam2 = loopKeyFrame->cam_->project(mptPose_cam2);
        if(!loopKeyFrame->cam_->isInFrame(mptPx_cam2.cast<int>(), 8))
            continue;

        const double maxDistance = fts_1[i1]->mpt_->getMaxDistanceInvariance();
        const double minDistance = fts_1[i1]->mpt_->getMinDistanceInvariance();
        const double dist3D = mptPose_cam2.norm();

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        //todo 5 是否可以直接使用特征点层数
        const double radius = th * (1 << fts_1[i1]->level_);
        //! 取出该区域内的所有特征点(mptPx_cam2,radius)
        std::vector<int > ftsIdx_InCam2Area = loopKeyFrame->getFeaturesInArea(mptPx_cam2[0],mptPx_cam2[1],radius);

        if(ftsIdx_InCam2Area.empty())
            continue;

        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (int i2 = 0; i2 < ftsIdx_InCam2Area.size(); ++i2)
        {
            if(matches_2_1.find(fts_2[ftsIdx_InCam2Area[i2]]->mpt_->id_) != matches_2_1.end())
                continue;
            //todo 6 level check这样是否可以
            int dist = DBoW3::DescManip::distance(curKeyFrame_->mptId_des[fts_1[i1]->mpt_->id_],loopKeyFrame->mptId_des[fts_2[ftsIdx_InCam2Area[i2]]->mpt_->id_]);


            if( dist < bestDist)
            {
                bestDist = dist;
                bestIdx = ftsIdx_InCam2Area[i2];
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1] = bestIdx;
        }

    }

    for (int i2 = 0; i2 < Num_2; ++i2)
    {
        if(matches_2_1.find(fts_2[i2]->mpt_->id_) != matches_2_1.end())
            continue;

        Vector3d mptPose_world = fts_2[i2]->mpt_->pose();
        Vector3d mptPose_cam2 = loopKeyFrame->Tcw() * mptPose_world;
        Vector3d mptPose_cam1 = sR12 * mptPose_cam2 + t12;

        if(mptPose_cam1[2]<0.0)
            continue;

        Vector2d mptPx_cam1 = curKeyFrame_->cam_->project(mptPose_cam1);
        if(!curKeyFrame_->cam_->isInFrame(mptPx_cam1.cast<int>(), 8))
            continue;

        const double maxDistance = fts_2[i2]->mpt_->getMaxDistanceInvariance();
        const double minDistance = fts_2[i2]->mpt_->getMinDistanceInvariance();
        const double dist3D = mptPose_cam1.norm();

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        const double radius = th * (1 << fts_2[i2]->level_);

        //! 取出该区域内的所有特征点(mptPx_cam2,radius)
        std::vector<int > fts_InCam1Area = curKeyFrame_->getFeaturesInArea(mptPx_cam1[0],mptPx_cam1[1],radius);

        int bestDist = INT_MAX;
        int bestIdx = -1;



        for (int i1 = 0; i1 < fts_InCam1Area.size(); ++i1)
        {
            if(matches_1_2.find(fts_1[fts_InCam1Area[i1]]->mpt_->id_) != matches_1_2.end())
                continue;
            //todo level check
//            if(fts_1[fts_InCam1Area[i1]]->level_ != fts_2[i2]->level_)
//                continue;
            int dist = DBoW3::DescManip::distance(loopKeyFrame->mptId_des[fts_2[i2]->mpt_->id_],curKeyFrame_->mptId_des[fts_1[fts_InCam1Area[i1]]->mpt_->id_]);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = fts_InCam1Area[i1];
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2] = bestIdx;
        }
    }
    //! 必须双向匹配到才行
    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<Num_1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2 >= 0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                Matches12[i1] = fts_2[idx2]->mpt_;
                matches_1_2.insert(std::make_pair(fts_1[i1]->mpt_->id_,fts_2[idx2]->mpt_->id_));
                matches_2_1.insert(std::make_pair(fts_2[idx2]->mpt_->id_,fts_1[i1]->mpt_->id_));
                bestidx[i1] = idx2;
                nFound++;
            }
        }
    }
    return nFound;
}

int LoopClosure::SearchByProjection(int th)
{

    Matrix3d sRcw = T_cw.rotationMatrix();
    double s2 = sRcw.block<1,3>(0,0) * sRcw.block<3,1>(0,0);
    double scw = sqrt(s2);
    Matrix3d Rcw = sRcw/scw;
    Vector3d tcw = T_cw.translation()/scw;
    Vector3d Ow = - Rcw.transpose() * tcw;

    std::set<MapPoint::Ptr> AlreadyFound(CurrentMatchedPoints.begin(), CurrentMatchedPoints.end());
    AlreadyFound.erase(static_cast<MapPoint::Ptr>(NULL));

    int nmatches=0;

    for(int iMP=0, iendMP = LoopMapPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint::Ptr pMP = LoopMapPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || AlreadyFound.count(pMP))
            continue;
        Vector3d mptPose_world = pMP->pose();

        Vector3d mptPose_cam = Rcw * mptPose_world + tcw;

        if(mptPose_cam[2]<0.0)
            continue;

        Vector2d px = curKeyFrame_->cam_->project(mptPose_cam);
        const float u = px[0];
        const float v = px[1];

        // Point must be inside the image
        if(!curKeyFrame_->cam_->isInFrame(px.cast<int>()))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->getMaxDistanceInvariance();
        const float minDistance = pMP->getMinDistanceInvariance();

        //todo 这里的方向有点问题吧
        Vector3d PO =Ow - mptPose_world;
        const float dist = PO.norm();


        if(dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        //todo 角度计算
        Vector3d Pn = pMP->getObsVec();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int predictedLevel = LoopMapPoints[iMP]->predictScale(dist,4);

        // Search in a radius
        const double radius = th * (1 << predictedLevel);

        std::vector<int > fts_InCurCamArea = curKeyFrame_->getFeaturesInArea(u,v,radius);

        if(fts_InCurCamArea.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //todo 每一个mpt被观测到的次数有点少
        std::vector<cv::Mat > dMPs = pMP->getDescriptors();

        if(dMPs.empty())
            continue;

        cv::Mat dMP = computeDistinctiveDescriptors(dMPs);

        int bestDist = 256;
        int bestIdx = -1;
        for (int i1 = 0; i1 < fts_InCurCamArea.size(); ++i1)
        {
            if(CurrentMatchedPoints[fts_InCurCamArea[i1]])
                continue;

            int kpLevel= curKeyFrame_->featuresInBow[fts_InCurCamArea[i1]]->level_;

            //todo level check
//            if(kpLevel < predictedLevel-1 || kpLevel > predictedLevel)
//                continue;

            int dist = DBoW3::DescManip::distance(dMP,curKeyFrame_->mptId_des[curKeyFrame_->mapPointsInBow[fts_InCurCamArea[i1]]->id_]);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = fts_InCurCamArea[i1];
            }
        }
        if(bestDist<=TH_LOW)
        {
            CurrentMatchedPoints[bestIdx]=pMP;
            nmatches++;
        }
    }
    return nmatches;
}

void LoopClosure::setLocalMapper(std::shared_ptr<LocalMapper> local_mapper)
{
    local_mapper_ = local_mapper;
}

/**
 * @brief 将闭环帧局部地图中的mpt投影到当前帧的共视关键帧中，进行特征点的融合和替换
 * @param CorrectedPosesMap kf-sim3 of current consist ,Corrected by the relative pose of kf-cur and cur sim3
 * @attention 之前已经将当前帧中的mpt替换过了，所以这里是否就不用了？
 */
void LoopClosure::searchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    int fuse_num = 0;

//    std::cout<<"LoopMapPoints number: "<<LoopMapPoints.size()<<std::endl;
    // 遍历闭环相连的关键帧
    for(KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame::Ptr pKF = mit->first;

        if(pKF == curKeyFrame_)
            continue;

        Sophus::Sim3d sim3_Scw = mit->second;

        std::vector<MapPoint::Ptr > vpReplacePoints(LoopMapPoints.size(),static_cast<MapPoint::Ptr>(NULL));
        int new_fuse= fuse(pKF,sim3_Scw,7.5,vpReplacePoints);// 搜索区域系数为4

        fuse_num += new_fuse;

        //todo  Get Map Mutex
        const int nLP = LoopMapPoints.size();

        std::set<KeyFrame::Ptr> kf_toUpdate;

        for(int i=0; i<nLP;i++)
        {
            MapPoint::Ptr pRep = vpReplacePoints[i];
            if(pRep)
            {
                MapPoint::Ptr pLoopMP = LoopMapPoints[i];
                if(pLoopMP->isBad())
                    continue;
                if(pRep != pLoopMP)
                {
                    pLoopMP->fusion(pRep,true);
                    std::map<KeyFrame::Ptr, Feature::Ptr> obs = pRep->getObservations();

                    for(auto &item:obs)
                    {
                        KeyFrame::Ptr kf = item.first;
                        Feature::Ptr ft = kf->getFeatureByMapPoint(pRep);

                        if(!ft)
                            continue;
                        ft->mpt_ = pLoopMP;
                        kf->removeMapPoint(pRep);
                        kf->addFeature(ft);

                        kf_toUpdate.insert(kf);
                    }
                }
            }
        }
    }
}

/**
 *
 * @param pKF  当前帧的相连关键帧
 * @param Scw  当前帧的相连关键帧的矫正后的sim3
 * @param th   投影阈值
 * @param vpReplacePoint 待替换的点(当前帧共视关键帧中的点)，用旧的点替换新的点
 * @return
 */
int LoopClosure::fuse(KeyFrame::Ptr pKF, const Sophus::Sim3d & Scw, float th,
                      std::vector<MapPoint::Ptr> &vpReplacePoint)
{

    double scw = Scw.scale();
    Matrix3d Rcw = Scw.rotationMatrix();
    Vector3d tcw = Scw.translation();
    Vector3d Ow = - Rcw.transpose() * tcw;

    std::set<MapPoint::Ptr> AlreadyFound(pKF->mapPointsInBow.begin(),pKF->mapPointsInBow.end());

    int nFused=0;

    const int nPoints = LoopMapPoints.size();

    // For each Candidate MapPoint Project and Match
    // 遍历所有的MapPoints
    for(int iMP=0, iendMP = LoopMapPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint::Ptr pMP = LoopMapPoints[iMP];

        // Discard Bad MapPoints and already found
        // 丢弃坏的MapPoints和已经匹配上的MapPoints
        if(pMP->isBad() || AlreadyFound.count(pMP))
            continue;
        Vector3d mptPose_world = pMP->pose();
        Vector3d mptPose_cam = Rcw * mptPose_world + tcw;
        if(mptPose_cam[2]<0.0)
            continue;

        Vector2d px = pKF->cam_->project(mptPose_cam);

        const float u = px[0];
        const float v = px[1];

        // Point must be inside the image
        if(!pKF->cam_->isInFrame(px.cast<int>()))
            continue;
        KeyFrame::Ptr ref = pMP->getReferenceKeyFrame();
        Feature::Ptr ft_pmp = ref->getFeatureByMapPoint(pMP);
        if(!ft_pmp)
            continue;

        // Depth must be inside the scale invariance region of the point
        // 判断距离是否在尺度协方差范围内
        const float maxDistance = pMP->getMaxDistanceInvariance();
        const float minDistance = pMP->getMinDistanceInvariance();

        Vector3d PO = Ow - mptPose_world;
        const float dist = PO.norm();

        if(dist < minDistance || dist > maxDistance)
        {
            continue;
        }

        // Viewing angle must be less than 60 deg
        Vector3d Pn = pMP->getObsVec();

        if(PO.dot(Pn)<0.5*dist)
        {
            continue;
        }

        int predictedLevel = LoopMapPoints[iMP]->predictScale(dist,4);

        // Search in a radius
        // 根据尺度确定搜索半径
        const double radius = th * (1 << ft_pmp->level_);

        std::vector<int > fts_InCurCamArea = pKF->getFeaturesInArea(u,v,radius);

        if(fts_InCurCamArea.empty())
        {
            continue;
        }

        // Match to the most similar keypoint in the radius
        std::vector<cv::Mat > dMPs = pMP->getDescriptors();

        if(dMPs.empty())
        {
            continue;
        }

        cv::Mat dMP = computeDistinctiveDescriptors(dMPs);

        int bestDist = 256;
        int bestIdx = -1;
        for (int i1 = 0; i1 < fts_InCurCamArea.size(); ++i1)
        {
            LOG_ASSERT(fts_InCurCamArea[i1]<pKF->featuresInBow.size());
            if(pKF->mapPointsInBow[fts_InCurCamArea[i1]]->isBad())
                continue;

            int kpLevel= pKF->featuresInBow[fts_InCurCamArea[i1]]->level_;

            //todo level check
//            if(kpLevel < predictedLevel-1 || kpLevel > predictedLevel)
//                continue;

            int dist = DBoW3::DescManip::distance(dMP,pKF->mptId_des[pKF->mapPointsInBow[fts_InCurCamArea[i1]]->id_]);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = fts_InCurCamArea[i1];
            }
        }
        if(bestDist<=TH_LOW)
        {
            vpReplacePoint[iMP] = pKF->mapPointsInBow[bestIdx];
            nFused++;
        }
    }

    return nFused;
}

void LoopClosure::RunGlobalBundleAdjustment(uint64_t nLoopKF)
{
    LOG(WARNING) << "[LoopClosure] Starting Global Bundle Adjustment! " << std::endl;

    int idx = FullBAIdx_;

    Optimizer::globleBundleAdjustment(local_mapper_->map_, 20, nLoopKF, true, true);

    {
        std::unique_lock<std::mutex> lock(mutex_GBA_);
        if(idx != FullBAIdx_)
            return;

        if(!StopGBA_)
        {
            LOG(WARNING) << "[LoopClosure] Global Bundle Adjustment finished" << std::endl;
            LOG(WARNING) << "[LoopClosure] Updating map ..." << std::endl;
            local_mapper_->setStop();

            while(!local_mapper_->isRequiredStop())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            std::vector<KeyFrame::Ptr > kfs = local_mapper_->map_->getAllKeyFrames();
            std::vector<MapPoint::Ptr > mpts = local_mapper_->map_->getAllMapPoints();
            std::list<KeyFrame::Ptr > kfs_miss;

            for(KeyFrame::Ptr kf:kfs)
            {
                kf->beforeUpdate_Tcw_ = kf->Tcw();
            }

            for (int iK = 0; iK < kfs.size(); ++iK)
            {
                KeyFrame::Ptr kf = kfs[iK];
                kf->updateConnections();

                kf->beforeGBA_Tcw_ = kf->Tcw();

                if(kf->GBA_KF_ != nLoopKF)
                {
                    kfs_miss.push_back(kf);
                    break;
                }
                kf->setTcw(kf->optimal_Tcw_);
            }

            int iter = 0;
            while(!kfs_miss.empty())
            {
                iter ++;
                KeyFrame::Ptr kf = kfs_miss.front();
                std::set<KeyFrame::Ptr > connectedKeyFrames = kf->getConnectedKeyFrames(5+iter,-1);
                for(KeyFrame::Ptr rkf:connectedKeyFrames)
                {
                    if(rkf->GBA_KF_ == nLoopKF)
                    {
                        SE3d Tci_c = kf->Tcw() * (rkf->beforeGBA_Tcw_.inverse());
                        kf->optimal_Tcw_ = Tci_c * (rkf->optimal_Tcw_);
                        kf->GBA_KF_ = nLoopKF;
                        kf->setTcw(kf->optimal_Tcw_);
                        break;
                    }
                }
                if(kf->GBA_KF_ != nLoopKF)
                {
                    kfs_miss.push_back(kf);
                    break;
                }
                kfs_miss.pop_front();
            }
            LOG_ASSERT(kfs_miss.size() ==0 )<<"There are some independent kfs.";

            for (int iM = 0; iM < mpts.size(); ++iM)
            {
                MapPoint::Ptr mpt = mpts[iM];

                if(mpt->isBad())
                    continue;

                if(mpt->GBA_KF_ == nLoopKF)
                    mpt->setPose(mpt->optimal_pose_);
                else
                {
                    KeyFrame::Ptr rkf = mpt->getReferenceKeyFrame();
                    Vector3d Pcb =  rkf->beforeGBA_Tcw_ * mpt->pose();
                    mpt->optimal_pose_ = rkf->Twc() * Pcb;
                    mpt->setPose(mpt->optimal_pose_);
                }

            }
            LOG(WARNING) << "[LoopClosure] Map updated!";

            bool traj_afterGBA = true;
            if(traj_afterGBA)
            {
                std::sort(kfs.begin(),kfs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});
                std::string trajAfterGBA = "traj_afterGBA.txt";
                std::ofstream f_trajAfterGBA;
                f_trajAfterGBA.open(trajAfterGBA.c_str());
                f_trajAfterGBA << std::fixed;
                for(KeyFrame::Ptr kf:kfs)
                {
                    Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
                    Vector3d t = frame_pose.translation();
                    Quaterniond q = frame_pose.unit_quaternion();

                    f_trajAfterGBA << std::setprecision(6) << kf->timestamp_ << " "
                                   << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

                }
                f_trajAfterGBA.close();
                std::cout<<"traj_afterGBA saved!"<<std::endl;
            }

            local_mapper_->release();
        }
        FinishedGBA_ = true;
        RunningGBA_ = false;
        update_finish_ = true;
    }
}
}

