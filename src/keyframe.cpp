#include "config.hpp"
#include "map.hpp"
#include "keyframe.hpp"

using namespace std;
namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->images(), next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_), isBad_(false), loop_query_(0),
    notErase(false),toBeErase(false),GBA_KF_(0)
{
    mpPrevKeyFrame = NULL;
    mpNextKeyFrame = NULL;
    mpt_fts_ = frame->features();
    setRefKeyFrame(frame->getRefKeyFrame());
    setPose(frame->pose());
}

KeyFrame::KeyFrame(const Frame::Ptr frame, std::vector<IMUData> vIMUData, KeyFrame::Ptr pPrevKF):
        Frame(frame->images(), next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_), isBad_(false), loop_query_(0),
        notErase(false),toBeErase(false),GBA_KF_(0)
{
    mvIMUData = vIMUData;
    if(pPrevKF)
    {
        pPrevKF->SetNextKeyFrame(std::shared_ptr<KeyFrame>(this));
    }
    mpPrevKeyFrame = pPrevKF;
    mpNextKeyFrame = NULL;

    mpt_fts_ = frame->features();
    setRefKeyFrame(frame->getRefKeyFrame());
    setPose(frame->pose());
}
void KeyFrame::updateConnections()
{
    if(isBad())
        return;

    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        for(const auto &it : mpt_fts_)
            fts.push_back(it.second);
    }

    std::map<KeyFrame::Ptr, int> connection_counter;

    for(const Feature::Ptr &ft : fts)
    {
        const MapPoint::Ptr &mpt = ft->mpt_;

        if(mpt->isBad())
        {
            removeFeature(ft);
            continue;
        }

        const std::map<KeyFrame::Ptr, Feature::Ptr> observations = mpt->getObservations();
        for(const auto &obs : observations)
        {
            if(obs.first->id_ == id_)
                continue;
            connection_counter[obs.first]++;
        }
    }

    if(connection_counter.empty())
    {
        setBad();
        return;
    }

    // TODO how to select proper connections
    int connection_threshold = Config::minConnectionObservations();

    KeyFrame::Ptr best_unfit_keyframe;
    int best_unfit_connections = 0;
    std::vector<std::pair<int, KeyFrame::Ptr> > weight_connections;
    for(const auto &obs : connection_counter)
    {
        if(obs.second < connection_threshold)
        {
            best_unfit_keyframe = obs.first;
            best_unfit_connections = obs.second;
        }
        else
        {
            obs.first->addConnection(shared_from_this(), obs.second);
            weight_connections.emplace_back(std::make_pair(obs.second, obs.first));
        }
    }

    if(weight_connections.empty())
    {
        best_unfit_keyframe->addConnection(shared_from_this(), best_unfit_connections);
        weight_connections.emplace_back(std::make_pair(best_unfit_connections, best_unfit_keyframe));
    }

    //! sort by weight
    std::sort(weight_connections.begin(), weight_connections.end(),
              [](const std::pair<int, KeyFrame::Ptr> &a, const std::pair<int, KeyFrame::Ptr> &b){ return a.first > b.first; });

    //! update
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        connectedKeyFrames_.clear();
        for(const auto &item : weight_connections)
        {
            connectedKeyFrames_.insert(std::make_pair(item.second, item.first));
        }

        orderedConnectedKeyFrames_ =
            std::multimap<int, KeyFrame::Ptr>(weight_connections.begin(), weight_connections.end());
    }
}

std::set<KeyFrame::Ptr> KeyFrame::getConnectedKeyFrames(int num, int min_fts)
{
    std::lock_guard<std::mutex> lock(mutex_connection_);

    std::set<KeyFrame::Ptr> connected_keyframes;
    if(num == -1) num = (int) orderedConnectedKeyFrames_.size();

    int count = 0;
    const auto end = orderedConnectedKeyFrames_.rend();
    for(auto it = orderedConnectedKeyFrames_.rbegin(); it != end && it->first >= min_fts && count < num; it++, count++)
    {
        connected_keyframes.insert(it->second);
    }

    return connected_keyframes;
}

std::set<KeyFrame::Ptr> KeyFrame::getSubConnectedKeyFrames(int num)
{
    std::set<KeyFrame::Ptr> connected_keyframes = getConnectedKeyFrames();

    std::map<KeyFrame::Ptr, int> candidate_keyframes;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            if(connected_keyframes.count(sub_kf) || sub_kf == shared_from_this())
                continue;

            if(candidate_keyframes.count(sub_kf))
                candidate_keyframes.find(sub_kf)->second++;
            else
                candidate_keyframes.emplace(sub_kf, 1);
        }
    }

    std::set<KeyFrame::Ptr> sub_connected_keyframes;
    if(num == -1)
    {
        for(const auto &item : candidate_keyframes)
            sub_connected_keyframes.insert(item.first);

        return sub_connected_keyframes;
    }

    //! stort by order
    std::map<int, KeyFrame::Ptr, std::greater<int> > ordered_candidate_keyframes;
    for(const auto &item : candidate_keyframes)
    {
        ordered_candidate_keyframes.emplace(item.second, item.first);
    }

    //! get best (num) keyframes
    for(const auto &item : ordered_candidate_keyframes)
    {
        sub_connected_keyframes.insert(item.second);
        if(sub_connected_keyframes.size() >= num)
            break;
    }

    return sub_connected_keyframes;
}

void KeyFrame::setNotErase()
{
    std::unique_lock<std::mutex> lock(mutex_connection_);
    notErase = true;
}

//! 这函数在闭环的时候常用，因为开始闭环的时候设了卡
void KeyFrame::setErase()
{
    {
        //todo 2 add loop detect finish condition
        std::unique_lock<std::mutex> lock(mutex_connection_);
        if(false)
        {
            notErase = false;
        }
    }
    if(toBeErase)
    {
        setBad();
    }
}

void KeyFrame::setBad()
{

    {
        std::unique_lock<std::mutex> lock(mutex_connection_);
        if(id_ == 0)
            return;
        //! 这里是如果想删除，就等一等，设置toBeErase变量，等闭环结束之后会调用keyframe的seterase，如果想删除的话那会再删除
        if(notErase)
        {
            toBeErase = true;
            return;
        }
    }

    std::cout << "The keyframe " << id_ << " was set to be earased." << std::endl;

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        mpt_fts = mpt_fts_;
    }

    for(const auto &it : mpt_fts)
    {
        it.first->removeObservation(shared_from_this());
    }

    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        isBad_ = true;

        for(const auto &connect : connectedKeyFrames_)
        {
            connect.first->removeConnection(shared_from_this());
        }

        connectedKeyFrames_.clear();
        orderedConnectedKeyFrames_.clear();
        mpt_fts_.clear();
        seed_fts_.clear();
    }
    // TODO change refKF
}

bool KeyFrame::isBad()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    return isBad_;
}

void KeyFrame::addConnection(const KeyFrame::Ptr &kf, const int weight)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        if(!connectedKeyFrames_.count(kf))
            connectedKeyFrames_[kf] = weight;
        else if(connectedKeyFrames_[kf] != weight)
            connectedKeyFrames_[kf] = weight;
        else
            return;
    }

    updateOrderedConnections();
}

void KeyFrame::updateOrderedConnections()
{
    int max = 0;
    std::lock_guard<std::mutex> lock(mutex_connection_);
    orderedConnectedKeyFrames_.clear();
    for(const auto &connect : connectedKeyFrames_)
    {
        auto it = orderedConnectedKeyFrames_.lower_bound(connect.second);
        orderedConnectedKeyFrames_.insert(it, std::pair<int, KeyFrame::Ptr>(connect.second, connect.first));

        if(connect.second > max)
        {
            max = connect.second;
            parent_ = connect.first;
        }
    }
}

void KeyFrame::removeConnection(const KeyFrame::Ptr &kf)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);
        if(connectedKeyFrames_.count(kf))
        {
            connectedKeyFrames_.erase(kf);
        }
    }

    updateOrderedConnections();
}

std::vector<int > KeyFrame::getFeaturesInArea(const double &x, const double &y, const double &r)
{
    std::vector<int > index;

    for(int i = 0; i < featuresInBow.size(); ++i)
    {
        Feature::Ptr it = featuresInBow[i];
        if( it->px_[0] < (x-r) || it->px_[0] > (x+r) || it->px_[1] < (y-r) || it->px_[1] > (y + r))
            continue;
        if(((it->px_[0]- x)*(it->px_[0]- x)+(it->px_[1]- y)*(it->px_[1]- y)) < (double)r*r)
            index.push_back(i);
    }
    return index;
}

void KeyFrame::addLoopEdge(KeyFrame::Ptr pKF)
{
    std::unique_lock<std::mutex > lock(mutex_connection_);
    notErase = true;
    loopEdges_.insert(pKF);
}

int KeyFrame::getWight(KeyFrame::Ptr pKF)
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    return connectedKeyFrames_[pKF];
}
KeyFrame::Ptr KeyFrame::getParent()
{
    return parent_;
}

std::set<KeyFrame::Ptr> KeyFrame::getLoopEdges()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    return loopEdges_;

}

void KeyFrame::UpdateNavStatePVRFromTcw(const SE3d &Tcw,const SE3d &Tbc)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    SE3d Twb = (Tbc*Tcw).inverse();
    Matrix3d Rwb = Twb.rotationMatrix();
    Vector3d Pwb = Twb.translation();

    Matrix3d Rw1 = mNavState.Get_RotMatrix();
    Vector3d Vw1 = mNavState.Get_V();
    Vector3d Vw2 = Rwb * Rw1.transpose()*Vw1;   // bV1 = bV2 ==> Rwb1^T*wV1 = Rwb2^T*wV2 ==> wV2 = Rwb2*Rwb1^T*wV1

    mNavState.Set_Pos(Pwb);
    mNavState.Set_Rot(Rwb);
    mNavState.Set_Vel(Vw2);
}

void KeyFrame::SetInitialNavStateAndBias(const NavState& ns)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState = ns;
    // Set bias as bias+delta_bias, and reset the delta_bias term
    mNavState.Set_BiasGyr(ns.Get_BiasGyr()+ns.Get_dBias_Gyr());
    mNavState.Set_BiasAcc(ns.Get_BiasAcc()+ns.Get_dBias_Acc());
    mNavState.Set_DeltaBiasGyr(Vector3d::Zero());
    mNavState.Set_DeltaBiasAcc(Vector3d::Zero());
}

KeyFrame::Ptr KeyFrame::GetPrevKeyFrame(void)
{
    std::unique_lock<std::mutex> lock(mMutexPrevKF);
    return mpPrevKeyFrame;
}

KeyFrame::Ptr KeyFrame::GetNextKeyFrame(void)
{
    std::unique_lock<std::mutex> lock(mMutexNextKF);
    return mpNextKeyFrame;
}

void KeyFrame::SetPrevKeyFrame(KeyFrame::Ptr pKF)
{
    std::unique_lock<std::mutex> lock(mMutexPrevKF);
    mpPrevKeyFrame = pKF;
}

void KeyFrame::SetNextKeyFrame(KeyFrame::Ptr pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNextKF);
    mpNextKeyFrame = pKF;
}

std::vector<IMUData> KeyFrame::GetVectorIMUData(void)
{
    std::unique_lock<std::mutex> lock(mMutexIMUData);
    return mvIMUData;
}

void KeyFrame::AppendIMUDataToFront(KeyFrame::Ptr pPrevKF)
{
    std::vector<IMUData> vimunew = pPrevKF->GetVectorIMUData();
    {
        std::unique_lock<std::mutex> lock(mMutexIMUData);
        vimunew.insert(vimunew.end(), mvIMUData.begin(), mvIMUData.end());
        mvIMUData = vimunew;
    }
}

void KeyFrame::UpdatePoseFromNS(const Eigen::Matrix4d &Tbc)
{
    Matrix3d Rbc = Tbc.block(0,0,3,3);
    Vector3d Pbc = Tbc.block(0,3,3,1);

    Matrix3d Rwb = mNavState.Get_RotMatrix();
    Vector3d Pwb = mNavState.Get_P();

    SE3d Twb(Rwb,Pwb);
    setTwb(Twb);
}

void KeyFrame::UpdateNavState(const IMUPreintegrator& imupreint, const Vector3d& gw)
{
//    std::unique_lock<std::mutex> lock(mMutexNavState);
//    Converter::updateNS(mNavState,imupreint,gw);
}

void KeyFrame::SetNavState(const NavState& ns)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState = ns;
}

const NavState& KeyFrame::GetNavState(void)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    return mNavState;
}

void KeyFrame::SetNavStateBiasGyr(const Vector3d &bg)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_BiasGyr(bg);
}

void KeyFrame::SetNavStateBiasAcc(const Vector3d &ba)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_BiasAcc(ba);
}

void KeyFrame::SetNavStateVel(const Vector3d &vel)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_Vel(vel);
}

void KeyFrame::SetNavStatePos(const Vector3d &pos)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_Pos(pos);
}

void KeyFrame::SetNavStateRot(const Matrix3d &rot)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_Rot(rot);
}

void KeyFrame::SetNavStateRot(const Sophus::SO3d &rot)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_Rot(rot);
}

void KeyFrame::SetNavStateDeltaBg(const Vector3d &dbg)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_DeltaBiasGyr(dbg);
}

void KeyFrame::SetNavStateDeltaBa(const Vector3d &dba)
{
    std::unique_lock<std::mutex> lock(mMutexNavState);
    mNavState.Set_DeltaBiasAcc(dba);
}

const IMUPreintegrator & KeyFrame::GetIMUPreInt(void)
{
    std::unique_lock<std::mutex> lock(mMutexIMUData);
    return mIMUPreInt;
}

//! jh correct
void KeyFrame::ComputePreInt(void)
{
    std::unique_lock<std::mutex> lock(mMutexIMUData);
    if(mpPrevKeyFrame == NULL)
    {
        if(id_!=0)
        {
            cerr<<"previous KeyFrame is NULL, pre-integrator not changed. id: "<<id_<<endl;
        }
        return;
    }
    else
    {
        // Debug log
        //cout<<std::fixed<<std::setprecision(3)<<
        //      "gyro bias: "<<mNavState.Get_BiasGyr().transpose()<<
        //      ", acc bias: "<<mNavState.Get_BiasAcc().transpose()<<endl;
        //cout<<std::fixed<<std::setprecision(3)<<
        //      "pre-int terms. prev KF time: "<<mpPrevKeyFrame->mTimeStamp<<endl<<
        //      "pre-int terms. this KF time: "<<mTimeStamp<<endl<<
        //      "imu terms times: "<<endl;

        // Reset pre-integrator first
        mIMUPreInt.reset();

        //todo 这里有一点疑问，还没有对关键帧的状态量进行处理呢啊？ 连续跟踪的状态下进行了处理，但是初始化的时候的那两帧是怎么处理的？
        // IMU pre-integration integrates IMU data from last to current, but the bias is from last
        LOG_ASSERT(id_ == (mpPrevKeyFrame->id_+1))<<"This is not the revKeyFrame";
        Vector3d bg = mpPrevKeyFrame->GetNavState().Get_BiasGyr();
        Vector3d ba = mpPrevKeyFrame->GetNavState().Get_BiasAcc();

        //! remember to consider the gap between the last KF and the first IMU
        {
            const IMUData& imu = mvIMUData.front();
            double dt = imu._t - mpPrevKeyFrame->timestamp_;
            LOG_ASSERT(dt>-1e-5)<<"dt is '-', please check";

            //! 对 上一帧---第一个imu数据 空缺的那一块进行预积分，使用的测量值是mvIMUData.front()的测量值
            mIMUPreInt.update(imu._g - bg,imu._a - ba,dt);


            // Test log
//            if(dt < 1e-8)
//            {
//                cerr<<std::fixed<<std::setprecision(3)<<"1 dt = "<<dt<<", prev KF vs last imu time: "<<mpPrevKeyFrame->timestamp_<<" vs "<<imu._t<<endl;
//                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
//            }
            // Debug log
            //cout<<std::fixed<<std::setprecision(3)<<imu._t<<", int dt: "<<dt<<"first imu int since prevKF"<<endl;
        }
        // integrate each imu
        for(size_t i=0; i<mvIMUData.size(); i++)
        {
            const IMUData& imu = mvIMUData[i];

            //! 对 最后一个imu数据---当前帧空缺的那一块进行预积分，使用的测量值是mvIMUData.back（）的测量值,对最后一个imu数据段进行处理的时候直接使用剩余所有的空余段（距离当前KF）
            double nextt;
            if(i==mvIMUData.size()-1)
                nextt = timestamp_;         // last IMU, next is this KeyFrame
            else
                nextt = mvIMUData[i+1]._t;  // regular condition, next is imu data

            // delta time
            double dt = nextt - imu._t;

            LOG_ASSERT(dt>-1e-5)<<"dt is '-', please check";

            // update pre-integrator
            mIMUPreInt.update(imu._g - bg,imu._a - ba,dt);

            // Debug log
            //cout<<std::fixed<<std::setprecision(3)<<imu._t<<", int dt: "<<dt<<endl;

            // Test log
//            if(dt <= 1e-8)
//            {
//                cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu._t<<" vs "<<nextt<<endl;
//                std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
//            }
        }
    }
    // Debug log
    //cout<<"pre-int delta time: "<<mIMUPreInt.getDeltaTime()<<", deltaR:"<<endl<<mIMUPreInt.getDeltaR()<<endl;
}

//! 从Navstate设置待优化变量的状态，即赋初值
void KeyFrame::setOptimizationState()
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

}