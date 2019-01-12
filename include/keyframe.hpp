#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "DBoW3/DBoW3.h"

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

    void setNotErase();

    void setErase();

    void addLoopEdge(KeyFrame::Ptr pKF);

    int getWight(KeyFrame::Ptr pKF);

    KeyFrame::Ptr getParent();

    std::set<KeyFrame::Ptr> getLoopEdges();

private:

    KeyFrame(const Frame::Ptr frame);

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

};

}

#endif