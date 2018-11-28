#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "global.hpp"
#include "frame.hpp"
#include <DBoW3/DBoW3.h>
#include <DBoW3/DescManip.h>
#include "brief.hpp"

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

    const ImgPyr opticalImages() const = delete;    //! disable this function

    inline static KeyFrame::Ptr create(DBoW3::Database *mpDatabase_, const Frame::Ptr frame)
    { return Ptr(new KeyFrame(mpDatabase_,frame)); }

    void SetNotErase();

//    void computeDescriptor(const BRIEF::Ptr &brief);

    void computeBoW(const DBoW3::Vocabulary* vocabulary);

private:

    KeyFrame(DBoW3::Database *mpDatabase_, const Frame::Ptr frame);

    void addConnection(const KeyFrame::Ptr &kf, const int weight);

    void updateOrderedConnections();

    void removeConnection(const KeyFrame::Ptr &kf);

public:

    static uint64_t next_id_;

    const uint64_t frame_id_;

    std::vector<Feature::Ptr> dbow_fts_;
    cv::Mat descriptors_;

    std::vector<cv::Mat> descriptors_vec;

    DBoW3::BowVector bow_vec_;

    DBoW3::FeatureVector feat_vec_;

    unsigned int dbow_Id_;

    DBoW3::Database* mpDatabase_;

    private:

    std::map<KeyFrame::Ptr, int> connectedKeyFrames_;

    std::multimap<int, KeyFrame::Ptr> orderedConnectedKeyFrames_;

    bool isBad_;

    std::mutex mutex_connection_;

public:
    // Variables used by the keyframe database
    uint64_t mnLoopQuery;
    int mnLoopWords;
    double mLoopScore;
    uint64_t mnRelocQuery;
    int mnRelocWords;
    double mRelocScore;

};

}

#endif