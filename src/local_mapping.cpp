#include "config.hpp"
#include "local_mapping.hpp"
#include "feature_alignment.hpp"
#include "feature_tracker.hpp"
#include "image_alignment.hpp"
#include "optimizer.hpp"
#include "time_tracing.hpp"
#include "add_math.hpp"

#ifdef SSVO_DBOW_ENABLE
#include <DBoW3/DescManip.h>
#endif

namespace ssvo{

std::ostream& operator<<(std::ostream& out, const Feature& ft)
{
    Vector3d xyz = ft.mpt_->pose();
    out << "{ px: [" << ft.px_[0] << ", " << ft.px_[1] << "],"
        << " fn: [" << ft.fn_[0] << ", " << ft.fn_[1] << ", " << ft.fn_[2] << "],"
        << " level: " << ft.level_
        << " mpt: " << ft.mpt_->id_ << ", [" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << "] "
        << " }";

    return out;
}

TimeTracing::Ptr mapTrace = nullptr;

//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr fast, bool report, bool verbose) :
    fast_detector_(fast), report_(report), verbose_(report&&verbose),
    mapping_thread_(nullptr), stop_require_(false),finish_once_(false)
{
    map_ = Map::create();

    brief_ = BRIEF::create(2.0, Config::imageNLevel());//,fast_detector_->getHeight(),fast_detector_->getWidth());

    options_.min_disparity = 100;
    options_.min_redundant_observations = 3;
    options_.max_features = Config::minCornersPerKeyFrame();
    options_.num_reproject_kfs = MAX(Config::maxReprojectKeyFrames(), 2);
    options_.num_local_ba_kfs = MAX(Config::maxLocalBAKeyFrames(), 1);
    options_.min_local_ba_connected_fts = Config::minLocalBAConnectedFts();
    options_.num_align_iter = 15;
    options_.max_align_epsilon = 0.01;
    options_.max_align_error2 = 3.0;
    options_.min_found_ratio_ = 0.15;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("local_ba");
    time_names.push_back("reproj");
    time_names.push_back("dbow");
    time_names.push_back("TryinitVIO");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("keyframe_id");
    log_names.push_back("num_reproj_kfs");
    log_names.push_back("num_reproj_mpts");
    log_names.push_back("num_matched");
    log_names.push_back("num_fusion");


    string trace_dir = Config::timeTracingDirectory();
    mapTrace.reset(new TimeTracing("ssvo_trace_map", trace_dir, time_names, log_names));

}
#ifdef SSVO_DBOW_ENABLE
LocalMapper::LocalMapper(DBoW3::Vocabulary* vocabulary, DBoW3::Database* database,const FastDetector::Ptr fast, ImuConfigParam* pParams, bool report, bool verbose) :
        fast_detector_(fast), report_(report), verbose_(report&&verbose),
        mapping_thread_(nullptr), stop_require_(false), vocabulary_(vocabulary), database_(database),
        RunningGBA_(false), FinishedGBA_(true), StopGBA_(false), thread_GBA_(NULL),
        FullBAIdx_(0),update_finish_(false),loop_time_(0),mbMapUpdateFlagForTracking(false),init_kfID_(0)
{
    mbVINSInited = false;
    mbFirstTry = true;
    mbFirstVINSInited = false;
    mnLocalWindowSize = ImuConfigParam::GetLocalWindowSize();
    mpParams = pParams;


    map_ = Map::create();

    brief_ = BRIEF::create(2.0, Config::imageNLevel());//,fast_detector_->getHeight(),fast_detector_->getWidth());

    options_.min_disparity = 100;
    options_.min_redundant_observations = 3;
    options_.max_features = Config::minCornersPerKeyFrame();
    options_.num_reproject_kfs = MAX(Config::maxReprojectKeyFrames(), 2);
    options_.num_local_ba_kfs = MAX(Config::maxLocalBAKeyFrames(), 1);
    options_.min_local_ba_connected_fts = Config::minLocalBAConnectedFts();
    options_.num_align_iter = 15;
    options_.max_align_epsilon = 0.01;
    options_.max_align_error2 = 3.0;
    options_.min_found_ratio_ = 0.15;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("local_ba");
    time_names.push_back("reproj");
    time_names.push_back("dbow");
    time_names.push_back("TryinitVIO");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("keyframe_id");
    log_names.push_back("num_reproj_kfs");
    log_names.push_back("num_reproj_mpts");
    log_names.push_back("num_matched");
    log_names.push_back("num_fusion");


    string trace_dir = Config::timeTracingDirectory();
    mapTrace.reset(new TimeTracing("ssvo_trace_map", trace_dir, time_names, log_names));

}

#endif

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = KeyFrame::create(frame_cur);

    //! before import, make sure the features are stored in the same order!
    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    for(size_t i = 0; i < N; i++)
    {
        fts_ref[i]->mpt_->addObservation(keyframe_ref, fts_ref[i]);
        fts_cur[i]->mpt_->addObservation(keyframe_cur, fts_cur[i]);
    }

    for(const Feature::Ptr &ft : fts_ref)
    {
        map_->insertMapPoint(ft->mpt_);
        ft->mpt_->resetType(MapPoint::STABLE);
        ft->mpt_->updateViewAndDepth();
//        addOptimalizeMapPoint(ft->mpt_);
    }

    keyframe_ref->setRefKeyFrame(keyframe_cur);
    keyframe_cur->setRefKeyFrame(keyframe_ref);
    keyframe_ref->updateConnections();
    keyframe_cur->updateConnections();
    insertKeyFrame(keyframe_ref);
    insertKeyFrame(keyframe_cur);

//    map_->insertKeyFrame(keyframe_ref);
//    map_->insertKeyFrame(keyframe_cur);

    LOG_IF(INFO, report_) << "[Mapper] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur,
                                  std::vector<IMUData> &mvIMUSinceLastKF)
{
    map_->clear();

    std::vector<IMUData> vimu1,vimu2;
    vimu1.empty();
    for(size_t i=0; i<mvIMUSinceLastKF.size(); i++)
    {
        IMUData imu = mvIMUSinceLastKF[i];
        if(imu._t <= frame_cur->timestamp_ && imu._t >= frame_ref->timestamp_)
            vimu2.push_back(imu);
    }

    LOG_ASSERT(frame_ref->timestamp_ == vimu2.front()._t )<<"Wrong in first KF (imu data)"<<std::endl;
    LOG_ASSERT(frame_cur->timestamp_ == vimu2.back()._t )<<"Wrong in first KF (imu data)"<<std::endl;

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = KeyFrame::create(frame_ref,vimu1,NULL);
    KeyFrame::Ptr keyframe_cur = KeyFrame::create(frame_cur,vimu2,keyframe_ref);

    keyframe_ref->ComputePreInt();
    keyframe_cur->ComputePreInt();
    mvIMUSinceLastKF.clear();

    //! before import, make sure the features are stored in the same order!
    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    for(size_t i = 0; i < N; i++)
    {
        fts_ref[i]->mpt_->addObservation(keyframe_ref, fts_ref[i]);
        fts_cur[i]->mpt_->addObservation(keyframe_cur, fts_cur[i]);
    }

    for(const Feature::Ptr &ft : fts_ref)
    {
        map_->insertMapPoint(ft->mpt_);
        ft->mpt_->resetType(MapPoint::STABLE);
        ft->mpt_->updateViewAndDepth();
//        addOptimalizeMapPoint(ft->mpt_);
    }

    keyframe_ref->setRefKeyFrame(keyframe_cur);
    keyframe_cur->setRefKeyFrame(keyframe_ref);
    keyframe_ref->updateConnections();
    keyframe_cur->updateConnections();
    insertKeyFrame(keyframe_ref);
    insertKeyFrame(keyframe_cur);

//    map_->insertKeyFrame(keyframe_ref);
//    map_->insertKeyFrame(keyframe_cur);

    LOG_IF(INFO, report_) << "[Mapper] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

void LocalMapper::startMainThread()
{
    if(mapping_thread_ == nullptr)
        mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::stopMainThread()
{
    setStop();
    if(mapping_thread_)
    {
        if(mapping_thread_->joinable())
            mapping_thread_->join();
        mapping_thread_.reset();
    }
}

void LocalMapper::setStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    stop_require_ = true;
}
bool LocalMapper::isRequiredStop()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    return stop_require_;
}

void LocalMapper::release()
{
    std::unique_lock<std::mutex> lock(mutex_stop_);
    stop_require_ = false;
}

bool LocalMapper::finish_once()
{
    return finish_once_;
}

void LocalMapper::run()
{
    while(!isRequiredStop())
    {
        finish_once_ = false;
        KeyFrame::Ptr keyframe_cur = checkNewKeyFrame();


        if(keyframe_cur)
        {
            mpCurrentKeyFrame = keyframe_cur;
            DeleteBadInLocalWindow();
            AddToLocalWindow(mpCurrentKeyFrame);
            std::cout<<"LocalMapper deal new keyframe～"<<std::endl;
            mapTrace->startTimer("total");
            std::list<MapPoint::Ptr> bad_mpts;
            int new_seed_features = 0;
            int new_local_features = 0;
            if(map_->kfs_.size() > 2)
            {
//                new_seed_features = createFeatureFromSeedFeature(keyframe_cur);
                mapTrace->startTimer("reproj");
                new_local_features = createFeatureFromLocalMap(keyframe_cur, options_.num_reproject_kfs);
                mapTrace->stopTimer("reproj");
                LOG_IF(INFO, report_) << "[Mapper] create " << new_seed_features << " features from seeds and " << new_local_features << " from local map.";

                mapTrace->startTimer("local_ba");
                if(keyframe_cur->id_ <= init_kfID_)
                {

                }
                else if(!GetVINSInited())
                {
                    Optimizer::localBundleAdjustment(keyframe_cur, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
                    SetMapUpdateFlagForTracking(true);
                }
                else
                {
//                    Optimizer::localBundleAdjustment(keyframe_cur, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
                    std::vector<KeyFrame::Ptr> mlLocalKeyFrames_vec;
                    mlLocalKeyFrames_vec.assign(mlLocalKeyFrames.begin(),mlLocalKeyFrames.end());
                    Optimizer::LocalBAPRVIDP(map_, keyframe_cur, mlLocalKeyFrames_vec, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, mGravityVec, report_, verbose_);
                    SetMapUpdateFlagForTracking(true);
                }

                if(!ImuConfigParam::GetRealTimeFlag())
                {
                    // Try to initialize VIO, if not inited
                    if(!GetVINSInited())
                    {
                        mapTrace->startTimer("TryinitVIO");
                        bool tmpbool = TryInitVIO();
                        mapTrace->stopTimer("TryinitVIO");
                        SetVINSInited(tmpbool);
//                        if(tmpbool)
//                        {
//                            // Set initialization flag
//                            SetFirstVINSInited(true);
//                        }

                    }
                }
                mapTrace->stopTimer("local_ba");
            }
            for(const MapPoint::Ptr &mpt : bad_mpts)
            {
                map_->removeMapPoint(mpt);
            }

            checkCulling(keyframe_cur);

            mapTrace->startTimer("dbow");
            addToDatabase(keyframe_cur);
            mapTrace->stopTimer("dbow");

            //todo 删除滑窗中关键帧是否需要对预积分的值进行处理，也就是


            mapTrace->stopTimer("total");
            mapTrace->writeToFile();

            keyframe_last_ = keyframe_cur;

            loop_closure_->insertKeyFrame(keyframe_cur);
        }
        finish_once_ = true;
    }
}

KeyFrame::Ptr LocalMapper::checkNewKeyFrame()
{
    std::unique_lock<std::mutex> lock(mutex_keyframe_);

    if(!keyframes_buffer_.empty())
        cond_process_.wait_for(lock, std::chrono::microseconds(5));

    if(keyframes_buffer_.empty())
        return nullptr;

    KeyFrame::Ptr keyframe = keyframes_buffer_.front();
    keyframes_buffer_.pop_front();

    return keyframe;
}

void LocalMapper::insertKeyFrame(const KeyFrame::Ptr &keyframe)
{
    //! incase add the same keyframe twice
    if(!map_->insertKeyFrame(keyframe))
        return;

    mapTrace->log("frame_id", keyframe->frame_id_);
    mapTrace->log("keyframe_id", keyframe->id_);
    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_keyframe_);
        keyframes_buffer_.push_back(keyframe);
        cond_process_.notify_one();
    }
    else
    {
//        map_->insertKeyFrame(keyframe);
        mapTrace->startTimer("total");
        std::list<MapPoint::Ptr> bad_mpts;
        int new_seed_features = 0;
        int new_local_features = 0;
        if(map_->kfs_.size() > 2)
        {
//            new_seed_features = createFeatureFromSeedFeature(keyframe);
            mapTrace->startTimer("reproj");
            new_local_features = createFeatureFromLocalMap(keyframe, options_.num_reproject_kfs);
            mapTrace->stopTimer("reproj");
            LOG_IF(INFO, report_) << "[Mapper] create " << new_seed_features << " features from seeds and " << new_local_features << " from local map.";

            mapTrace->startTimer("local_ba");
            Optimizer::localBundleAdjustment(keyframe, bad_mpts, options_.num_local_ba_kfs, options_.min_local_ba_connected_fts, report_, verbose_);
            SetMapUpdateFlagForTracking(true);
            mapTrace->stopTimer("local_ba");

            cout<<"会到这里！！！？？？"<<endl;
            std::abort();
        }

        for(const MapPoint::Ptr &mpt : bad_mpts)
        {
            map_->removeMapPoint(mpt);
        }

        checkCulling(keyframe);

        addToDatabase(keyframe);

        mapTrace->stopTimer("total");
        mapTrace->writeToFile();

        keyframe_last_ = keyframe;
    }
}

void LocalMapper::finishLastKeyFrame()
{
//    DepthFilter::updateByConnectedKeyFrames(keyframe_last_, 3);
}

void LocalMapper::createFeatureFromSeed(const Seed::Ptr &seed)
{
    //! create new feature
    MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->getInvDepth()));
    Feature::Ptr ft = Feature::create(seed->px_ref, seed->fn_ref, seed->level_ref, mpt);
    seed->kf->addFeature(ft);
    map_->insertMapPoint(mpt);
    mpt->addObservation(seed->kf, ft);
    mpt->updateViewAndDepth();

    std::set<KeyFrame::Ptr> local_keyframes = seed->kf->getConnectedKeyFrames(10);

    for(const KeyFrame::Ptr &kf : local_keyframes)
    {
        Vector3d xyz_cur(kf->Tcw() * mpt->pose());
        if(xyz_cur[2] < 0.0f)
            continue;

        Vector2d px_cur(kf->cam_->project(xyz_cur));
        if(!kf->cam_->isInFrame(px_cur.cast<int>(), 8))
            continue;

        int level_cur = 0;
        const Vector2d px_cur_last = px_cur;
        int result = FeatureTracker::reprojectMapPoint(kf, mpt, px_cur, level_cur, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2);
        if(result != 1)
            continue;

        double error = (px_cur_last-px_cur).norm();
        if(error > 2.0)
            continue;

        Vector3d ft_cur = kf->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        kf->addFeature(new_feature);

        if(mpt->isBad())
            continue;

        mpt->addObservation(kf, new_feature);
    }

    mpt->updateViewAndDepth();

    if(mpt->observations() > 1)
        Optimizer::refineMapPoint(mpt, 10, true);
}

int LocalMapper::createFeatureFromSeedFeature(const KeyFrame::Ptr &keyframe)
{
    std::vector<Feature::Ptr> seeds = keyframe->getSeeds();

    for(const Feature::Ptr & ft_seed : seeds)
    {
        const Seed::Ptr &seed = ft_seed->seed_;
        MapPoint::Ptr mpt = MapPoint::create(seed->kf->Twc() * (seed->fn_ref/seed->getInvDepth()));

        Feature::Ptr ft_ref = Feature::create(seed->px_ref, seed->fn_ref, seed->level_ref, mpt);
        Feature::Ptr ft_cur = Feature::create(ft_seed->px_, keyframe->cam_->lift(ft_seed->px_), ft_seed->level_, mpt);
        seed->kf->addFeature(ft_ref);
        keyframe->addFeature(ft_cur);
        keyframe->removeSeed(seed);

        map_->insertMapPoint(mpt);
        mpt->addObservation(seed->kf, ft_ref);
        mpt->addObservation(keyframe, ft_cur);

        mpt->updateViewAndDepth();
//        addOptimalizeMapPoint(mpt);
    }

    return (int) seeds.size();
}

int LocalMapper::createFeatureFromLocalMap(const KeyFrame::Ptr &keyframe, const int num)
{
    std::set<KeyFrame::Ptr> local_keyframes = keyframe->getConnectedKeyFrames(num);

    std::unordered_set<MapPoint::Ptr> local_mpts;
    std::vector<MapPoint::Ptr> mpts_cur = keyframe->getMapPoints();
    for(const MapPoint::Ptr &mpt : mpts_cur)
    {
        local_mpts.insert(mpt);
    }

    std::unordered_set<MapPoint::Ptr> candidate_mpts;
    for(const KeyFrame::Ptr &kf : local_keyframes)
    {
        std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        for(const MapPoint::Ptr &mpt : mpts)
        {
            if(local_mpts.count(mpt) || candidate_mpts.count(mpt))
                continue;

            if(mpt->isBad())//! however it should not happen, maybe still some bugs in somewhere
            {
                kf->removeMapPoint(mpt);
                continue;
            }

            if(mpt->observations() == 1 && mpt->getFoundRatio() < options_.min_found_ratio_)
            {
                mpt->setBad();
                map_->removeMapPoint(mpt);
                continue;
            }

            candidate_mpts.insert(mpt);
        }
    }

    const size_t max_new_count = options_.max_features * 1.5 - mpts_cur.size();
    //! match the mappoints from nearby keyframes
    int project_count = 0;
    std::list<Feature::Ptr> new_fts;
    for(const MapPoint::Ptr &mpt : candidate_mpts)
    {
        Vector3d xyz_cur(keyframe->Tcw() * mpt->pose());
        if(xyz_cur[2] < 0.0f)
            continue;

        if(mpt->isBad())
            continue;

        Vector2d px_cur(keyframe->cam_->project(xyz_cur));
        if(!keyframe->cam_->isInFrame(px_cur.cast<int>(), 8))
            continue;

        project_count++;

        int level_cur = 0;
        int result = FeatureTracker::reprojectMapPoint(keyframe, mpt, px_cur, level_cur, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2);
        if(result != 1)
            continue;

        Vector3d ft_cur = keyframe->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        new_fts.push_back(new_feature);

        if(new_fts.size() > max_new_count)
            break;
    }

    //! check whether the matched corner is near a exsit corner.
    //! firstly, create a mask for exsit corners
    const int cols = keyframe->cam_->width();
    const int rows = keyframe->cam_->height();
    cv::Mat mask(rows, cols, CV_16SC1, -1);
    std::vector<Feature::Ptr> old_fts = keyframe->getFeatures();
    const int old_fts_size = (int) old_fts.size();
    for(int i = 0; i < old_fts_size; ++i)
    {
        const Vector2i px = old_fts[i]->px_.cast<int>();
        for(int c = -2; c <= 2; ++c)
        {
            int16_t* ptr = mask.ptr<int16_t>(px[1]+c) + px[0];
            ptr[-2] = (int16_t)i;
            ptr[-1] = (int16_t)i;
            ptr[0] = (int16_t)i;
            ptr[1] = (int16_t)i;
            ptr[2] = (int16_t)i;
        }
    }

    //! check whether the mappoint is already exist
    int created_count = 0;
    int fusion_count = 0;
    for(const Feature::Ptr &ft : new_fts)
    {
        const Vector2i px = ft->px_.cast<int>();
        int64_t id = mask.ptr<int16_t>(px[1])[px[0]];
        //! if not occupied, create new feature
        if(id == -1)
        {
            //! create new features
            keyframe->addFeature(ft);
            ft->mpt_->addObservation(keyframe, ft);
            ft->mpt_->increaseVisible(2);
            ft->mpt_->increaseFound(2);
//            addOptimalizeMapPoint(ft->mpt_);
            created_count++;
            LOG_IF(INFO, verbose_) << " create new feature from mpt " << ft->mpt_->id_;
        }
        //! if already occupied, check whether the mappoint is the same
        else
        {
            MapPoint::Ptr mpt_new = ft->mpt_;
            MapPoint::Ptr mpt_old = old_fts[id]->mpt_;

            if(mpt_new == mpt_old) //! rarely happen
                continue;

            const std::map<KeyFrame::Ptr, Feature::Ptr> obs_new = mpt_new->getObservations();
            const std::map<KeyFrame::Ptr, Feature::Ptr> obs_old = mpt_old->getObservations();

            bool is_same = true;
            std::list<double> squared_dist;
            std::unordered_set<KeyFrame::Ptr> sharing_keyframes;
            for(const auto &it : obs_new)
            {
                const auto it_old = obs_old.find(it.first);
                if(it_old == obs_old.end())
                    continue;

                const Feature::Ptr &ft_old = it_old->second;
                const Feature::Ptr &ft_new = it.second;
                Vector2d px_delta(ft_new->px_ - ft_old->px_);
                squared_dist.push_back(px_delta.squaredNorm());
                is_same &= squared_dist.back() < 1.0; //! only if all the points pair match the conditon

                sharing_keyframes.insert(it.first);
            }

            if(!squared_dist.empty() && !is_same)
            {
//                std::cout << " ***=-=*** ";
//                std::for_each(squared_dist.begin(), squared_dist.end(), [](double dis){std::cout << dis << ", ";});
//                std::cout << std::endl;
//                goto SHOW;
                continue;
            }


            if(obs_old.size() >= obs_new.size())
            {
                //! match all ft in obs_new
                std::list<std::tuple<Feature::Ptr, double, double, int> > fts_to_update;
                for(const auto &it_new : obs_new)
                {
                    const KeyFrame::Ptr &kf_new = it_new.first;
                    if(sharing_keyframes.count(kf_new))
                        continue;

                    const Vector3d kf_new_dir(kf_new->ray().normalized());
                    double max_cos_angle = 0;
                    KeyFrame::Ptr kf_old_ref;
                    for(const auto &it_old : obs_old)
                    {
                        Vector3d kf_old_dir(it_old.first->ray().normalized());
                        double view_cos_angle = kf_old_dir.dot(kf_new_dir);

                        //! find min angle, max cosangle
                        if(view_cos_angle < max_cos_angle)
                            continue;

                        max_cos_angle = view_cos_angle;
                        kf_old_ref = it_old.first;
                    }

                    Feature::Ptr ft_old = obs_old.find(kf_old_ref)->second;
                    Vector3d xyz_new(kf_new->Tcw() * ft_old->mpt_->pose());
                    if(xyz_new[2] < 0.0f)
                        continue;

                    Vector2d px_new(kf_new->cam_->project(xyz_new));
                    if(!kf_new->cam_->isInFrame(px_new.cast<int>(), 8))
                        continue;

                    int level_new = 0;
                    bool matched = FeatureTracker::trackFeature(kf_old_ref, kf_new, ft_old, px_new, level_new, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_new.find(kf_new)->second, px_new[0], px_new[1], level_new);
                }

                //! update ft if succeed
                const AbstractCamera::Ptr &cam = keyframe->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px_[0] = std::get<1>(it);
                    ft_update->px_[1] = std::get<2>(it);
                    ft_update->level_ = std::get<3>(it);
                    ft_update->fn_ = cam->lift(ft_update->px_);
                }

                //! fusion the mappoint
                //! just reject the new one
                mpt_old->fusion(mpt_new);
                map_->removeMapPoint(mpt_new);

//                addOptimalizeMapPoint(mpt_old);

                LOG_IF(INFO, verbose_) << " Fusion mpt " << mpt_old->id_ << " with mpt " << mpt_new->id_;
//                goto SHOW;
            }
            else
            {
                //! match all ft in obs_old
                std::list<std::tuple<Feature::Ptr, double, double, int> > fts_to_update;
                for(const auto &it_old : obs_old)
                {
                    const KeyFrame::Ptr &kf_old = it_old.first;
                    if(sharing_keyframes.count(kf_old))
                        continue;

                    const Vector3d kf_old_dir(kf_old->ray().normalized());
                    double max_cos_angle = 0;
                    KeyFrame::Ptr kf_new_ref;
                    for(const auto &it_new : obs_new)
                    {
                        Vector3d kf_new_dir(it_new.first->ray().normalized());
                        double view_cos_angle = kf_new_dir.dot(kf_old_dir);

                        //! find min angle, max cosangle
                        if(view_cos_angle < max_cos_angle)
                            continue;

                        max_cos_angle = view_cos_angle;
                        kf_new_ref = it_new.first;
                    }

                    Feature::Ptr ft_new = obs_new.find(kf_new_ref)->second;

                    Vector3d xyz_old(kf_old->Tcw() * ft_new->mpt_->pose());
                    if(xyz_old[2] < 0.0f)
                        continue;

                    Vector2d px_old(kf_old->cam_->project(xyz_old));
                    if(!kf_old->cam_->isInFrame(px_old.cast<int>(), 8))
                        continue;

                    int level_old = 0;
                    bool matched = FeatureTracker::trackFeature(kf_new_ref, kf_old, ft_new, px_old, level_old, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

                    if(!matched)
                        continue;

                    //! observation for update
                    fts_to_update.emplace_back(obs_old.find(kf_old)->second, px_old[0], px_old[1], level_old);
                }

                //! update ft if succeed
                const AbstractCamera::Ptr &cam = keyframe->cam_;//! all camera is the same
                for(const auto &it : fts_to_update)
                {
                    const Feature::Ptr &ft_update = std::get<0>(it);
                    ft_update->px_[0] = std::get<1>(it);
                    ft_update->px_[1] = std::get<2>(it);
                    ft_update->level_ = std::get<3>(it);
                    ft_update->fn_ = cam->lift(ft_update->px_);
                }

                //! add new feature for keyframe, then fusion the mappoint
                ft->mpt_ = mpt_new;
                keyframe->addFeature(ft);
                mpt_new->addObservation(keyframe, ft);

                mpt_new->fusion(mpt_old);
                map_->removeMapPoint(mpt_old);

//                addOptimalizeMapPoint(mpt_new);

                LOG_IF(INFO, verbose_) << " Fusion mpt " << mpt_new->id_ << " with mpt " << mpt_old->id_;
//                goto SHOW;
            }

            fusion_count++;
            continue;

//            SHOW:
//            std::cout << " mpt_new: " << mpt_new->id_ << ", " << mpt_new->pose().transpose() << std::endl;
//            for(const auto &it : obs_new)
//            {
//                std::cout << "-kf: " << it.first->id_ << " px: [" << it.second->px_[0] << ", " << it.second->px_[1] << "]" << std::endl;
//            }
//
//            std::cout << " mpt_old: " << mpt_old->id_ << ", " << mpt_old->pose().transpose() << std::endl;
//            for(const auto &it : obs_old)
//            {
//                std::cout << "=kf: " << it.first->id_ << " px: [" << it.second->px_[0] << ", " << it.second->px_[1] << "]" << std::endl;
//            }
//
//            for(const auto &it : obs_new)
//            {
//                string name = "new -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level_).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px_[0]/(1<<it.second->level_), it.second->px_[1]/(1<<it.second->level_));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//
//            for(const auto &it : obs_old)
//            {
//                string name = "old -kf" + std::to_string(it.first->id_);
//                cv::Mat show = it.first->getImage(it.second->level_).clone();
//                cv::cvtColor(show, show, CV_GRAY2RGB);
//                cv::Point2d px(it.second->px_[0]/(1<<it.second->level_), it.second->px_[1]/(1<<it.second->level_));
//                cv::circle(show, px, 5, cv::Scalar(0, 0, 255));
//                cv::imshow(name, show);
//            }
//            cv::waitKey(0);
        }

    }

    mapTrace->log("num_reproj_mpts", project_count);
    mapTrace->log("num_reproj_kfs", local_keyframes.size());
    mapTrace->log("num_fusion", fusion_count);
    mapTrace->log("num_matched", created_count);
    LOG_IF(WARNING, report_) << "[Mapper][1] old points: " << mpts_cur.size() << ". All candidate: " << candidate_mpts.size() << ", projected: " << project_count
                             << ", points matched: " << new_fts.size() << " with " << created_count << " created, " << fusion_count << " fusioned. ";

    return created_count;
}

void LocalMapper::addOptimalizeMapPoint(const MapPoint::Ptr &mpt)
{
    std::unique_lock<std::mutex> lock(mutex_optimalize_mpts_);
    optimalize_candidate_mpts_.push_back(mpt);
}

bool mptOptimizeOrder(const MapPoint::Ptr &mpt1, const MapPoint::Ptr &mpt2)
{
    if(mpt1->type() < mpt1->type())
        return true;
    else if(mpt1->type() == mpt1->type())
    {
        if(mpt1->last_structure_optimal_ < mpt1->last_structure_optimal_)
            return true;
    }

    return false;
}

int LocalMapper::refineMapPoints(const int max_optimalize_num, const double outlier_thr)
{
    double t0 = (double)cv::getTickCount();
    static uint64_t optimal_time = 0;
    std::unordered_set<MapPoint::Ptr> mpts_for_optimizing;
    int optilize_num = 0;
    int remain_num = 0;
    {
        std::unique_lock<std::mutex> lock(mutex_optimalize_mpts_);
        optimalize_candidate_mpts_.sort(mptOptimizeOrder);

        optilize_num = max_optimalize_num == -1 ? (int)optimalize_candidate_mpts_.size() : max_optimalize_num;
        for(int i = 0; i < optilize_num && !optimalize_candidate_mpts_.empty(); ++i)
        {
            if(!optimalize_candidate_mpts_.front()->isBad())
                mpts_for_optimizing.insert(optimalize_candidate_mpts_.front());

            optimalize_candidate_mpts_.pop_front();
        }

        std::list<MapPoint::Ptr>::iterator mpt_ptr = optimalize_candidate_mpts_.begin();
        for(; mpt_ptr!=optimalize_candidate_mpts_.end(); mpt_ptr++)
        {
            if(mpts_for_optimizing.count(*mpt_ptr))
            {
                mpt_ptr = optimalize_candidate_mpts_.erase(mpt_ptr);
            }
        }
        remain_num = (int)optimalize_candidate_mpts_.size();
    }

    std::set<KeyFrame::Ptr> changed_keyframes;
    for(const MapPoint::Ptr &mpt:mpts_for_optimizing)
    {
        Optimizer::refineMapPoint(mpt, 10);

        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->pose());
            if(residual < outlier_thr)
                continue;

            mpt->removeObservation(item.first);
            changed_keyframes.insert(item.first);

            if(mpt->type() == MapPoint::BAD)
                map_->removeMapPoint(mpt);
            else if(mpt->type() == MapPoint::SEED)
                mpt->resetType(MapPoint::STABLE);

            mpt->last_structure_optimal_ = optimal_time;
        }
    }

    optimal_time++;

    for(const KeyFrame::Ptr &kf : changed_keyframes)
    {
        kf->updateConnections();
    }

    double t1 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[Mapper][2] Refine MapPoint Time: " << (t1-t0)*1000/cv::getTickFrequency()
                             << "ms, mpts: " << mpts_for_optimizing.size() << ", remained: " << remain_num;

    return (int)mpts_for_optimizing.size();
}

void LocalMapper::checkCulling(const KeyFrame::Ptr &keyframe)
{

    return;

    const std::set<KeyFrame::Ptr> connected_keyframes = keyframe->getConnectedKeyFrames();

    int count = 0;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        if(kf->id_ == 0 || kf->isBad())
            continue;

        const int observations_threshold = 3;
        int redundant_observations = 0;
        std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        //std::cout << "mpt obs: [";
        for(const MapPoint::Ptr &mpt : mpts)
        {
            std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
            //std::cout << obs.size() << ",";
            if(obs.size() > observations_threshold)
            {
                int observations = 0;
//                const Feature::Ptr &ft = obs[kf];
                for(const auto &it : obs)
                {
                    if(it.first == kf)
                        continue;

                    //if(it.second->level_ <= ft->level_+1)
                    {
                        observations++;
                        if(observations >= options_.min_redundant_observations)
                            break;
                    }
                }

                if(observations >= options_.min_redundant_observations)
                    redundant_observations++;
            }

            if(redundant_observations > mpts.size() * 0.8)
            {
                kf->setBad();
                map_->removeKeyFrame(kf);
                count++;
            }
        }
        //std::cout << "]" <<std::endl;

       // std::cout <<"redundant_observations: " << redundant_observations << " mpts: " << mpts.size() << std::endl;

    }
}

template <>
inline size_t Grid<Feature::Ptr>::getIndex(const Feature::Ptr &element)
{
    const Vector2d &px = element->px_;
    return static_cast<size_t>(px[1]/grid_size_)*grid_n_cols_
        + static_cast<size_t>(px[0]/grid_size_);
}

void LocalMapper::addToDatabase(const KeyFrame::Ptr &keyframe)
{
#ifdef SSVO_DBOW_ENABLE

    std::vector<uint64_t > mpt_id;
    std::vector<Feature::Ptr> fts;
    std::vector<MapPoint::Ptr> mpts;
    keyframe->getFeaturesAndMapPoints(fts,mpts);

    keyframe->featuresInBow = fts;
    keyframe->mapPointsInBow = mpts;

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px_[0], ft->px_[1], 0, ft->level_));
        mpt_id.emplace_back(ft->mpt_->id_);
    }

    Corners new_corners;
    fast_detector_->detect(keyframe->images(), new_corners, old_corners, 1200);

    if(new_corners.size()+old_corners.size()>1000)
    {
        std::sort(new_corners.begin(),new_corners.end(),[](Corner a,Corner b) -> bool { return a.score>b.score;});
        new_corners.resize(1000 - old_corners.size());
    }

    std::vector<cv::KeyPoint> kps;
    for(const Corner & corner : old_corners)
        kps.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));
    for(const Corner & corner : new_corners)
        kps.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));

    brief_->compute(keyframe->images(), kps, keyframe->descriptors_);

    LOG_ASSERT(old_corners.size()==fts.size())<<"the number of two should be equal"<<std::endl;
    for (int j = 0; j < fts.size(); ++j) {
        fts[j]->angle = kps[j].angle;
    }

    keyframe->KeyPoints.assign(kps.begin(),kps.end());

    //! save descriptors of every mpt
    for(int i=0;i< mpt_id.size();i++)
    {
        keyframe->mptId_des.insert(std::make_pair(mpt_id[i],keyframe->descriptors_.row(i)));
    }
    for(int i=0;i< mpt_id.size();i++)
    {
        fts[i]->descriptors_ = keyframe->descriptors_.row(i);
    }

    {
        std::unique_lock<std::mutex> lock(loop_closure_->mutex_database_);
        keyframe->dbow_Id_ = database_->add(keyframe->descriptors_, &keyframe->bow_vec_, &keyframe->feat_vec_);
    }

    for(auto &it:keyframe->feat_vec_)
    {
        for(auto &id:it.second)
        {
            if((int)id < mpt_id.size())
            {
                keyframe->mptId_nodeId.insert(std::make_pair(mpt_id[id],it.first));
                LOG_ASSERT(std::find(keyframe->mapPointsInBow.begin(),keyframe->mapPointsInBow.end(),mpts[id])!=keyframe->mapPointsInBow.end());
            }

        }
    }
    LOG_ASSERT(keyframe->dbow_Id_ == keyframe->id_) << "DBoW Id(" << keyframe->dbow_Id_ << ") is not match the keyframe's Id(" << keyframe->id_ << ")!";
#endif
}

KeyFrame::Ptr LocalMapper::relocalizeByDBoW(const Frame::Ptr &frame, const Corners &corners)
{
    KeyFrame::Ptr reference = nullptr;

#ifdef SSVO_DBOW_ENABLE

    std::vector<cv::KeyPoint> kps;
    for(const Corner & corner : corners)
    {
//        if(!brief_->checkBorder(corner.x,corner.y,corner.level,true))
//            continue;

        kps.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));
    }

    cv::Mat _descriptors;
    brief_->compute(frame->images(), kps, _descriptors);
    std::vector<cv::Mat> descriptors;
    descriptors.reserve(_descriptors.rows);
    for(int i = 0; i < _descriptors.rows; i++)
        descriptors.push_back(_descriptors.row(i));

    DBoW3::BowVector bow_vec;
    DBoW3::FeatureVector feat_vec;
    vocabulary_->transform(descriptors, bow_vec, feat_vec, 4);

    DBoW3::QueryResults results;
    database_->query(bow_vec, results, 1);

    if(results.empty())
        return nullptr;

    DBoW3::Result result = results[0];

    reference = map_->getKeyFrame(result.Id);

#endif

    // TODO 如果有关键帧剔除，则数据库索引存在问题。
    if(reference == nullptr)
        return nullptr;

    LOG_ASSERT(reference->dbow_Id_ == reference->id_) << "DBoW Id(" << reference->dbow_Id_ << ") is not match the keyframe's Id(" << reference->id_ << ")!";

    return reference;
}

#ifdef SSVO_DBOW_ENABLE
void LocalMapper::setLoopCloser(LoopClosure::Ptr loop_closure)
{
    loop_closure_ = loop_closure;
}

#endif

//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
bool LocalMapper::GetUpdatingInitPoses(void)
{
    std::unique_lock<std::mutex> lock(mMutexUpdatingInitPoses);
    return mbUpdatingInitPoses;
}

void LocalMapper::SetUpdatingInitPoses(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexUpdatingInitPoses);
    mbUpdatingInitPoses = flag;
}

KeyFrame::Ptr LocalMapper::GetMapUpdateKF()
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    return mpMapUpdateKF;
}

bool LocalMapper::GetMapUpdateFlagForTracking()
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    return mbMapUpdateFlagForTracking;
}

void LocalMapper::SetMapUpdateFlagForTracking(bool bflag)
{
    std::unique_lock<std::mutex> lock(mMutexMapUpdateFlag);
    mbMapUpdateFlagForTracking = bflag;
    if(bflag)
    {
        mpMapUpdateKF = mpCurrentKeyFrame;
    }
}

bool LocalMapper::GetVINSInited(void)
{
    std::unique_lock<std::mutex> lock(mMutexVINSInitFlag);
    return mbVINSInited;
}

void LocalMapper::SetVINSInited(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexVINSInitFlag);
    mbVINSInited = flag;
}

bool LocalMapper::GetFirstVINSInited(void)
{
    std::unique_lock<std::mutex> lock(mMutexFirstVINSInitFlag);
    return mbFirstVINSInited;
}

void LocalMapper::SetFirstVINSInited(bool flag)
{
    std::unique_lock<std::mutex> lock(mMutexFirstVINSInitFlag);
    mbFirstVINSInited = flag;
}

Vector3d LocalMapper::GetGravityVec()
{
    return mGravityVec;
}

Eigen::Matrix3d LocalMapper::GetRwiInit()
{
    return mRwiInit;
}

void LocalMapper::VINSInitThread()
{
//    unsigned long initedid = 0;
//    std::cerr<<"start VINSInitThread"<<std::endl;
//    while(1)
//    {
//        if(KeyFrame::nNextId > 2)
//            if(!GetVINSInited() && mpCurrentKeyFrame->mnId > initedid)
//            {
//                initedid = mpCurrentKeyFrame->mnId;
//
//                bool tmpbool = TryInitVIO();
//                if(tmpbool)
//                {
//                    //SetFirstVINSInited(true);
//                    //SetVINSInited(true);
//                    break;
//                }
//            }
//        usleep(3000);
//        if(isFinished())
//            break;
//    }
}

//! 在LocalMapper中 只要地图中的关键帧的数目足够多就进行VIO初始化
bool LocalMapper::TryInitVIO()
{
    LOG(WARNING)<<"[LocalMapper] TryInitVIO!"<<mpCurrentKeyFrame->id_<<std::endl;
    if(mbFirstTry)
    {
        mbFirstTry = false;
        mnStartTime = mpCurrentKeyFrame->timestamp_;
    }

    if(mpCurrentKeyFrame->id_ <= mnLocalWindowSize)
    {
        LOG(WARNING)<<"[LocalMapper] No enough kf in mnLocalWindowSize to Init."<<std::endl;
        return false;
    }

    if(mpCurrentKeyFrame->timestamp_ - mnStartTime < ImuConfigParam::GetVINSInitTime())
    {
        return false;
    }
    //设置待保存量的文件
    static bool fopened = false;
    static std::ofstream fgw,fscale,fbiasa,fcondnum,ftime,fbiasg,finit_traj,finit_traj_afterScale,finit_traj_afterScale_gba,finit_traj_biasa,finit_traj_biasg,finit_traj_v_before,finit_traj_v;
    string tmpfilepath = ImuConfigParam::getTmpFilePath();
    if(!fopened)
    {
        // Need to modify this to correct path
        fgw.open(tmpfilepath+"gw.txt");
        fscale.open(tmpfilepath+"scale.txt");
        fbiasa.open(tmpfilepath+"biasa.txt");
        //todo 这个是啥？
        fcondnum.open(tmpfilepath+"condnum.txt");
        ftime.open(tmpfilepath+"computetime.txt");
        fbiasg.open(tmpfilepath+"biasg.txt");
        finit_traj.open(tmpfilepath+"init_traj.txt");
        finit_traj_afterScale.open(tmpfilepath+"init_traj_scale.txt");
        finit_traj_afterScale_gba.open(tmpfilepath+"init_traj_scale_gba.txt");
        finit_traj_biasa.open(tmpfilepath+"init_traj_biasa.txt");
        finit_traj_biasg.open(tmpfilepath+"init_traj_biasg.txt");
        finit_traj_v.open(tmpfilepath+"init_traj_v.txt");
        finit_traj_v_before.open(tmpfilepath+"finit_traj_v_before.txt");
        if(fgw.is_open() && fscale.is_open() && fbiasa.is_open() &&
           fcondnum.is_open() && ftime.is_open() && fbiasg.is_open())
            fopened = true;
        else
        {
            std::cerr<<"file open error in TryInitVIO"<<std::endl;
            fopened = false;
        }
        fgw<<std::fixed<<std::setprecision(6);
        fscale<<std::fixed<<std::setprecision(6);
        fbiasa<<std::fixed<<std::setprecision(6);
        fcondnum<<std::fixed<<std::setprecision(6);
        ftime<<std::fixed<<std::setprecision(6);
        fbiasg<<std::fixed<<std::setprecision(6);
    }

    //! 步骤1. 先进行全局BA并等待位姿优化完成,仅仅在优化之前再进行
    if(mpCurrentKeyFrame->timestamp_ - mnStartTime >= ImuConfigParam::GetVINSInitTime())
        Optimizer::globleBundleAdjustment(map_,20,0,false,false);

    // 保存 finit_traj 轨迹
    finit_traj << std::fixed;
    std::vector<KeyFrame::Ptr> kfs = map_->getAllKeyFrames();
    std::sort(kfs.begin(),kfs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});

    for(auto kf:kfs)
    {
        Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        finit_traj << std::setprecision(6) << kf->timestamp_ << " "
                   << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    finit_traj.close();

    //设置用于变量估计的关键帧的各个状态变量，位姿、预积分等
    Eigen::Matrix4d E_Tbc = ImuConfigParam::GetEigTbc();
    Eigen::Matrix3d E_Rbc = E_Tbc.block<3,3>(0,0);
    Eigen::Vector3d E_pbc = E_Tbc.block<3,1>(0,3);
    Eigen::Matrix3d E_Rcb = E_Rbc.transpose();
    Eigen::Vector3d E_pcb = -E_Rcb*E_pbc;

    // Use all KeyFrames in map to compute
    std::vector<KeyFrame::Ptr> vScaleGravityKF = map_->getAllKeyFrames();
    std::sort(vScaleGravityKF.begin(),vScaleGravityKF.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->id_<kf2->id_;});
    int N = vScaleGravityKF.size();
    std::vector<SE3d,Eigen::aligned_allocator<SE3d>> vE_Twc;
    std::vector<IMUPreintegrator> vIMUPreInt;
    std::vector<KeyFrameInit::Ptr> vKFInit;

    for(int i=0;i<N;i++)
    {
        KeyFrame::Ptr pKF = vScaleGravityKF[i];
        vE_Twc.push_back(pKF->Twc());

        vIMUPreInt.push_back(pKF->GetIMUPreInt());
        KeyFrameInit::Ptr pkfi = KeyFrameInit::Creat(pKF);
        if(i!=0)
        {
            pkfi->mpPrevKeyFrame = vKFInit[i-1];
        }
        vKFInit.push_back(pkfi);
    }

    /** @brief 步骤2. 计算初始的陀螺仪bias
      * @ 计算结果： -0.0037  0.02217 0.0798
      * @ 陀螺仪bias优化结果基本正确，说明关键帧的预积分过程中与陀螺仪bias相关的变量计算是正确的。
      * @ 仅计算，未对关键帧的陀螺仪bias进行更新，后面将ba也求出来之后统一更新
      */
    Vector3d bgest = Optimizer::OptimizeInitialGyroBias(vE_Twc,vIMUPreInt);

    //! 步骤3. 更新 biasg 并且重新预积分。关键帧中的预积分信息在ba，bg，g，都估计完事之后再更新。Update biasg and pre-integration in LocalWindow. Remember to reset back to zero
    /*
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0);//降低外点的影响
    double X[4];
     */
    for(int i=0;i<N;i++)
        vKFInit[i]->bg = bgest;
    for(int i=0;i<N;i++)
        vKFInit[i]->ComputePreInt();

    //! 步骤4. 估计尺度和重力向量（'world' frame (first KF's camera frame)）,并保存初始化变量的结果
    Eigen::MatrixXd A{3*(N-2),4};
    Eigen::VectorXd B{3*(N-2)};

    Eigen::Matrix3d I3;
    A.setZero();
    B.setZero();
    I3.setIdentity();

    for(int i = 0; i<N-2; i++)
    {
        //KeyFrameInit* pKF1 = vKFInit[i];//vScaleGravityKF[i];
        KeyFrameInit::Ptr pKF2 = vKFInit[i+1];
        KeyFrameInit::Ptr pKF3 = vKFInit[i+2];
        // Delta time between frames
        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
        // Pre-integrated measurements
        Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
        Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
        Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();

        SE3d Twc1 = vE_Twc[i];
        SE3d Twc2 = vE_Twc[i+1];
        SE3d Twc3 = vE_Twc[i+2];

        Vector3d pc1 = Twc1.translation();
        Vector3d pc2 = Twc2.translation();
        Vector3d pc3 = Twc3.translation();

        Matrix3d Rc1 = Twc1.rotationMatrix();
        Matrix3d Rc2 = Twc2.rotationMatrix();
        Matrix3d Rc3 = Twc3.rotationMatrix();

        // Stack to A/B matrix
        // lambda*s + beta*g = gamma
        Vector3d lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
        Matrix3d beta = 0.5*I3*(dt12*dt12*dt23 + dt12*dt23*dt23);
        Vector3d gamma = (Rc3-Rc2)*E_pcb*dt12 + (Rc1-Rc2)*E_pcb*dt23 + Rc1*E_Rcb*dp12*dt23 - Rc2*E_Rcb*dp23*dt12 - Rc1*E_Rcb*dv12*dt12*dt23;

        A.block(3*i,0,3,1) = lambda;
        A.block(3*i,1,3,3) = beta;
        B.block(3*i,0,3,1) = gamma;
        // Tested the formulation in paper, -gamma. Then the scale and gravity vector is -xx
        /*
         Matrix<double,3,4> A_;
         Vector3d B_;
         A_.block(0,0,3,1) = lambda;
         A_.block(0,1,3,3) = beta;
         B_ = gamma;

         ceres::CostFunction* costFunction = ceres_slover::AlignmentError::Creat(A_,B_);
         problem.AddResidualBlock(costFunction,loss_function,X);
          */
    }
    // Use svd to compute A*x=B, x=[s,gw] 4x1 vector
    // A = u*w*vt, u*w*vt*x=B
    // w = ut*A*vt'
    // Then x = vt'*winv*u'*B
//    cv::Mat w,u,vt;
    Eigen::MatrixXd u{3*(N-2),4};
    Eigen::MatrixXd w{4,4};
    VectorXd singular{4};
    Eigen::Matrix<double,4,4> v,vt;
    // Note w is 4x1 vector by SVDecomp()

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV );
    v = svd.matrixV();
    vt = v.transpose();
    u = svd.matrixU();
//    w = u.transpose() * A * vt.transpose();
    singular = svd.singularValues();

    for(int i = 0 ; i <4; i++)
        w(i,i) = singular[i];
    // Compute winv
    Eigen::Matrix4d winv = Eigen::Matrix4d::Identity();
//    cv::Mat winv=cv::Mat::eye(4,4,CV_32F);
    for(int i=0;i<4;i++)
    {
        //todo if can do this?
        if(fabs(w(i,i))<1e-10)
        {
            w(i,i) += 1e-10;
            std::cerr<<"w(i) < 1e-10, w="<<std::endl<<w<<std::endl;
        }
        winv(i,i) = 1./w(i,i);
    }
    // Then x = vt'*winv*u'*B
    //todo LearnVIorb中用的是转置，为什么？
    //! ceres 测试过，与求解出的结果一致
    Matrix<double,4,1> x = vt.transpose() * winv * u.transpose() * B;

    Vector3d g_tmp = x.block(1,0,3,1);

    /*
//    X[0] = x(0,0);
//    X[1] = x(1,0);
//    X[2] = x(2,0);
//    X[3] = x(3,0);
//    ceres::Solver::Options options;
//
//    options.linear_solver_type = ceres::DENSE_SCHUR;
//    options.trust_region_strategy_type = ceres::DOGLEG;
//    options.max_num_iterations = 10;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//
//    std::cout<<"Alignment report:"<<std::endl;
//    std::cout<<summary.FullReport()<<std::endl;
     */

//    std::cout<<"x norm: "<<std::endl<<g_tmp.norm()<<std::endl;
    Eigen::VectorXd Error{3*(N-2)};

    Error = A*x - B;
    std::cout<<"First Step Error=====> "<<Error.transpose()*Error<<std::endl;
//
//    x(0,0) = X[0];
//    x(1,0) = X[1];
//    x(2,0) = X[2];
//    x(3,0) = X[3];
//
//    Error = A*x - B;
//    std::cout<<"Error new: "<<std::endl<<Error<<std::endl;
//    std::abort();

    // x=[s,gw] 4x1 vector
    double sstar = x(0,0);    // scale should be positive
    Vector3d gwstar = x.block(1,0,3,1);
    //! 步骤5. 使用重力的magnitude 9.8约束
    // gI = [0;0;1], the normalized gravity vector in an inertial frame, NED type with no orientation.
    Vector3d gI;
    gI.setZero();
    gI[2] = 1;
    // Normalized approx. gravity vecotr in world frame
    Vector3d gwn = gwstar/gwstar.norm();
    // Debug log
    //cout<<"gw normalized: "<<gwn<<endl;

    // vhat = (gI x gw) / |gI x gw|
    Vector3d gIxgwn = gI.cross(gwn);
    double normgIxgwn = gIxgwn.norm();
    Vector3d vhat = gIxgwn/normgIxgwn;
    double theta = std::atan2(normgIxgwn,gI.dot(gwn));
    // Debug log
    //cout<<"vhat: "<<vhat<<", theta: "<<theta*180.0/M_PI<<endl;

    Eigen::Vector3d vhateig = vhat;
    Eigen::Matrix3d RWIeig = Sophus::SO3d::exp(vhateig*theta).matrix();

    Matrix3d Rwi = RWIeig;
    Vector3d GI = gI * ImuConfigParam::GetG();//9.8012;
    // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector

    Eigen::MatrixXd C{3*(N-2),6};
    Eigen::VectorXd D{3*(N-2)};

    for(int i=0; i<N-2; i++)
    {
        KeyFrameInit::Ptr pKF2 = vKFInit[i+1];
        KeyFrameInit::Ptr pKF3 = vKFInit[i+2];
        // Delta time between frames
        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
        // Pre-integrated measurements

        Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
        Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
        Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();

        Eigen::Matrix3d Jpba12 = pKF2->mIMUPreInt.getJPBiasa();
        Eigen::Matrix3d Jvba12 = pKF2->mIMUPreInt.getJVBiasa();
        Eigen::Matrix3d Jpba23 = pKF3->mIMUPreInt.getJPBiasa();

        SE3d Twc1 = vE_Twc[i];
        SE3d Twc2 = vE_Twc[i+1];
        SE3d Twc3 = vE_Twc[i+2];

        Vector3d pc1 = Twc1.translation();
        Vector3d pc2 = Twc2.translation();
        Vector3d pc3 = Twc3.translation();

        Matrix3d Rc1 = Twc1.rotationMatrix();
        Matrix3d Rc2 = Twc2.rotationMatrix();
        Matrix3d Rc3 = Twc3.rotationMatrix();

        // Stack to C/D matrix
        // lambda*s + phi*dthetaxy + zeta*ba = psi
        Matrix<double,3,1> lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
        Matrix<double,3,3> phi = - 0.5*(dt12*dt12*dt23 + dt12*dt23*dt23)*Rwi*Sophus::SO3d::hat(GI);  // note: this has a '-', different to paper
        Matrix<double,3,3> zeta = Rc2*E_Rcb*Jpba23*dt12 + Rc1*E_Rcb*Jvba12*dt12*dt23 - Rc1*E_Rcb*Jpba12*dt23;
        Matrix<double,3,1> psi = (Rc1-Rc2)*E_pcb*dt23 + Rc1*E_Rcb*dp12*dt23 - (Rc2-Rc3)*E_pcb*dt12
                      - Rc2*E_Rcb*dp23*dt12 - Rc1*E_Rcb*dv12*dt23*dt12 - 0.5*Rwi*GI*(dt12*dt12*dt23 + dt12*dt23*dt23); // note:  - paper

        C.block(3*i,0,3,1) = lambda;
        C.block(3*i,1,3,2) = phi.block(0,0,3,2);
        C.block(3*i,3,3,3) = zeta;
        D.block(3*i,0,3,1) = psi;

    }
    Eigen::MatrixXd u2{3*(N-2),6};
    Eigen::MatrixXd w2{6,6};
    VectorXd singular2{6};
    Eigen::Matrix<double,6,6> v2,vt2;
    // Note w is 4x1 vector by SVDecomp()
    // A is changed in SVDecomp() with cv::SVD::MODIFY_A for speed

    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(C, ComputeThinU | ComputeThinV );
    v2 = svd2.matrixV();
    vt2 = v2.transpose();
    u2 = svd2.matrixU();
//    w2 = u2.transpose() * C * vt2.transpose();
    singular2 = svd2.singularValues();

    for(int i = 0 ; i <6; i++)
    {
        w2(i,i) = singular2[i];
    }

    Eigen::Matrix<double,6,6> w2inv = Eigen::Matrix<double,6,6>::Identity();
    for(int i=0;i<6;i++)
    {
        //todo if can do this?
        if(fabs(w2(i,i))<1e-10)
        {
            w2(i,i) += 1e-10;
            // Test log
            std::cerr<<"w2(i) < 1e-10, w2="<<std::endl<<w2<<std::endl;
        }

        w2inv(i,i) = 1./w2(i,i);
    }
    // Then x = vt'*winv*u'*B
    // Use svd to compute C*x=D, x=[s,dthetaxy,ba] 6x1 vector
    // C = u*w*vt, u*w*vt*x=D
    // Then x = vt'*winv*u'*D

    // Then y = vt'*winv*u'*D
    Matrix<double,6,1> y = vt2.transpose() * w2inv * u2.transpose() * D;


    Eigen::VectorXd Error2{3*(N-2)};

    Error2 = C*y - D;

    std::cout<<"Second Step Error=====> "<<Error2.transpose()*Error2<<std::endl;

    //从求解的变量中提取待求变量，然后保存
    double s_ = y(0,0);
    Matrix<double,2,1> dthetaxy = y.block(1,0,2,1);
    Matrix<double,3,1> dbiasa_ = y.block(3,0,3,1);

    // dtheta = [dx;dy;0]
    Matrix<double,3,1> dtheta = Matrix<double,3,1>::Zero();
    dtheta.block(0,0,2,1) = dthetaxy;

    Eigen::Vector3d dthetaeig = dtheta;
    // Rwi_ = Rwi*exp(dtheta)
    Eigen::Matrix3d Rwieig_ = RWIeig * Sophus::SO3d::exp(dthetaeig).matrix();
    Eigen::Matrix3d Rwi_ = Rwieig_;
    // Debug log
    {
        Vector3d gwbefore = Rwi*GI;
        Vector3d gwafter = Rwi_*GI;

        fgw<<gwafter[0]<<" "<<gwafter[1]<<" "<<gwafter[2]<<" "
           <<gwbefore[0]<<" "<<gwbefore[1]<<" "<<gwbefore[2]<<" "
           <<std::endl;
        fscale<<s_<<" "<<sstar<<" "<<std::endl;
        fbiasa<<dbiasa_(0,0)<<" "<<dbiasa_(1,0)<<" "<<dbiasa_(2,0)<<" "<<std::endl;
//        fcondnum<<w2(0,0)<<" "<<w2(1,0)<<" "<<w2(2,0)<<" "<<w2.at<float>(3)<<" "
//                <<w2.at<float>(4)<<" "<<w2.at<float>(5)<<" "<<std::endl;
        //        ftime<<mpCurrentKeyFrame->mTimeStamp<<" "
        //             <<(t3-t0)/cv::getTickFrequency()*1000<<" "<<endl;
        fbiasg<<bgest(0)<<" "<<bgest(1)<<" "<<bgest(2)<<" "<<std::endl;

        std::ofstream fRwi(tmpfilepath+"Rwi.txt");
        fRwi<<Rwieig_(0,0)<<" "<<Rwieig_(0,1)<<" "<<Rwieig_(0,2)<<" "
            <<Rwieig_(1,0)<<" "<<Rwieig_(1,1)<<" "<<Rwieig_(1,2)<<" "
            <<Rwieig_(2,0)<<" "<<Rwieig_(2,1)<<" "<<Rwieig_(2,2)<<std::endl;
        fRwi.close();
    }
    std::cout<<"Finish calculate bg s g ba!"<<std::endl;
    // todo: 怎么判断初始化的结果？ Add some logic or strategy to confirm init status
    // 尺度判断，error判断，bias与近似真值比较

    //根据时间判断是否完成初始化的过程
    bool bVIOInited = false;

    if(mpCurrentKeyFrame->id_>mnLocalWindowSize && mpCurrentKeyFrame->timestamp_ - mnStartTime >= ImuConfigParam::GetVINSInitTime())
    {
        bVIOInited = true;
    }
    else
    {
        std::cout<<"No enough time to init!"<<std::endl;
        return false;
    }


    if(bVIOInited)
    {
        // Set NavState , scale and bias for all KeyFrames
        // Scale
        double scale = s_;
        mnVINSInitScale = s_;
        // gravity vector in world frame
        Vector3d gw = Rwi_*GI;
        mGravityVec = gw;
        Vector3d gweig = gw;
        mRwiInit = Rwi_;

//        scale = 1;
        //! 步骤6. 更新关键帧的状态
        std::cout<<"<================= Begin update pose and v =================>"<<std::endl;
        SetUpdatingInitPoses(true);
        {
            std::unique_lock<std::mutex> lock(map_->mutex_update_);

            //todo 这个要不要加上, 肯定是会有新的关键帧的
            vScaleGravityKF = map_->getAllKeyFrames();
            std::sort(vScaleGravityKF.begin(),vScaleGravityKF.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->id_<kf2->id_;});
            //更新NavState
            for(std::vector<KeyFrame::Ptr>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
            {
                KeyFrame::Ptr pKF = *vit;
                if(pKF->isBad()) continue;

                // Position and rotation of visual SLAM
                Vector3d wPc = pKF->Twc().translation();                   // wPc
                Matrix3d Rwc = pKF->Twc().rotationMatrix();            // Rwc
                // Set position and rotation of navstate
                Vector3d wPb = scale * wPc + Rwc * E_pcb;
                pKF->SetNavStatePos(wPb);
                pKF->SetNavStateRot(Rwc * E_Rcb);
                // Update bias of Gyr & Acc
                pKF->SetNavStateBiasGyr(bgest);
                pKF->SetNavStateBiasAcc(dbiasa_);
                // Set delta_bias to zero. (only updated during optimization)
                pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
                pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
            }

            //利用更新后的状态预积分
            int cnt=0;
            for(std::vector<KeyFrame::Ptr>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++)
            {
                KeyFrame::Ptr pKF = *vit;
                if(pKF->isBad()) continue;
                pKF->ComputePreInt();
            }

            //计算速度

            finit_traj_v_before<<std::fixed;
            for(std::vector<KeyFrame::Ptr>::const_iterator vit=vScaleGravityKF.begin(), vend=vScaleGravityKF.end(); vit!=vend; vit++,cnt++)
            {
                KeyFrame::Ptr pKF = *vit;
                if(pKF->isBad()) continue;

                // compute velocity
                if(pKF != vScaleGravityKF.back())
                {
                    Vector3d wPc = pKF->Twc().translation();                   // wPc
                    Matrix3d Rwc = pKF->Twc().rotationMatrix();            // Rwc

                    KeyFrame::Ptr pKFnext = pKF->GetNextKeyFrame();
                    LOG_ASSERT(pKFnext != NULL)<<" Why no NextKeyFrame"<<std::endl;
                    // IMU pre-int between pKF ~ pKFnext
                    const IMUPreintegrator& imupreint = pKFnext->GetIMUPreInt();
                    // Time from this(pKF) to next(pKFnext)
                    double dt = imupreint.getDeltaTime();                                       // deltaTime

                    Eigen::Vector3d dp = imupreint.getDeltaP();       // deltaP
                    Eigen::Matrix3d Jpba = imupreint.getJPBiasa();    // J_deltaP_biasa
                    Vector3d wPcnext = pKFnext->Twc().translation();           // wPc next
                    Matrix3d Rwcnext = pKFnext->Twc().rotationMatrix();    // Rwc next

                    Eigen::Vector3d vel = - 1./dt*( scale*(wPc - wPcnext) + (Rwc - Rwcnext)*E_pcb + Rwc*E_Rcb*(dp/* + Jpba*dbiasa_*/) + 0.5*gw*dt*dt );

//                    std::cout<<"kf"<<pKF->id_<<"-vel:"<<vel.transpose()<<std::endl;

                    finit_traj_v_before << std::setprecision(6) << pKF->timestamp_ << " "<< std::setprecision(9)<< vel[0] << " " << vel[1] << " " << vel[2] << std::endl;
                    pKF->SetNavStateVel(vel);
                }
                else // 最后一帧关键帧的速度
                {
                    std::cout<<"-----------here is the last KF in vScaleGravityKF------------"<<std::endl;
                    // If this is the last KeyFrame, no 'next' KeyFrame exists
                    KeyFrame::Ptr pKFprev = pKF->GetPrevKeyFrame();
                    LOG_ASSERT(pKFprev)<<"pKFprev is NULL, cnt="<<cnt<<std::endl;
                    if(pKFprev!=vScaleGravityKF[cnt-1]) std::cerr<<"pKFprev!=vScaleGravityKF[cnt-1], cnt="<<cnt<<", id: "<<pKFprev->id_<<" != "<<vScaleGravityKF[cnt-1]->id_<<std::endl;
                    const IMUPreintegrator& imupreint_prev_cur = pKF->GetIMUPreInt();
                    double dt = imupreint_prev_cur.getDeltaTime();
                    Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                    Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
                    //
                    Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                    Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                    Eigen::Vector3d veleig = velpre + gweig*dt + rotpre*( dv + Jvba*dbiasa_ );
                    pKF->SetNavStateVel(veleig);
//                    std::cout<<"last kf"<<pKF->id_<<"-vel:"<<veleig.transpose()<<std::endl;
                    finit_traj_v_before << std::setprecision(6) << pKF->timestamp_ << " "<< std::setprecision(9)<< veleig[0] << " " << veleig[1] << " " << veleig[2] << std::endl;

                }
            }
            finit_traj_v_before.close();

            std::cout<<"Finish update V"<<std::endl;
//            std::abort();

            //! 更新关键帧和地图点的位姿（乘上尺度）


//            std::vector<KeyFrame::Ptr> mspKeyFrames = map_->getAllKeyFrames();
            for(std::vector<KeyFrame::Ptr>::iterator sit=vScaleGravityKF.begin(), send=vScaleGravityKF.end(); sit!=send; sit++)
            {
                KeyFrame::Ptr pKF = *sit;
                SE3d Tcw = pKF->Tcw();
                pKF->beforeUpdate_Tcw_ = Tcw;
                Matrix3d Rcw = Tcw.rotationMatrix();
                Vector3d tcw = Tcw.translation() *scale;
                Tcw = Sophus::SE3d(Rcw,tcw);
                pKF->setTcw(Tcw);
            }

            std::vector<MapPoint::Ptr> mspMapPoints = map_->getAllMapPoints();
            for(std::vector<MapPoint::Ptr>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
            {
                MapPoint::Ptr pMP = *sit;
                pMP->updateScale(scale);
                pMP->updateViewAndDepth();
            }

            std::vector<Seed::Ptr> seed_fts = mpCurrentKeyFrame->getTrueSeeds();;
            for(std::vector<Seed::Ptr>::iterator sit=seed_fts.begin(), send=seed_fts.end(); sit!=send; sit++)
            {
                Seed::Ptr pSeed = (*sit);
                pSeed->updateScale(scale);
            }


            std::cout<<std::endl<<"... Map scale updated ..."<<std::endl<<std::endl;
            KeyFrame::Ptr pNewestKF = vScaleGravityKF.back();


            //! 将没有参与初始化的关键帧的NavState也设置一下，这个应该用不到了
            //这个应该是实时的检测过程中的会用到的代码,不是实时的应该不会发生这个情况
//            LOG_ASSERT(pNewestKF == mpCurrentKeyFrame);
            if(pNewestKF != mpCurrentKeyFrame && ImuConfigParam::GetRealTimeFlag())
            {
                KeyFrame::Ptr pKF;
                // step1. bias&d_bias
                pKF = pNewestKF;
                do {
                    pKF = pKF->GetNextKeyFrame();
                    // Update bias of Gyr & Acc
                    pKF->SetNavStateBiasGyr(bgest);
                    pKF->SetNavStateBiasAcc(dbiasa_);
                    // Set delta_bias to zero. (only updated during optimization)
                    pKF->SetNavStateDeltaBg(Eigen::Vector3d::Zero());
                    pKF->SetNavStateDeltaBa(Eigen::Vector3d::Zero());
                } while (pKF != mpCurrentKeyFrame);


                // step2. re-compute pre-integration
                //！ 然后进行重新积分
                pKF = pNewestKF;
                do {
                    pKF = pKF->GetNextKeyFrame();
                    pKF->ComputePreInt();
                } while (pKF != mpCurrentKeyFrame);

                // step3. update pos/rot
                pKF = pNewestKF;

                //todo 与viorb不同，因为localmapper队列与map的时序问题，产生一点问题
                do {
                    pKF = pKF->GetNextKeyFrame();

                    // Update rot/pos
                    // Position and rotation of visual SLAM
                    Vector3d wPc = pKF->Twc().translation();                   // wPc
                    Matrix3d Rwc = pKF->Twc().rotationMatrix();            // Rwc
                    Vector3d wPb = wPc + Rwc * E_pcb;
                    pKF->SetNavStatePos(wPb);
                    pKF->SetNavStateRot(Rwc * E_Rcb);

                    //pKF->SetNavState();

                    if (pKF != mpCurrentKeyFrame) {
                        KeyFrame::Ptr pKFnext = pKF->GetNextKeyFrame();
                        // IMU pre-int between pKF ~ pKFnext
                        const IMUPreintegrator &imupreint = pKFnext->GetIMUPreInt();
                        // Time from this(pKF) to next(pKFnext)
                        double dt = imupreint.getDeltaTime();                                       // deltaTime

                        Eigen::Vector3d dp = imupreint.getDeltaP();       // deltaP
                        Eigen::Matrix3d Jpba = imupreint.getJPBiasa();    // J_deltaP_biasa
                        Vector3d wPcnext = pKFnext->Twc().translation();           // wPc next
                        Matrix3d Rwcnext = pKFnext->Twc().rotationMatrix();    // Rwc next

                        Eigen::Vector3d vel = -1. / dt * ((wPc - wPcnext) + (Rwc - Rwcnext) * E_pcb +
                                                          Rwc * E_Rcb * (dp + Jpba * dbiasa_) + 0.5 * gw * dt * dt);
                        Eigen::Vector3d veleig = vel;
                        pKF->SetNavStateVel(veleig);

                    } else {
                        // If this is the last KeyFrame, no 'next' KeyFrame exists
                        KeyFrame::Ptr pKFprev = pKF->GetPrevKeyFrame();
                        const IMUPreintegrator &imupreint_prev_cur = pKF->GetIMUPreInt();
                        double dt = imupreint_prev_cur.getDeltaTime();
                        Eigen::Matrix3d Jvba = imupreint_prev_cur.getJVBiasa();
                        Eigen::Vector3d dv = imupreint_prev_cur.getDeltaV();
                        //
                        Eigen::Vector3d velpre = pKFprev->GetNavState().Get_V();
                        Eigen::Matrix3d rotpre = pKFprev->GetNavState().Get_RotMatrix();
                        Eigen::Vector3d veleig = velpre + gweig * dt + rotpre * (dv + Jvba * dbiasa_);
                        pKF->SetNavStateVel(veleig);
                    }
                } while (pKF != mpCurrentKeyFrame);
            }
            std::cout<<std::endl<<"... Map NavState updated ..."<<std::endl<<std::endl;
            update_finish_ = true;
            SetFirstVINSInited(true);
        }
        SetUpdatingInitPoses(false);
//        std::unique_lock<std::mutex> lock(map_->mutex_update_);


        //! 保存加上尺度的轨迹
        finit_traj_afterScale << std::fixed;
        std::vector<KeyFrame::Ptr> kfs_scale = map_->getAllKeyFrames();
        std::sort(kfs_scale.begin(),kfs_scale.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});
        for(auto kf:kfs_scale)
        {
            Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();
            finit_traj_afterScale << std::setprecision(6) << kf->timestamp_ << " "
                                  << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        finit_traj_afterScale.close();

        init_kfID_ = kfs_scale.back()->id_;
//        std::abort();

        //! 保存各个帧的 bias

        //! 在进行Globa BA并保存轨迹
        uint64_t nGBAKF = vScaleGravityKF.back()->id_;
//        RunningGBA_ = true;
//        FinishedGBA_ = false;
//        StopGBA_ = false;
//        thread_GBA_ = new std::thread(&LocalMapper::RunGlobalBundleAdjustment,this,nGBAKF,true);
//        while(!FinishedGBA_)
//        {
//            std::this_thread::sleep_for(std::chrono::milliseconds(1));
//        }
        setStop();
//        std::unique_lock<std::mutex> lock(map_->mutex_update_);
        Optimizer::GlobalBundleAdjustmentNavStatePRV(map_,mGravityVec,10,nGBAKF,false,false);

        finit_traj_afterScale_gba << std::fixed;
        finit_traj_biasa<< std::fixed;
        finit_traj_biasg<< std::fixed;
        finit_traj_v<< std::fixed;
        std::vector<KeyFrame::Ptr> kfs_scale_gba = map_->getAllKeyFrames();
        std::sort(kfs_scale_gba.begin(),kfs_scale_gba.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});
        for(auto kf:kfs_scale_gba)
        {
            Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();
            finit_traj_afterScale_gba << std::setprecision(6) << kf->timestamp_ << " "
                                  << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            Vector3d ba = kf->GetNavState().Get_BiasAcc();
            Vector3d bg = kf->GetNavState().Get_BiasGyr();
            Vector3d v = kf->GetNavState().Get_V();
            finit_traj_biasa << std::setprecision(6) << kf->timestamp_ << " "<< std::setprecision(9)<< ba[0] << " " << ba[1] << " " << ba[2] << std::endl;
            finit_traj_biasg << std::setprecision(6) << kf->timestamp_ << " "<< std::setprecision(9)<< bg[0] << " " << bg[1] << " " << bg[2] << std::endl;
            finit_traj_v << std::setprecision(6) << kf->timestamp_ << " "<< std::setprecision(9)<< v[0] << " " << v[1] << " " << v[2] << std::endl;

        }
        finit_traj_afterScale_gba.close();
        finit_traj_biasa.close();
        finit_traj_biasg.close();
        finit_traj_v.close();


        std::cout<<"==============Finish global BA after v-i init================"<<std::endl;


//        std::abort();
        SetFlagInitGBAFinish(true);
        //不能在这删除，因为还要添加到数据库里面
//        keyframes_buffer_.clear();
//        SetMapUpdateFlagForTracking(true);
        release();
    }
    std::cout<<"-----------------End InitVIO------------------"<<std::endl;
//    std::abort();
    return bVIOInited;
}

void LocalMapper::AddToLocalWindow(KeyFrame::Ptr pKF)
{
    mlLocalKeyFrames.push_back(pKF);
    if(mlLocalKeyFrames.size() > mnLocalWindowSize)
    {
        mlLocalKeyFrames.pop_front();
    }
    else
    {
        KeyFrame::Ptr pKF0 = mlLocalKeyFrames.front();
        while(mlLocalKeyFrames.size() < mnLocalWindowSize && pKF0->GetPrevKeyFrame()!=NULL)
        {
            pKF0 = pKF0->GetPrevKeyFrame();
            if(!pKF0->isBad())
                mlLocalKeyFrames.push_front(pKF0);
        }
    }
}

void LocalMapper::DeleteBadInLocalWindow(void)
{
    if(mlLocalKeyFrames.empty())
        return;

    std::list<KeyFrame::Ptr>::iterator lit = mlLocalKeyFrames.begin();
    while(lit != mlLocalKeyFrames.end())
    {
        KeyFrame::Ptr pKF = *lit;
        //Test log
        if(!pKF) std::cout<<"pKF null?"<<std::endl;
        if(pKF->isBad())
        {
            lit = mlLocalKeyFrames.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}

void LocalMapper::RunGlobalBundleAdjustment(uint64_t nLoopKF,bool vi)
{
    LOG(WARNING) << "[LocalMapping] Starting Global Bundle Adjustment! " << std::endl;

//    int idx = FullBAIdx_;

    if(vi)
    {
//        Optimizer::globleBundleAdjustment(map_, 10, nLoopKF, true, true);
        Optimizer::GlobalBundleAdjustmentNavStatePRV(map_,mGravityVec,20,nLoopKF,false,false);
    }
    else
        Optimizer::globleBundleAdjustment(map_, 10, nLoopKF, true, true);

    {
        std::unique_lock<std::mutex> lock(map_->mutex_update_);

        if(!StopGBA_)
        {
            LOG(WARNING) << "[LocalMapper] Global Bundle Adjustment finished" << std::endl;
            LOG(WARNING) << "[LocalMapper] Updating map ..." << std::endl;
            setStop();

            while(!isRequiredStop())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            std::vector<KeyFrame::Ptr > kfs = map_->getAllKeyFrames();
            std::vector<MapPoint::Ptr > mpts = map_->getAllMapPoints();
            std::list<KeyFrame::Ptr > kfs_miss;

            if(!vi)
            {
                for(KeyFrame::Ptr kf:kfs)
                {
                    std::cout<<kf->id_<<"save beforeUpdate_Tcw_"<<std::endl;
                    kf->beforeUpdate_Tcw_ = kf->Tcw();
                }
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
                        kf->SetNavStateBiasAcc(rkf->GetNavState().Get_BiasAcc());
                        kf->SetNavStateBiasGyr(rkf->GetNavState().Get_BiasGyr());
                        kf->SetNavStateBiasAcc(Vector3d::Zero());
                        kf->SetNavStateBiasGyr(Vector3d::Zero());
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
            release();
        }
        FinishedGBA_ = true;
        RunningGBA_ = false;
//        update_finish_ = true;

    }
}

}