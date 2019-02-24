#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "image_alignment.hpp"
#include "feature_alignment.hpp"
#include "time_tracing.hpp"

namespace ssvo{

std::string Config::file_name_;

TimeTracing::Ptr sysTrace = nullptr;

System::System(std::string config_file, std::string calib_flie) :
    stage_(STAGE_INITALIZE), status_(STATUS_INITAL_RESET),
    last_frame_(nullptr), current_frame_(nullptr), reference_keyframe_(nullptr),loopId_(0)
{
    LOG_ASSERT(!calib_flie.empty()) << "Empty Calibration file input!!!";
    LOG_ASSERT(!config_file.empty()) << "Empty Config file input!!!";
    Config::file_name_ = config_file;

    AbstractCamera::Model model = AbstractCamera::checkCameraModel(calib_flie);
    if(AbstractCamera::Model::PINHOLE == model)
    {
        PinholeCamera::Ptr pinhole_camera = PinholeCamera::create(calib_flie);
        camera_ = std::static_pointer_cast<AbstractCamera>(pinhole_camera);
    }
    else if(AbstractCamera::Model::ATAN == model)
    {
        AtanCamera::Ptr atan_camera = AtanCamera::create(calib_flie);
        camera_ = std::static_pointer_cast<AbstractCamera>(atan_camera);
    }
    else
    {
        LOG(FATAL) << "Error camera model: " << model;
    }

    double fps = camera_->fps();
    if(fps < 1.0) fps = 1.0;
    //! image
    const int nlevel = Config::imageNLevel();
    const int width = camera_->width();
    const int height = camera_->height();
    const int image_border = AlignPatch::Size;
    //! corner detector
    const int grid_size = Config::gridSize();
    const int grid_min_size = Config::gridMinSize();
    const int fast_max_threshold = Config::fastMaxThreshold();
    const int fast_min_threshold = Config::fastMinThreshold();

    fast_detector_ = FastDetector::create(width, height, image_border, nlevel, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    feature_tracker_ = FeatureTracker::create(width, height, 20, image_border, true);
    initializer_ = Initializer::create(fast_detector_, true);
#ifdef SSVO_DBOW_ENABLE
    std::string voc_dir = Config::DBoWDirectory();

    LOG_ASSERT(!voc_dir.empty()) << "Please check the config file! The DBoW directory is not set!";
    DBoW3::Vocabulary* vocabulary = new DBoW3::Vocabulary(voc_dir);
    DBoW3::Database* database= new DBoW3::Database(*vocabulary, true, 4);

    mapper_ = LocalMapper::create(vocabulary, database, fast_detector_,mpParams , true, false);

    loop_closure_ = LoopClosure::creat(vocabulary, database);
    loop_closure_->startMainThread();

    mapper_->setLoopCloser(loop_closure_);
    loop_closure_->setLocalMapper(mapper_);
#else
    mapper_ = LocalMapper::create(fast_detector_, true, false);
#endif
    DepthFilter::Callback depth_fliter_callback = std::bind(&LocalMapper::createFeatureFromSeed, mapper_, std::placeholders::_1);
    depth_filter_ = DepthFilter::create(fast_detector_, depth_fliter_callback, true);
    viewer_ = Viewer::create(mapper_->map_, cv::Size(width, height));

    mapper_->startMainThread();
    depth_filter_->startMainThread();

    time_ = 1000.0/fps;

    options_.min_kf_disparity = /*50*/100;//MIN(Config::imageHeight(), Config::imageWidth())/5;
    options_.min_ref_track_rate = /*0.9*/0.7;


    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("processing");
    time_names.push_back("frame_create");
    time_names.push_back("img_align");
    time_names.push_back("feature_reproj");
    time_names.push_back("motion_ba");
    time_names.push_back("light_affine");
    time_names.push_back("per_depth_filter");
    time_names.push_back("finish");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("num_feature_reproj");
    log_names.push_back("stage");

    string trace_dir = Config::timeTracingDirectory();
    sysTrace.reset(new TimeTracing("ssvo_trace_system", trace_dir, time_names, log_names));
}

System::~System()
{
    sysTrace.reset();

    viewer_->setStop();
    depth_filter_->stopMainThread();
    mapper_->stopMainThread();
    loop_closure_->stopMainThread();

    viewer_->waitForFinish();
}

void System::process(const cv::Mat &image, const double timestamp)
{
    sysTrace->startTimer("total");
    sysTrace->startTimer("frame_create");
    //! get gray image
    double t0 = (double)cv::getTickCount();
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    current_frame_ = Frame::create(gray, timestamp, camera_);
    double t1 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Frame " << current_frame_->id_ << " create time: " << (t1-t0)/cv::getTickFrequency();
    sysTrace->log("frame_id", current_frame_->id_);
    sysTrace->stopTimer("frame_create");

    sysTrace->startTimer("processing");
    if(STAGE_NORMAL_FRAME == stage_)
    {
        status_ = tracking();
    }
    else if(STAGE_INITALIZE == stage_)
    {
        status_ = initialize();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {
        status_ = relocalize();
    }
    sysTrace->stopTimer("processing");

    finishFrame();
}

System::Status System::initialize()
{
    const Initializer::Result result = initializer_->addImage(current_frame_);

    if(result == Initializer::RESET)
        return STATUS_INITAL_RESET;
    else if(result == Initializer::FAILURE || result == Initializer::READY)
        return STATUS_INITAL_PROCESS;

    std::vector<Vector3d> points;
    initializer_->createInitalMap(Config::mapScale());
    mapper_->createInitalMap(initializer_->getReferenceFrame(), current_frame_, mvIMUSinceLastKF);

    LOG(WARNING) << "[System] Start two-view BA";

    KeyFrame::Ptr kf0 = mapper_->map_->getKeyFrame(0);
    KeyFrame::Ptr kf1 = mapper_->map_->getKeyFrame(1);

    LOG_ASSERT(kf0 != nullptr && kf1 != nullptr) << "Can not find intial keyframes in map!";

    Optimizer::globleBundleAdjustment(mapper_->map_, 20, 0, true);

    LOG(WARNING) << "[System] End of two-view BA";

    current_frame_->setPose(kf1->pose());
    current_frame_->setRefKeyFrame(kf1);
    reference_keyframe_ = kf1;
    last_keyframe_ = kf1;

    depth_filter_->insertFrame(current_frame_, kf1);

    initializer_->reset();

    return STATUS_INITAL_SUCCEED;
}

System::Status System::tracking()
{
    LOG(WARNING)<<"[SYSTEM] tracking()"<<std::endl;

    bool bMapUpdated = false;
    if(mapper_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        mapper_->SetMapUpdateFlagForTracking(false);
    }
    if(loop_closure_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        loop_closure_->SetMapUpdateFlagForTracking(false);
    }

    //! loop closure need
    if(loop_closure_->update_finish_ == true || mapper_->update_finish_ == true /*|| bMapUpdated*/)
    {
        std::cout<<"VO Fix last_frame_ pose!"<<std::endl;
        KeyFrame::Ptr ref = last_keyframe_;
        SE3d Tlr = last_frame_->Tcw()* ref->beforeUpdate_Tcw_.inverse();
        last_frame_->setTcw( Tlr * ref->Tcw() );
        loop_closure_->update_finish_ = false;
        mapper_->update_finish_ = false;
        //test
        /*
        std::cout<<"ref gba id: "<<ref->GBA_KF_<<std::endl;
        std::cout<<"reference_keyframe_ id: "<<reference_keyframe_->id_<<std::endl;
        std::cout<<"reference_keyframe_ frameid: "<<reference_keyframe_->frame_id_<<std::endl;
        std::cout<<"current_frame_ frameid: "<<current_frame_->id_<<std::endl;
        std::cout<<ref->id_ <<" SE3d: "<<std::endl;
        std::cout<<Tlr.rotationMatrix()<<std::endl<<std::endl;
        std::cout<<Tlr.translation()<<std::endl;
        std::vector<KeyFrame::Ptr > kfs_to = mapper_->map_->getAllKeyFrames();
        bool traj_afterGBA = true;
        if(traj_afterGBA)
        {
            std::sort(kfs_to.begin(),kfs_to.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});
            std::string beforeUpdate_Tcw_ = "/home/jh/temp/ssvo/traj_beforeUpdate_Tcw_.txt";
            std::ofstream f_beforeUpdate_Tcw_;
            f_beforeUpdate_Tcw_.open(beforeUpdate_Tcw_.c_str());
            f_beforeUpdate_Tcw_ << std::fixed;
            for(KeyFrame::Ptr kf:kfs_to)
            {
                Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
                Vector3d t = frame_pose.translation();
                Quaterniond q = frame_pose.unit_quaternion();

                f_beforeUpdate_Tcw_ << std::setprecision(6) << kf->timestamp_ << " "
                                    << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

            }

            Sophus::SE3d frame_pose = last_frame_->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();

            f_beforeUpdate_Tcw_ << std::setprecision(6) << last_frame_->timestamp_ << " "
                                << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

            f_beforeUpdate_Tcw_.close();
            std::cout<<"traj_afterGBA saved!"<<std::endl;

        }
         */
    }

    //todo 位姿优化后如果跟踪状态不连续这列可能需要设置一些跟踪上一个关键帧而不是上一帧，注意一下。。虽然对上一阵的位子进行了调整，但是不能确保是正确的。
    current_frame_->setRefKeyFrame(reference_keyframe_);

    //! track seeds
    depth_filter_->trackFrame(last_frame_, current_frame_);

    //! 设置先验信息
    current_frame_->setPose(last_frame_->pose());
    //! alignment by SE3
    AlignSE3 align;
    sysTrace->startTimer("img_align");
    align.run(last_frame_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);
    sysTrace->stopTimer("img_align");

    //! track local map
    sysTrace->startTimer("feature_reproj");
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);
    sysTrace->stopTimer("feature_reproj");
    sysTrace->log("num_feature_reproj", matches);
    LOG(WARNING) << "[System] Track with " << matches << " points";

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;

    //! motion-only BA
    sysTrace->startTimer("motion_ba");
    Optimizer::motionOnlyBundleAdjustment(current_frame_, false, false, true);
    sysTrace->stopTimer("motion_ba");

    sysTrace->startTimer("per_depth_filter");
    if(createNewKeyFrame())
    {
        depth_filter_->insertFrame(current_frame_, reference_keyframe_);
        mapper_->insertKeyFrame(reference_keyframe_);
    }
    else
    {
        depth_filter_->insertFrame(current_frame_, nullptr);
    }
    sysTrace->stopTimer("per_depth_filter");

    sysTrace->startTimer("light_affine");
    calcLightAffine();
    sysTrace->stopTimer("light_affine");

    //！ save frame pose
    frame_timestamp_buffer_.push_back(current_frame_->timestamp_);
    reference_keyframe_buffer_.push_back(current_frame_->getRefKeyFrame());
    frame_pose_buffer_.push_back(current_frame_->pose());//current_frame_->getRefKeyFrame()->Tcw() * current_frame_->pose());

    return STATUS_TRACKING_GOOD;
}

System::Status System::relocalize()
{
    std::cout<<"Lost!!!"<<std::endl;
    std::abort();

    Corners corners_new;
    Corners corners_old;
    fast_detector_->detect(current_frame_->images(), corners_new, corners_old, Config::minCornersPerKeyFrame());

    reference_keyframe_ = mapper_->relocalizeByDBoW(current_frame_, corners_new);

    if(reference_keyframe_ == nullptr)
        return STATUS_TRACKING_BAD;

    current_frame_->setPose(reference_keyframe_->pose());

    //! alignment by SE3
    AlignSE3 align;
    int matches = align.run(reference_keyframe_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    current_frame_->setRefKeyFrame(reference_keyframe_);
    matches = feature_tracker_->reprojectLoaclMap(current_frame_);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    Optimizer::motionOnlyBundleAdjustment(current_frame_, false, true, true);

    if(current_frame_->featureNumber() < 30)
        return STATUS_TRACKING_BAD;

    return STATUS_TRACKING_GOOD;
}

void System::calcLightAffine()
{
    std::vector<Feature::Ptr> fts_last = last_frame_->getFeatures();

    const cv::Mat img_last = last_frame_->getImage(0);
    const cv::Mat img_curr = current_frame_->getImage(0).clone() * 1.3;

    const int size = 4;
    const int patch_area = size*size;
    const int N = (int)fts_last.size();
    cv::Mat patch_buffer_last = cv::Mat::zeros(N, patch_area, CV_32FC1);
    cv::Mat patch_buffer_curr = cv::Mat::zeros(N, patch_area, CV_32FC1);

    int count = 0;
    for(int i = 0; i < N; ++i)
    {
        const Feature::Ptr ft_last = fts_last[i];
        const Feature::Ptr ft_curr = current_frame_->getFeatureByMapPoint(ft_last->mpt_);

        if(ft_curr == nullptr)
            continue;

        utils::interpolateMat<uchar, float, size>(img_last, patch_buffer_last.ptr<float>(count), ft_last->px_[0], ft_last->px_[1]);
        utils::interpolateMat<uchar, float, size>(img_curr, patch_buffer_curr.ptr<float>(count), ft_curr->px_[0], ft_curr->px_[1]);

        count++;
    }

    patch_buffer_last.resize(count);
    patch_buffer_curr.resize(count);

    if(count < 20)
    {
        Frame::light_affine_a_ = 1;
        Frame::light_affine_b_ = 0;
        return;
    }

    float a=1;
    float b=0;
    calculateLightAffine(patch_buffer_last, patch_buffer_curr, a, b);
    Frame::light_affine_a_ = a;
    Frame::light_affine_b_ = b;

//    std::cout << "a: " << a << " b: " << b << std::endl;
}

bool System::createNewKeyFrame()
{
    if(mapper_->isRequiredStop())
        return false;

    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();

    std::vector<Feature::Ptr> fts = current_frame_->getFeatures();
    std::map<MapPoint::Ptr, Feature::Ptr> mpt_ft;
    for(const Feature::Ptr &ft : fts)
    {
        mpt_ft.emplace(ft->mpt_, ft);
    }

    KeyFrame::Ptr max_overlap_keyframe;
    int max_overlap = 0;
    for(const auto &olp_kf : overlap_kfs)
    {
        if(olp_kf.second < max_overlap || (olp_kf.second == max_overlap && olp_kf.first->id_ < max_overlap_keyframe->id_))
            continue;

        max_overlap_keyframe = olp_kf.first;
        max_overlap = olp_kf.second;
    }

    //! check distance
    bool c1 = true;
    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);
//    for(const auto &ovlp_kf : overlap_kfs)
//    {
//        SE3d T_cur_from_ref = current_frame_->Tcw() * ovlp_kf.first->pose();
//        Vector3d tran = T_cur_from_ref.translation();
//        double dist1 = tran.dot(tran);
//        double dist2 = 0.1 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
//        double dist = dist1 + dist2;
////        std::cout << "d1: " << dist1 << ". d2: " << dist2 << std::endl;
//        if(dist  < 0.10 * median_depth)
//        {
//            c1 = false;
//            break;
//        }
//    }

    SE3d T_cur_from_ref = current_frame_->Tcw() * last_keyframe_->pose();
    Vector3d tran = T_cur_from_ref.translation();
    double dist1 = tran.dot(tran);
    double dist2 = 0.01 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();

    double depth_th = 0.01;
    if(!mapper_->GetVINSInited())
    {
        depth_th /= 2;
    }

    if(dist1+dist2  < depth_th * median_depth)
        c1 = false;

    //! check disparity
    std::list<float> disparities;
    const int threahold = int (max_overlap * 0.6);
    for(const auto &ovlp_kf : overlap_kfs)
    {
        if(ovlp_kf.second < threahold)
            continue;

        std::vector<float> disparity;
        disparity.reserve(ovlp_kf.second);
        std::vector<MapPoint::Ptr> mpts = ovlp_kf.first->getMapPoints();
        for(const MapPoint::Ptr &mpt : mpts)
        {
            Feature::Ptr ft_ref = mpt->findObservation(ovlp_kf.first);
            if(ft_ref == nullptr) continue;

            if(!mpt_ft.count(mpt)) continue;
            Feature::Ptr ft_cur = mpt_ft.find(mpt)->second;

            const Vector2d px(ft_ref->px_ - ft_cur->px_);
            disparity.push_back(px.norm());
        }

        std::sort(disparity.begin(), disparity.end());
        float disp = disparity.at(disparity.size()/2);
        disparities.push_back(disp);
    }
    disparities.sort();

    if(!disparities.empty())
        current_frame_->disparity_ = *std::next(disparities.begin(), disparities.size()/2);

    LOG(INFO) << "[System] Max overlap: " << max_overlap << " min disaprity " << disparities.front() << ", median: " << current_frame_->disparity_;

//    int all_features = current_frame_->featureNumber() + current_frame_->seedNumber();
    bool c2 = disparities.front() > options_.min_kf_disparity;
    bool c3 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * options_.min_ref_track_rate;
//    bool c4 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * 0.9;

    //! create new keyFrame
    if(c1 && (c2 || c3))
    {
        //! create new keyframe
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_,mvIMUSinceLastKF,last_keyframe_);

        //! Set initial NavState for KeyFrame TODO 但是这里的普通帧的NavState是什么时候设置的？
        new_keyframe->SetInitialNavStateAndBias(current_frame_->GetNavState());
        //! 通过刚刚设置gyr 和 acc的状态 Compute pre-integrator
        new_keyframe->ComputePreInt();
        //! Clear IMUData buffer
        mvIMUSinceLastKF.clear();


        for(const Feature::Ptr &ft : fts)
        {
            if(ft->mpt_->isBad())
            {
                current_frame_->removeFeature(ft);
                continue;
            }

            ft->mpt_->addObservation(new_keyframe, ft);
            ft->mpt_->updateViewAndDepth();
//            mapper_->addOptimalizeMapPoint(ft->mpt_);
        }
        new_keyframe->updateConnections();
        reference_keyframe_ = new_keyframe;
        last_keyframe_ = new_keyframe;
//        LOG(ERROR) << "C: (" << c1 << ", " << c2 << ", " << c3 << ") cur_n: " << current_frame_->N() << " ck: " << reference_keyframe_->N();
        return true;
    }
        //! change reference keyframe
    else
    {
        if(overlap_kfs[reference_keyframe_] < max_overlap * 0.85)
            reference_keyframe_ = max_overlap_keyframe;
        return false;
    }
}

void System::finishFrame()
{
    sysTrace->startTimer("finish");
    cv::Mat image_show;
//    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_BAD == status_)
        {
            stage_ = STAGE_RELOCALIZING;
            current_frame_->setPose(last_frame_->pose());
        }
    }
    else if(STAGE_INITALIZE == stage_)
    {
        if(STATUS_INITAL_SUCCEED == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else if(STATUS_INITAL_RESET == status_)
            initializer_->reset();

        mvIMUSinceLastKF.clear();
        initializer_->drowOpticalFlow(image_show);
    }
    else if(STAGE_RELOCALIZING == stage_)
    {
        if(STATUS_TRACKING_GOOD == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else
            current_frame_->setPose(last_frame_->pose());
    }

    //! update
    last_frame_ = current_frame_;

    //! display
    viewer_->setCurrentFrame(current_frame_, image_show);

    sysTrace->log("stage", stage_);
    sysTrace->stopTimer("finish");
    sysTrace->stopTimer("total");
    const double time = sysTrace->getTimer("total");
    LOG(WARNING) << "[System] Finish Current Frame with Stage: " << stage_ << ", total time: " << time;

    sysTrace->writeToFile();

}

void System::saveTrajectoryTUM(const std::string &file_name)
{
    std::cout<<"Begin save trajectory.";
    LOG(INFO) << "Begin save trajectory.";
    std::ofstream f;
    f.open(file_name.c_str());
    f << std::fixed;

    std::list<double>::iterator frame_timestamp_ptr = frame_timestamp_buffer_.begin();
    std::list<Sophus::SE3d>::iterator frame_pose_ptr = frame_pose_buffer_.begin();
    std::list<KeyFrame::Ptr>::iterator reference_keyframe_ptr = reference_keyframe_buffer_.begin();
    const std::list<double>::iterator frame_timestamp = frame_timestamp_buffer_.end();
    for(; frame_timestamp_ptr!= frame_timestamp; frame_timestamp_ptr++, frame_pose_ptr++, reference_keyframe_ptr++)
    {
        Sophus::SE3d frame_pose = (*frame_pose_ptr);//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        f << std::setprecision(6) << *frame_timestamp_ptr << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    LOG(INFO) << " trajectory saved!";

    std::string KFfilename = "KF" + file_name;

    f.open(KFfilename.c_str());
    f << std::fixed;

    std::vector<KeyFrame::Ptr> kfs = mapper_->map_->getAllKeyFrames();
    std::sort(kfs.begin(),kfs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});

    for(auto kf:kfs)
    {
        Sophus::SE3d frame_pose = kf->pose();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        f << std::setprecision(6) << kf->timestamp_ << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    LOG(INFO) << " KFtrajectory saved!";

}


//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------

void System::process(const cv::Mat &image, const double timestamp, const std::vector<ssvo::IMUData> &vimu)
{
    //将两个关键帧之间的imu数据保存下来
    mvIMUSinceLastKF.insert(mvIMUSinceLastKF.end(), vimu.begin(),vimu.end());

    sysTrace->startTimer("total");
    sysTrace->startTimer("frame_create");
    //! get gray image
    double t0 = (double)cv::getTickCount();
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    //构造帧，保存imu数据
    current_frame_ = Frame::create(gray, timestamp, camera_,vimu);

    if(last_frame_)
        current_frame_->SetInitialNavStateAndBias(last_frame_->GetNavState());

    double t1 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Frame " << current_frame_->id_ << " create time: " << (t1-t0)/cv::getTickFrequency();
    sysTrace->log("frame_id", current_frame_->id_);
    sysTrace->stopTimer("frame_create");

    sysTrace->startTimer("processing");

    if(STAGE_NORMAL_FRAME == stage_)
    {
        std::unique_lock<std::mutex> lock(mapper_->map_->mutex_update_);
        if(mapper_->GetVINSInited())
            status_ = trackingVIO();
        else
            status_ = tracking();

    }
    else if(STAGE_INITALIZE == stage_)
    {
        status_ = initialize();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {
        //todo 这部分还完全没处理
        status_ = relocalize();
    }
    sysTrace->stopTimer("processing");

    finishFrame();

}

System::Status System::trackingVIO()
{
    LOG(WARNING)<<"[SYSTEM] tracking()"<<std::endl;

    bool bMapUpdated = false;
    if(mapper_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        mapper_->SetMapUpdateFlagForTracking(false);
    }
    if(loop_closure_->GetMapUpdateFlagForTracking())
    {
        bMapUpdated = true;
        loop_closure_->SetMapUpdateFlagForTracking(false);
    }

    //! 有了bMapUpdated变量之后这个应该没什么用了
    if(loop_closure_->update_finish_ == true || mapper_->update_finish_ == true)
    {
        std::cout<<"VIO Fix last_frame_ pose!"<<std::endl;
        KeyFrame::Ptr ref = /*last_frame_->getRefKeyFrame()*/last_keyframe_;
        SE3d Tlr = last_frame_->Tcw()* ref->beforeUpdate_Tcw_.inverse();
        last_frame_->setTcw( Tlr * ref->Tcw() );
        loop_closure_->update_finish_ = false;
        mapper_->update_finish_ = false;
        //test
        /*
        std::cout<<"ref gba id: "<<ref->GBA_KF_<<std::endl;
        std::cout<<"reference_keyframe_ id: "<<reference_keyframe_->id_<<std::endl;
        std::cout<<"reference_keyframe_ frameid: "<<reference_keyframe_->frame_id_<<std::endl;
        std::cout<<"current_frame_ frameid: "<<current_frame_->id_<<std::endl;
        std::cout<<ref->id_ <<" SE3d: "<<std::endl;
        std::cout<<Tlr.rotationMatrix()<<std::endl<<std::endl;
        std::cout<<Tlr.translation()<<std::endl;
        std::vector<KeyFrame::Ptr > kfs_to = mapper_->map_->getAllKeyFrames();
        bool traj_afterGBA = true;
        if(traj_afterGBA)
        {
            std::sort(kfs_to.begin(),kfs_to.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->timestamp_<kf2->timestamp_;});
            std::string beforeUpdate_Tcw_ = "/home/jh/temp/ssvo/traj_beforeUpdate_Tcw_.txt";
            std::ofstream f_beforeUpdate_Tcw_;
            f_beforeUpdate_Tcw_.open(beforeUpdate_Tcw_.c_str());
            f_beforeUpdate_Tcw_ << std::fixed;
            for(KeyFrame::Ptr kf:kfs_to)
            {
                Sophus::SE3d frame_pose = kf->beforeUpdate_Tcw_.inverse();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
                Vector3d t = frame_pose.translation();
                Quaterniond q = frame_pose.unit_quaternion();

                f_beforeUpdate_Tcw_ << std::setprecision(6) << kf->timestamp_ << " "
                               << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

            }

            Sophus::SE3d frame_pose = last_frame_->Tcw().inverse();//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
            Vector3d t = frame_pose.translation();
            Quaterniond q = frame_pose.unit_quaternion();

            f_beforeUpdate_Tcw_ << std::setprecision(6) << last_frame_->timestamp_ << " "
                                << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;

            f_beforeUpdate_Tcw_.close();
            std::cout<<"traj_afterGBA saved!"<<std::endl;

        }
         */
    }

    current_frame_->setRefKeyFrame(reference_keyframe_);

    //! track seeds
    depth_filter_->trackFrame(last_frame_, current_frame_);

    PredictNavStateByIMU(bMapUpdated);

    SE3d Tlr = current_frame_->Tcw()* last_frame_->Twc();
    std::cout<<Tlr.rotationMatrix()<<std::endl<<std::endl;
    std::cout<<Tlr.translation()<<std::endl;
    std::abort();


    //! alignment by SE3 ，得到准确的位姿
    AlignSE3 align;
    sysTrace->startTimer("img_align");
    if(bMapUpdated)
        align.run(last_keyframe_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);
    else
        align.run(last_frame_, current_frame_, Config::alignTopLevel(), Config::alignBottomLevel(), 30, 1e-8);
    sysTrace->stopTimer("img_align");

    //! track local map
    sysTrace->startTimer("feature_reproj");
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);
    sysTrace->stopTimer("feature_reproj");
    sysTrace->log("num_feature_reproj", matches);
    LOG(WARNING) << "[System] Track with " << matches << " points";

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;

    //! motion-only BA
    sysTrace->startTimer("motion_ba");
    if(mapper_->GetFirstVINSInited() || bMapUpdated)
        Optimizer::PoseOptimization(current_frame_,last_keyframe_,mIMUPreIntInTrack,mapper_->GetGravityVec(),false);
    else
        Optimizer::PoseOptimization(current_frame_,last_frame_,mIMUPreIntInTrack,mapper_->GetGravityVec(),false);
//    Optimizer::motionOnlyBundleAdjustment(current_frame_, false, false, true);
    sysTrace->stopTimer("motion_ba");

    if(mapper_->GetFirstVINSInited())
    {
        mapper_->SetFirstVINSInited(false);
    }

    sysTrace->startTimer("per_depth_filter");
    if(createNewKeyFrame())
    {
        depth_filter_->insertFrame(current_frame_, reference_keyframe_);
        mapper_->insertKeyFrame(reference_keyframe_);
    }
    else
    {
        depth_filter_->insertFrame(current_frame_, nullptr);
    }
    sysTrace->stopTimer("per_depth_filter");

    sysTrace->startTimer("light_affine");
    calcLightAffine();
    sysTrace->stopTimer("light_affine");

    //！ save frame pose
    frame_timestamp_buffer_.push_back(current_frame_->timestamp_);
    reference_keyframe_buffer_.push_back(current_frame_->getRefKeyFrame());
    frame_pose_buffer_.push_back(current_frame_->pose());//current_frame_->getRefKeyFrame()->Tcw() * current_frame_->pose());

    return STATUS_TRACKING_GOOD;
}
    
    
    
void System::PredictNavStateByIMU(bool bMapUpdated) 
{
    LOG_ASSERT(mapper_->GetVINSInited())<<"mapper_->GetVINSInited() not, shouldn't in PredictNavStateByIMU"<<std::endl;

    //! 如果局部BA或全局BA导致位姿发生了跳变，那么就要跟踪关键帧来计算初始位姿了 Map updated, optimize with last KeyFrame
//    if(mapper_->GetFirstVINSInited() || bMapUpdated)
//    {
//        // Compute IMU Pre-integration
//        mIMUPreIntInTrack = GetIMUPreIntSinceLastKF(current_frame_, last_keyframe_, mvIMUSinceLastKF);
//        Vector3d dp = mIMUPreIntInTrack.getDeltaP();
//        Vector3d dv = mIMUPreIntInTrack.getDeltaV();
//        Matrix3d dr = mIMUPreIntInTrack.getDeltaR();
//
//        double dt = mIMUPreIntInTrack.getDeltaTime();
//
//        Vector3d init_p = last_keyframe_->GetNavState().Get_P();
//        Vector3d init_v = last_keyframe_->GetNavState().Get_V();
//        Matrix3d init_r = last_keyframe_->GetNavState().Get_RotMatrix();
//
//
//        // Get initial NavState&pose from Last KeyFrame
//        current_frame_->SetInitialNavStateAndBias(last_keyframe_->GetNavState());
//        current_frame_->UpdateNavState(mIMUPreIntInTrack,mapper_->GetGravityVec());
//        current_frame_->UpdatePoseFromNS(ImuConfigParam::GetEigTbc());
//
//        Vector3d p = current_frame_->GetNavState().Get_P();
//        Vector3d v = current_frame_->GetNavState().Get_V();
//        Matrix3d r = current_frame_->GetNavState().Get_RotMatrix();
//
//    }
//    else
//    {
        // Compute IMU Pre-integration
        mIMUPreIntInTrack = GetIMUPreIntSinceLastFrame(current_frame_, last_frame_);
        // Get initial pose from Last Frame
        current_frame_->UpdateNavState(mIMUPreIntInTrack,mapper_->GetGravityVec());
        current_frame_->UpdatePoseFromNS(ImuConfigParam::GetEigTbc());
//    }
}

IMUPreintegrator System::GetIMUPreIntSinceLastKF(Frame::Ptr pCurF, KeyFrame::Ptr pLastKF, const std::vector<IMUData>& vIMUSInceLastKF)
{
    // Reset pre-integrator first
    IMUPreintegrator IMUPreInt;
    IMUPreInt.reset();

    Vector3d bg = pLastKF->GetNavState().Get_BiasGyr();
    Vector3d ba = pLastKF->GetNavState().Get_BiasAcc();

    // remember to consider the gap between the last KF and the first IMU
    {
        const IMUData& imu = vIMUSInceLastKF.front();
        double dt = imu._t - pLastKF->timestamp_;
        LOG_ASSERT(dt>-1e-5)<<"dt is '-', please check";

        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log
//        if(dt < 0)
//        {
//            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this KF vs last imu time: "<<pLastKF->timestamp_<<" vs "<<imu._t<<std::endl;
//            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
//        }
    }
    // integrate each imu
    for(size_t i=0; i<vIMUSInceLastKF.size(); i++)
    {
        const IMUData& imu = vIMUSInceLastKF[i];
        double nextt;
        if(i==vIMUSInceLastKF.size()-1)
            nextt = pCurF->timestamp_;         // last IMU, next is this KeyFrame
        else
            nextt = vIMUSInceLastKF[i+1]._t;  // regular condition, next is imu data

        // delta time
        double dt = nextt - imu._t;
        LOG_ASSERT(dt>-1e-5)<<"dt is '-', please check";

        // update pre-integrator
        IMUPreInt.update(imu._g - bg, imu._a - ba, dt);

        // Test log
//        if(dt <= 1e-8)
//        {
//            std::cerr<<std::fixed<<std::setprecision(3)<<"dt = "<<dt<<", this vs next time: "<<imu._t<<" vs "<<nextt<<std::endl;
//            std::cerr.unsetf ( std::ios::showbase );                // deactivate showbase
//        }
    }

    return IMUPreInt;
}

IMUPreintegrator System::GetIMUPreIntSinceLastFrame(Frame::Ptr pCurF, Frame::Ptr pLastF)
{
    // Reset pre-integrator first
    IMUPreintegrator IMUPreInt;
    IMUPreInt.reset();

    pCurF->ComputeIMUPreIntSinceLastFrame(pLastF,IMUPreInt);

    return IMUPreInt;
}

}

