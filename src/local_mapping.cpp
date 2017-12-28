#include <include/config.hpp>
#include "local_mapping.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace ssvo{

void showMatch(const cv::Mat &src, const cv::Mat &dst, const Vector2d &epl0, const Vector2d &epl1, const Vector2d &px_src, const Vector2d &px_dst)
{
    cv::Mat src_show = src.clone();
    cv::Mat dst_show = dst.clone();
    if(src_show.channels() == 1)
        cv::cvtColor(src_show, src_show, CV_GRAY2RGB);
    if(dst_show.channels() == 1)
        cv::cvtColor(dst_show, dst_show, CV_GRAY2RGB);

    cv::Point2d p0(epl0[0], epl0[1]);
    cv::Point2d p1(epl1[0], epl1[1]);
    cv::Point2d p_src(px_src[0], px_src[1]);
    cv::Point2d p_dst(px_dst[0], px_dst[1]);

    cv::line(dst_show, p0, p1, cv::Scalar(0, 255, 0), 1);
    cv::circle(dst_show, p1, 3, cv::Scalar(0,255,0));
    cv::circle(src_show, p_src, 5, cv::Scalar(255,0,0));
    cv::circle(dst_show, p_dst, 5, cv::Scalar(255,0,0));

    cv::imshow("cur", src_show);
    cv::imshow("dst", dst_show);
    cv::waitKey(0);
}


//! Seed
int Seed::seed_counter = 0;

Seed::Seed(Feature::Ptr ft, double depth_mean, double depth_min) :
    id(seed_counter++),
    ft(ft),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{}

double Seed::computeTau(
    const SE3d& T_ref_cur,
    const Vector3d& f,
    const double z,
    const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(-a.dot(t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

void Seed::update(const double x, const double tau2)
{
    double norm_scale = sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;

    double s2 = 1./(1./sigma2 + 1./tau2);
    double m = s2*(mu/sigma2 + x/tau2);
    double C1 = a/(a+b) * utils::normal_distribution<double>(x, mu, norm_scale);
    double C2 = b/(a+b) * 1./z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    double e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));

    // update parameters
    double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e-f)/(f-e/f);
    b = a*(1.0f-f)/f;
}

//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report, bool verbose) :
    fast_detector_(fast_detector), delay_(static_cast<int>(1000.0/fps)), report_(report), verbose_(report&&verbose)
{
    map_ = Map::create();
}

void LocalMapper::startThread()
{
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::stopThread()
{
    if(mapping_thread_ != nullptr)
    {
        setStop();
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

void LocalMapper::run()
{
    while(!isRequiredStop())
    {
        if(!checkNewFrame())
            continue;

        processNewFrame();

        processNewKeyFrame();
    }
}

void LocalMapper::insertNewFrame(Frame::Ptr frame, KeyFrame::Ptr keyframe, double mean_depth, double min_depth)
{
    map_->insertKeyFrame(keyframe);
    keyframe->updateConnections();

    LOG_ASSERT(frame != nullptr) << "Error input! Frame should not be null!";
    if(mapping_thread_ != nullptr)
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);

        frames_buffer_.emplace_back(frame, keyframe);
        depth_buffer_.emplace_back(mean_depth, min_depth);
        cond_process_.notify_one();
    }
    else
    {
        current_frame_ = std::make_pair(frame, keyframe);
        current_depth_ = std::make_pair(mean_depth, min_depth);

        processNewFrame();
        processNewKeyFrame();
    }
}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = ssvo::KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = ssvo::KeyFrame::create(frame_cur);

    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    LOG_ASSERT(N == points.size()) << "Error in create inital map! Two frames' features is not matched mappoints!";
    for(size_t i = 0; i < N; i++)
    {
        MapPoint::Ptr mpt = ssvo::MapPoint::create(points[i], keyframe_ref);

        fts_ref[i]->mpt = mpt;
        fts_cur[i]->mpt = mpt;

        map_->insertMapPoint(mpt);

        mpt->addObservation(keyframe_ref, fts_ref[i]);
        mpt->addObservation(keyframe_cur, fts_cur[i]);
        mpt->updateViewAndDepth();
    }

    Vector2d mean_depth, min_depth;
    keyframe_ref->getSceneDepth(mean_depth[0], min_depth[0]);
    keyframe_cur->getSceneDepth(mean_depth[1], min_depth[1]);
    this->insertNewFrame(frame_ref, keyframe_ref, mean_depth[0], min_depth[0]);
    this->insertNewFrame(frame_cur, keyframe_cur, mean_depth[1], min_depth[1]);

    LOG_IF(INFO, report_) << "[Mapping] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

bool LocalMapper::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        cond_process_.wait_for(lock, std::chrono::milliseconds(delay_));
        if(frames_buffer_.empty())
            return false;

        current_frame_ = frames_buffer_.front();
        current_depth_ = depth_buffer_.front();
        frames_buffer_.pop_front();
        depth_buffer_.pop_front();
    }

    return true;
}

bool LocalMapper::processNewKeyFrame()
{
    if(current_frame_.second == nullptr)
        return false;

    const KeyFrame::Ptr &kf = current_frame_.second;
    const double mean_depth = current_depth_.first;
    const double min_depth = current_depth_.second;
    std::vector<Feature::Ptr> fts = kf->getFeatures();

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    Corners new_corners;
    fast_detector_->detect(kf->image(), new_corners, old_corners, 150);

    Seeds new_seeds;
    for(const Corner &corner : new_corners)
    {
        const Vector2d px(corner.x, corner.y);
        const Vector3d fn(kf->cam_->lift(px));
        Feature::Ptr ft = Feature::create(px, fn, corner.level, nullptr);
        new_seeds.emplace_back(Seed(ft, mean_depth, min_depth*0.5));
    }
    seeds_buffer_.emplace_back(kf, std::make_shared<Seeds>(new_seeds));

    LOG_IF(WARNING, report_) << "[Mapping] Add new keyframe " << kf->id_ << " with " << new_seeds.size() << " seeds";

    return true;
}

bool LocalMapper::processNewFrame()
{
    if(current_frame_.first == nullptr)
        return false;

    const Frame::Ptr &frame = current_frame_.first;
    for(auto &seeds_pair : seeds_buffer_)
    {
        const KeyFrame::Ptr keyframe = seeds_pair.first;
        Seeds seeds = *seeds_pair.second;

        // TODO old seeds_pair remove

        SE3d T_cur_from_ref = frame->Tcw() * keyframe->pose();
        for(auto it = seeds.begin(); it!=seeds.end(); it++)
        {
            double depth;
            findEpipolarMatch(*it, keyframe, frame, T_cur_from_ref, depth);


        }
    }

    return true;
}

bool LocalMapper::findEpipolarMatch(const Seed &seed,
                                    const KeyFrame::Ptr &keyframe,
                                    const Frame::Ptr &frame,
                                    const SE3d &T_cur_from_ref,
                                    double &depth)
{
    static const int patch_size = Align2DI::PatchSize;
    static const int patch_area = Align2DI::PatchArea;
    static const int half_patch_size = Align2DI::HalfPatchSize;

    //! check if in the view of current frame
    const double z_ref = 1.0/seed.mu;
    const Vector3d xyz_cur(T_cur_from_ref * (seed.ft->fn * z_ref ));
    const double z_cur = xyz_cur[2];
    if(z_cur < 0.00f)
        return false;

    const Vector2d px_cur(frame->cam_->project(xyz_cur));
    if(!frame->cam_->isInFrame(px_cur.cast<int>(), half_patch_size))
        return false;

    //! d - inverse depth, z - depth
    const double sigma = sqrt(seed.sigma2);
    const double d_max = seed.mu + sigma;
    const double d_min = MAX(seed.mu - sigma, 0.00000001f);
    const double z_min = 1.0/d_max;
    const double z_max = 1.0/d_min;

    //! on unit plane in cur frame
    Vector3d xyz_near = T_cur_from_ref * (seed.ft->fn * z_min);
    Vector3d xyz_far  = T_cur_from_ref * (seed.ft->fn * z_max);
    xyz_near /= xyz_near[2];
    xyz_far  /= xyz_far[2];
    Vector2d epl_dir = (xyz_near - xyz_far).head<2>();

    //! calculate best search level
    const int level_ref = seed.ft->level;
    const int level_cur = MapPoint::predictScale(z_ref, z_cur, level_ref, frame->nlevels_-1);
    const double factor = static_cast<double>(1 << level_cur);
    //! L * (1 << level_ref) / L' * (1 << level_cur) = z_ref / z_cur
//    const double scale = z_cur / z_ref * (1 << level_ref) / (1 << level_cur);

    //! px in image plane
    Vector2d px_near = frame->cam_->project(xyz_near) / factor;
    Vector2d px_far  = frame->cam_->project(xyz_far) / factor;
    if(!frame->cam_->isInFrame(px_far.cast<int>(), half_patch_size, level_cur) ||
       !frame->cam_->isInFrame(px_near.cast<int>(), half_patch_size, level_cur))
        return false;

    //! get warp patch
    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(keyframe->cam_, frame->cam_, seed.ft->px, seed.ft->fn, level_ref,
                               z_ref, T_cur_from_ref, patch_size, A_cur_from_ref);

    static const int patch_border_size = patch_size+2;
    cv::Mat image_ref = keyframe->getImage(level_ref);
    Matrix<double, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<double, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                 seed.ft->px, level_ref, level_cur);

    Matrix<double, patch_size, patch_size, RowMajor> patch;
    patch = patch_with_border.block(1, 1, patch_size, patch_size);

    const cv::Mat image_cur = frame->getImage(level_cur);
    Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> > image_cur_eigen((uchar*)image_cur.data, image_cur.rows, image_cur.cols);
    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > patch_eigen(patch.data(), patch_area, 1);

    Vector2d px_best(-1,-1);
    double epl_length = (px_near-px_far).norm();
    if(epl_length > 2.0)
    {
        int n_steps = epl_length / 0.707;
        Vector2d step = epl_dir / n_steps;

        // TODO check epl max length

        // TODO 使用模板来加速！！！
        //! SSD
        double t0 = (double)cv::getTickCount();
        ZSSD<double, patch_area> zssd(patch_eigen);
        double score_best = std::numeric_limits<double>::max();
        double score_second = score_best;
        int index_best = -1;
        int index_second = -1;

        Vector2d fn_start = xyz_far.head<2>() - step * 2;
        n_steps += 2;
        Vector2d fn(fn_start);
        for(int i = 0; i < n_steps; ++i, fn += step) {
            Vector2d px(frame->cam_->project(fn) / factor);

            //! always in frame's view
            if(!frame->cam_->isInFrame(px.cast<int>(), half_patch_size, level_cur))
                continue;

            Matrix<double, patch_area, 1> patch_cur;
            utils::interpolateMat<uchar, double, patch_size>(image_cur_eigen, patch_cur, px[0], px[1]);

            double score = zssd.compute_score(patch_cur);

            if(score < score_best) {
                score_best = score;
                index_best = i;
            } else if(score < score_second) {
                score_second = score;
                index_second = i;
            }
        }

        if(score_best > 0.8 * score_second && std::abs(index_best - index_second) > 2)
            return false;

        if(score_best > zssd.threshold())
            return false;

        Vector2d pn_best = fn_start + index_best * step;
        px_best = frame->cam_->project(pn_best) / factor;

        double t1 = (double)cv::getTickCount();
        double time = (t1-t0)/cv::getTickFrequency();
        LOG_IF(INFO, report_) << "Id" << seed.id << " Step: " << n_steps << " T:" << time << "(" << time/n_steps << ")"
                              << " best: [" << index_best << ", " << score_best<< "]"
                              << " second: [" << index_second << ", " << score_second<< "]";
    }
    else
    {
        px_best = (px_near + px_far) * 0.5;
    }

    Matrix<double, patch_size, patch_size, RowMajor> dx, dy;
    dx = 0.5*(patch_with_border.block(1, 2, patch_size, patch_size)
        - patch_with_border.block(1, 0, patch_size, patch_size));
    dy = 0.5*(patch_with_border.block(2, 1, patch_size, patch_size)
        - patch_with_border.block(0, 1, patch_size, patch_size));

    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dx_eigen(dx.data(), patch_area, 1);
    Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dy_eigen(dy.data(), patch_area, 1);

    Align2DI matcher(verbose_);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_best;
    if(!matcher.run(image_cur_eigen, patch_eigen, dx_eigen, dy_eigen, estimate))
        return false;

    // TODO 增加极点距离的检查？
    //! check distance to epl, incase of the aligen draft
    Vector2d a = estimate.head<2>() - px_far;
    Vector2d b = px_near - px_far;
    double lacos = a.dot(b) / b.norm();
    double dist2 = a.squaredNorm() - lacos*lacos;
    if(dist2 > 16) //! 4^2
        return false;

    Vector2d px_matched = estimate.head<2>() * factor;

    LOG_IF(INFO, verbose_) << "Found! [" << seed.ft->px.transpose() << "] "
                           << "dst: [" << px_matched.transpose() << "] "
                           << "epl: [" << px_near.transpose() << "]--[" << px_far.transpose() << "]" << std::endl;

//    showMatch(image_ref, image_cur, px_near, px_far, seed.ft->px/factor, px_matched/factor);

    return true;
}


}