#include <iomanip>
#include "optimizer.hpp"
#include "config.hpp"
#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <string>

namespace ssvo{


cv::Mat showMatch_op(const cv::Mat& img1,const cv::Mat& img2,const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2)
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

bool compute_residual(const Sophus::Sim3d& camera, const double* const point, double obs_x, double obs_y, double* residuals, KeyFrame::Ptr pKF1)
{
    Vector3d Mp_cam(point[0], point[1], point[2]);

    Sophus::Sim3d Sim3_k12 = camera;

//    Eigen::Map<Sophus::Sim3d const> const Sim3_k12(camera);

    Mp_cam = Sim3_k12 * Mp_cam;

    Vector2d px = pKF1->cam_->project(Mp_cam);

    residuals[0] = px[0] - obs_x;
    residuals[1] = px[1] - obs_y;

    return true;
}

bool compute_residualInv(const Sophus::Sim3d& camera, const double* const point, double obs_x, double obs_y, double* residuals, KeyFrame::Ptr pKF2)
{
    Vector3d Mp_cam;
    Mp_cam <<point[0], point[1], point[2];

    Sophus::Sim3d Sim3_k12 = camera;
    Mp_cam = Sim3_k12.inverse() * Mp_cam;
    Vector2d px = pKF2->cam_->project(Mp_cam);

    residuals[0] = px[0] - obs_x;
    residuals[1] = px[1] - obs_y;

    return true;
}

void Optimizer::globleBundleAdjustment(const Map::Ptr &map, int max_iters,const uint64_t nLoopKF, bool report, bool verbose)
{
    if (map->KeyFramesInMap() < 2)
        return;

    std::vector<KeyFrame::Ptr> all_kfs = map->getAllKeyFrames();
    std::vector<MapPoint::Ptr> all_mpts = map->getAllMapPoints();

    static double focus_length = MIN(all_kfs.back()->cam_->fx(), all_kfs.back()->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma() / focus_length;

    ceres::Problem problem;

    for (const KeyFrame::Ptr &kf : all_kfs)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        if(kf->id_ == 0)
            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    double scale = pixel_usigma * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
    for (const MapPoint::Ptr &mpt : all_mpts)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for (const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_num_iterations = max_iters;
//    options_.gradient_tolerance = 1e-4;
//    options_.function_tolerance = 1e-4;
    //options_.max_solver_time_in_seconds = 0.2;

    ceres::Solve(options, &problem, &summary);

    std::cout<<"globleBundleAdjustment FullReport()"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

    //! update pose
    if((int)nLoopKF ==0 )
    {
        std::for_each(all_kfs.begin(), all_kfs.end(), [](KeyFrame::Ptr kf) {kf->setTcw(kf->optimal_Tcw_); });
        std::for_each(all_mpts.begin(), all_mpts.end(), [](MapPoint::Ptr mpt){mpt->setPose(mpt->optimal_pose_);});
    }
    else
    {
        //! set flag
    for(auto kf:all_kfs)
    {
        kf->GBA_KF_ = nLoopKF;kf->beforeGBA_Tcw_ = kf->Tcw();
    }
    for(auto mpt:all_mpts)
    {
        mpt->GBA_KF_ = nLoopKF;
    }

    }

    //! Report
    reportInfo<2>(problem, summary, report, verbose);
}

void Optimizer::localBundleAdjustment(const KeyFrame::Ptr &keyframe, std::list<MapPoint::Ptr> &bad_mpts, int size, int min_shared_fts, bool report, bool verbose)
{
    static double focus_length = MIN(keyframe->cam_->fx(), keyframe->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma()/focus_length;

    double t0 = (double)cv::getTickCount();
    size = size > 0 ? size-1 : 0;
    std::set<KeyFrame::Ptr> actived_keyframes = keyframe->getConnectedKeyFrames(size, min_shared_fts);
    actived_keyframes.insert(keyframe);
    std::unordered_set<MapPoint::Ptr> local_mappoints;
    std::set<KeyFrame::Ptr> fixed_keyframe;

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        for(const MapPoint::Ptr &mpt : mpts)
        {
            local_mappoints.insert(mpt);
        }
    }

    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            if(actived_keyframes.count(item.first))
                continue;

            fixed_keyframe.insert(item.first);
        }
    }

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();

    for(const KeyFrame::Ptr &kf : fixed_keyframe)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->optimal_Tcw_ = kf->Tcw();
        problem.AddParameterBlock(kf->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);
        if(kf->id_ <= 1)
            problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    double scale = pixel_usigma * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);
    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for(const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function1, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = report & verbose;

    ceres::Solve(options, &problem, &summary);

    //! update pose
    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->setTcw(kf->optimal_Tcw_);
    }

    //! update mpts & remove mappoint with large error
    std::set<KeyFrame::Ptr> changed_keyframes;
    static const double max_residual = pixel_usigma * pixel_usigma * std::sqrt(3.81);
    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            double residual = utils::reprojectError(item.second->fn_.head<2>(), item.first->Tcw(), mpt->optimal_pose_);
            if(residual < max_residual)
                continue;

            mpt->removeObservation(item.first);
            changed_keyframes.insert(item.first);
//            std::cout << " rm outlier: " << mpt->id_ << " " << item.first->id_ << " " << obs.size() << std::endl;

            if(mpt->type() == MapPoint::BAD)
            {
                bad_mpts.push_back(mpt);
            }
        }

        mpt->setPose(mpt->optimal_pose_);
    }

    for(const KeyFrame::Ptr &kf : changed_keyframes)
    {
        kf->updateConnections();
    }

    //! Report
    double t1 = (double)cv::getTickCount();
    LOG_IF(INFO, report) << "[Optimizer] Finish local BA for KF: " << keyframe->id_ << "(" << keyframe->frame_id_ << ")"
                         << ", KFs: " << actived_keyframes.size() << "(+" << fixed_keyframe.size() << ")"
                         << ", Mpts: " << local_mappoints.size()
                         << ", remove " << bad_mpts.size() << " bad mpts."
                         << " (" << (t1-t0)/cv::getTickFrequency() << "ms)";

    reportInfo<2>(problem, summary, report, verbose);
}

void Optimizer::motionOnlyBundleAdjustment(const Frame::Ptr &frame, bool use_seeds, bool reject, bool report, bool verbose)
{
    const double focus_length = MIN(frame->cam_->fx(), frame->cam_->fy());
    const double pixel_usigma = Config::imagePixelSigma()/focus_length;

    static const size_t OPTIMAL_MPTS = 150;

    frame->optimal_Tcw_ = frame->Tcw();

    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres_slover::SE3Parameterization();
    problem.AddParameterBlock(frame->optimal_Tcw_.data(), SE3d::num_parameters, local_parameterization);

    static const double scale = pixel_usigma * std::sqrt(3.81);
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    std::vector<Feature::Ptr> fts = frame->getFeatures();
    const size_t N = fts.size();
    std::vector<ceres::ResidualBlockId> res_ids(N);
    for(size_t i = 0; i < N; ++i)
    {
        Feature::Ptr ft = fts[i];
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr)
            continue;

        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
        res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(mpt->optimal_pose_.data());
    }

    if(N < OPTIMAL_MPTS)
    {
        std::vector<Feature::Ptr> ft_seeds = frame->getSeeds();
        const size_t needed = OPTIMAL_MPTS - N;
        if(ft_seeds.size() > needed)
        {
            std::nth_element(ft_seeds.begin(), ft_seeds.begin()+needed, ft_seeds.end(),
                             [](const Feature::Ptr &a, const Feature::Ptr &b)
                             {
                               return a->seed_->getInfoWeight() > b->seed_->getInfoWeight();
                             });

            ft_seeds.resize(needed);
        }

        const size_t M = ft_seeds.size();
        res_ids.resize(N+M);
        for(int i = 0; i < M; ++i)
        {
            Feature::Ptr ft = ft_seeds[i];
            Seed::Ptr seed = ft->seed_;
            if(seed == nullptr)
                continue;

            seed->optimal_pose_.noalias() = seed->kf->Twc() * (seed->fn_ref / seed->getInvDepth());

            ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(seed->fn_ref[0]/seed->fn_ref[2], seed->fn_ref[1]/seed->fn_ref[2], seed->getInfoWeight());
            res_ids[i] = problem.AddResidualBlock(cost_function, lossfunction, frame->optimal_Tcw_.data(), seed->optimal_pose_.data());
            problem.SetParameterBlockConstant(seed->optimal_pose_.data());

        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    if(reject)
    {
        int remove_count = 0;

        static const double TH_REPJ = 3.81 * pixel_usigma * pixel_usigma;
        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];
            if(evaluateResidual<2>(problem, res_ids[i]).squaredNorm() > TH_REPJ * (1 << ft->level_))
            {
                remove_count++;
                problem.RemoveResidualBlock(res_ids[i]);
                frame->removeFeature(ft);
            }
        }

        ceres::Solve(options, &problem, &summary);

        LOG_IF(WARNING, report) << "[Optimizer] Motion-only BA removes " << remove_count << " points";
    }

    //! update pose
    frame->setTcw(frame->optimal_Tcw_);

    //! Report
    reportInfo<2>(problem, summary, report, verbose);
}

void Optimizer::refineMapPoint(const MapPoint::Ptr &mpt, int max_iter, bool report, bool verbose)
{

#if 0
    ceres::Problem problem;
    double scale = Config::pixelUnSigma() * 2;
    ceres::LossFunction* lossfunction = new ceres::HuberLoss(scale);

    //! add obvers kf
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const KeyFrame::Ptr kf_ref = mpt->getReferenceKeyFrame();

    mpt->optimal_pose_ = mpt->pose();

    for(const auto &obs_item : obs)
    {
        const KeyFrame::Ptr &kf = obs_item.first;
        const Feature::Ptr &ft = obs_item.second;
        kf->optimal_Tcw_ = kf->Tcw();

        ceres::CostFunction* cost_function = ceres_slover::ReprojectionErrorSE3::Create(ft->fn_[0]/ft->fn_[2], ft->fn_[1]/ft->fn_[2]);//, 1.0/(1<<ft->level_));
        problem.AddResidualBlock(cost_function, lossfunction, kf->optimal_Tcw_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(kf->optimal_Tcw_.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = report & verbose;
    options.max_linear_solver_iterations = max_iter;

    ceres::Solve(options, &problem, &summary);

    mpt->setPose(mpt->optimal_pose_);

    reportInfo(problem, summary, report, verbose);
#else

    double t0 = (double)cv::getTickCount();
    mpt->optimal_pose_ = mpt->pose();
    Vector3d pose_last = mpt->optimal_pose_;
    const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
    const size_t n_obs = obs.size();

    Matrix3d A;
    Vector3d b;
    double init_chi2 = std::numeric_limits<double>::max();
    double last_chi2 = std::numeric_limits<double>::max();
    const double EPS = 1E-10;

    const bool progress_out = report&verbose;
    bool convergence = false;
    int i = 0;
    for(; i < max_iter; ++i)
    {
        A.setZero();
        b.setZero();
        double new_chi2 = 0.0;

        //! compute res
        for(const auto &obs_item : obs)
        {
            const SE3d Tcw = obs_item.first->Tcw();
            const Vector2d fn = obs_item.second->fn_.head<2>();

            const Vector3d point(Tcw * mpt->optimal_pose_);
            const Vector2d resduial(point.head<2>()/point[2] - fn);

            new_chi2 += resduial.squaredNorm();

            Eigen::Matrix<double, 2, 3> Jacobain;

            const double z_inv = 1.0 / point[2];
            const double z_inv2 = z_inv*z_inv;
            Jacobain << z_inv, 0.0, -point[0]*z_inv2, 0.0, z_inv, -point[1]*z_inv2;

            Jacobain = Jacobain * Tcw.rotationMatrix();

            A.noalias() += Jacobain.transpose() * Jacobain;
            b.noalias() -= Jacobain.transpose() * resduial;
        }

        if(i == 0)  {init_chi2 = new_chi2;}

        if(last_chi2 < new_chi2)
        {
            LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": failure, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs;
            mpt->setPose(pose_last);
            return;
        }

        last_chi2 = new_chi2;

        const Vector3d dp(A.ldlt().solve(b));

        pose_last = mpt->optimal_pose_;
        mpt->optimal_pose_.noalias() += dp;

        LOG_IF(INFO, progress_out) << "iter " << std::setw(2) << i << ": success, chi2: " << std::scientific << std::setprecision(6) << new_chi2/n_obs << ", step: " << dp.transpose();

        if(dp.norm() <= EPS)
        {
            convergence = true;
            break;
        }
    }

    mpt->setPose(mpt->optimal_pose_);
    double t1 = (double)cv::getTickCount();
    LOG_IF(INFO, report) << std::scientific  << "[Optimizer] MapPoint " << mpt->id_
                         << " Error(MSE) changed from " << std::scientific << init_chi2/n_obs << " to " << last_chi2/n_obs
                         << "(" << obs.size() << "), time: " << std::fixed << (t1-t0)*1000/cv::getTickFrequency() << "ms, "
                         << (convergence? "Convergence" : "Unconvergence");

#endif
}


int Optimizer::optimizeSim3(KeyFrame::Ptr pKF1, KeyFrame::Ptr pKF2, std::vector<MapPoint::Ptr> &vpMatches1,
                            Sophus::Sim3d &S12, const float th2, const bool bFixScale)
{

    LOG(WARNING) << "[LoopClosure] Begin to optimize Sim3! ";
    //! sim3.data: x,y,z,w,t(3)

    Sophus::Sim3d sim3(S12);

    ceres::Problem problem;

    problem.AddParameterBlock(sim3.data(), 7);
//    problem.SetParameterBlockConstant(sim3.data());

    const std::vector<MapPoint::Ptr> vpMapPoints1 = pKF1->mapPointsInBow;
    int N = vpMatches1.size();

    std::vector<Vector3d,Eigen::aligned_allocator<Vector3d>> Mp_sets1, Mp_sets2;
    Mp_sets1.resize(N);
    Mp_sets2.resize(N);

    Eigen::Matrix3d tK;
    cv::cv2eigen(pKF1->cam_->K(),tK);

    // Camera poses
    std::vector<ceres::ResidualBlockId> res_ids(2*N);

    int residual_num = 0;

    for (int i = 0; i < N; ++i)
    {
//        if (outliers[i])
//            continue;

        MapPoint::Ptr pMP1 = vpMapPoints1[i];
        MapPoint::Ptr pMP2 = vpMatches1[i];

        if (!pMP1 || !pMP2)
            continue;

        Feature::Ptr ft_1 = pKF1->featuresInBow[i];
        Feature::Ptr ft_2 = pMP2->findObservation(pKF2);

        //todo ft_2->mpt_可能是空的？
        if (pMP1->isBad() || !ft_1 || pMP2->isBad() || !ft_2 || ft_2->mpt_ != pMP2 || ft_1->mpt_ != pMP1)
            continue;

        // x1  x2
        Mp_sets1[i] = pKF1->Tcw() * pMP1->pose();
        Mp_sets2[i] = pKF2->Tcw() * pMP2->pose();


        // X1 = se3 * X2
        ceres::LossFunction *lossfunction1 = new ceres::HuberLoss(0.5);
        ceres::CostFunction *costFunction1 = ceres_slover::ReprojErrorOnlyPose::Create(ft_1->px_[0], ft_1->px_[1],  Mp_sets2[i].data(), ft_1->level_,pKF1);
        res_ids[i*2] = problem.AddResidualBlock(costFunction1, lossfunction1, sim3.data());

        // X2 = 1/se3 * X1
        ceres::LossFunction *lossFunction2 = new ceres::HuberLoss(0.5);
        ceres::CostFunction *costFunction2 = ceres_slover::ReprojErrorOnlyPoseInvSim3::Create(ft_2->px_[0], ft_2->px_[1], Mp_sets1[i].data(),ft_2->level_,pKF2);
        res_ids[i*2+1] = problem.AddResidualBlock(costFunction2, lossFunction2, sim3.data());

        residual_num += 2;
    }

    res_ids.resize(residual_num);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
    //options.minimizer_progress_to_stdout = true;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 50;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout<<summary.FullReport()<<std::endl;

    for(int iter = 0; iter < 1; iter++)
    {
        for (int i = 0; i < N; i++)
        {
            MapPoint::Ptr pMP1 = vpMapPoints1[i];
            MapPoint::Ptr pMP2 = vpMatches1[i];

            if (!pMP1 || !pMP2)
                continue;

            Feature::Ptr ft_1 = pKF1->featuresInBow[i];
            Feature::Ptr ft_2 = pMP2->findObservation(pKF2);

            if (pMP1->isBad() || pMP2->isBad() || !ft_2)
                continue;

            // x1  x2

            double X1[3], X2[3];
            for (int k = 0; k < 3; k++)
            {
                X1[k] = Mp_sets1[i][k];
                X2[k] = Mp_sets2[i][k];
            }

            double residual1[2], residual2[2];

            compute_residual(sim3, X2, ft_1->px_[0], ft_1->px_[1], residual1,pKF1);
            compute_residualInv(sim3, X1, ft_2->px_[0], ft_2->px_[1], residual2,pKF2);

            double chi1 = residual1[0] * residual1[0] + residual1[1] * residual1[1];
            double chi2 = (residual2[0] * residual2[0] + residual2[1] * residual2[1]);

            if ((chi1/(1<<(2*ft_1->level_))) > th2 || (chi2/(1<<(2*ft_2->level_))) > th2 && (i*2+1) < residual_num)
            {
                //todo use residual block to do
                if(!res_ids[i*2])
                    continue;

                problem.RemoveResidualBlock(res_ids[i*2]);
                problem.RemoveResidualBlock(res_ids[i*2+1]);
            }
        }
        ceres::Solve(options, &problem, &summary);
        std::cout<<summary.FullReport()<<std::endl;
    }

    int good = 0;

    for (int i = 0; i < N; i++)
    {
        MapPoint::Ptr pMP1 = vpMapPoints1[i];
        MapPoint::Ptr pMP2 = vpMatches1[i];

        if (!pMP1 || !pMP2)
            continue;

        Feature::Ptr ft_1 = pKF1->featuresInBow[i];
        Feature::Ptr ft_2 = pMP2->findObservation(pKF2);

        if (pMP1->isBad() || !ft_1 || pMP2->isBad() || !ft_2)
        {
            vpMatches1[i] = static_cast<MapPoint::Ptr>(NULL);
            continue;
        }

        double X1[3], X2[3];
        for (int k = 0; k < 3; k++)
        {
            X1[k] = Mp_sets1[i][k];
            X2[k] = Mp_sets2[i][k];
        }
        double residual1[2], residual2[2];
        compute_residual(sim3, X2, ft_1->px_[0], ft_1->px_[1], residual1,pKF1);
        compute_residualInv(sim3, X1, ft_2->px_[0], ft_2->px_[1], residual2,pKF2);
        double chi1 = residual1[0] * residual1[0] + residual1[1] * residual1[1];
        double chi2 = (residual2[0] * residual2[0] + residual2[1] * residual2[1]);

        if ((chi1/(1<<(2*ft_1->level_))) > th2 || (chi2/(1<<(2*ft_2->level_))) > th2 && (i*2+1) < residual_num)
        {
            vpMatches1[i] = static_cast<MapPoint::Ptr>(NULL);
        }
        else
            good++;
    }


//    std::cout<<"==================after OptimizerSim3====================="<<std::endl;
//    std::cout<<" R ======== <<"<<std::endl<< sim3.rotationMatrix() <<std::endl;
//    std::cout<<" t ======== <<"<<std::endl<< sim3.translation().transpose() <<std::endl;
//    std::cout<<" s ======== <<"<<std::endl<< sim3.scale() <<std::endl;

    if (good < 30)
    {
        LOG(WARNING) << "[LoopClosure] Wrong result after optimize sim3!!!";
        return 0;
    }

    S12 = sim3;
    LOG(WARNING) << "[LoopClosure] Good result after optimize sim3!!!";

    bool test_sim3 = true;
    if(test_sim3)
    {
        std::vector<cv::Point2f> points1,points21;
        for (int i = 0; i < N; ++i)
        {
            MapPoint::Ptr pMP1 = vpMapPoints1[i];
            MapPoint::Ptr pMP2 = vpMatches1[i];

            if (!pMP1 || !pMP2)
                continue;

            Feature::Ptr ft_1 = pKF1->featuresInBow[i];
            Feature::Ptr ft_2 = pMP2->findObservation(pKF2);

            if (pMP1->isBad() || !ft_1 || pMP2->isBad() || !ft_2)
                continue;

            points1.push_back(cv::Point2f(ft_1->px_[0],ft_1->px_[1]));

            // x1  x2
            Mp_sets1[i] = pKF1->Tcw() * pMP1->pose();
            Mp_sets2[i] = pKF2->Tcw() * pMP2->pose();


            Vector3d Mp_cam = sim3.inverse() * Mp_sets1[i];


            Vector2d px_21 = pKF2->cam_->project(Mp_cam);

            points21.push_back(cv::Point2f(px_21[0],px_21[1]));

        }

        cv::Mat image_show;
        image_show = showMatch_op(pKF1->getImage(0),pKF2->getImage(0),points1,points21);
        std::string name_after  = "sim3ProjectAfterOpti.png";
        cv::imwrite(name_after,image_show);
    }
    return good;
}

void Optimizer::OptimizeEssentialGraph(Map::Ptr pMap, KeyFrame::Ptr pLoopKF, KeyFrame::Ptr pCurKF, KeyFrameAndPose &NonCorrectedSim3, KeyFrameAndPose &CorrectedSim3,
                                       const std::map<KeyFrame::Ptr, std::set<KeyFrame::Ptr> > &LoopConnections, const bool &bFixScale)
{
    //! Set ceres problem
    ceres::Problem problem;
    const std::vector<KeyFrame::Ptr> vpKFs = pMap->getAllKeyFrames();
    const std::vector<MapPoint::Ptr> vpMPs = pMap->getAllMapPoints();

    for(KeyFrame::Ptr kf:vpKFs)
    {
        kf->beforeUpdate_Tcw_ = kf->Tcw();
    }

    int N = vpKFs.size();

    std::vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>> mvSim3; //参与优化的变量
    std::vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>> vScw; // 优化前变量，用于mappoint校正
    mvSim3.resize(N);
    vScw.resize(N);

    //todo 50->100
    const int minFeat = 50;

    // Set KeyFrame vertices
    for(size_t i=0; i < vpKFs.size(); i++)
    {
        KeyFrame::Ptr pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        int kfID = pKF->id_;

        KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[kfID] = it->second;

            mvSim3[kfID] = it->second;
            problem.AddParameterBlock(mvSim3[kfID].data(), 7);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = pKF->Tcw().rotationMatrix();
            Eigen::Matrix<double,3,1> tcw = pKF->Tcw().translation( );


            Eigen::Quaterniond q_r(Rcw);
//            Matrix4d temp;
//            temp.topLeftCorner(3,3) =  Rcw;
//            temp.topRightCorner(3,1) = tcw;

            vScw[kfID] = Sophus::Sim3d(q_r,tcw);
            mvSim3[kfID] = Sophus::Sim3d(q_r,tcw);
            problem.AddParameterBlock(mvSim3[kfID].data(), 7);
        }

        if(pKF==pLoopKF)
            problem.SetParameterBlockConstant(mvSim3[kfID].data());
    }

    std::set<std::pair<uint64_t ,uint64_t> > sInsertedEdges;

    std::vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>> tDeltaSim3Add;
    std::vector<double*> tDeltaSim3Address;
    tDeltaSim3Add.reserve(3000);
    int edge_num = 0;
    // Set Loop edges
    for(std::map<KeyFrame::Ptr, std::set<KeyFrame::Ptr> >::const_iterator mit = LoopConnections.begin(), mend=LoopConnections.end(); mit!=mend; mit++)
    {
        KeyFrame::Ptr pKF = mit->first;
        uint64_t nIDi = pKF->id_;
        const std::set<KeyFrame::Ptr> &spConnections = mit->second;

        Sophus::Sim3d Swi = mvSim3[nIDi].inverse();

        for(std::set<KeyFrame::Ptr>::const_iterator sit=spConnections.begin(), send=spConnections.end(); sit!=send; sit++)
        {
            const long unsigned int nIDj = (*sit)->id_;
//            std::cout<<" Wight of LoopConnections < "<<pKF->frame_id_<<","<<(*sit)->frame_id_<<"> is "<<pKF->getWight(*sit)<<std::endl;
            if((nIDi!=pCurKF->id_ || nIDj!=pLoopKF->id_) && pKF->getWight(*sit)< 0.2*minFeat)
                continue;
            //! Sji = Sjw*Swi
            double *Sim3_Address = new double[7];
            Sophus::Sim3d Sji = mvSim3[nIDj] * Swi;
            Sim3_Address = Sji.data();
            tDeltaSim3Address.push_back(Sim3_Address);

            ceres::LossFunction *lossfunc = new ceres::HuberLoss(0.5);
            ceres::CostFunction *costfunc = ceres_slover::RelativeSim3Error::Create(tDeltaSim3Address.back());
            problem.AddResidualBlock(costfunc, lossfunc, mvSim3[nIDi].data(), mvSim3[nIDj].data());

            sInsertedEdges.insert(std::make_pair(std::min(nIDi,nIDj),std::max(nIDi,nIDj)));
            edge_num++;
        }
    }

//    std::cout<<"LoopConnections ResidualBlock edge_num: "<<edge_num<<std::endl;

    edge_num = 0;
    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame::Ptr pKF = vpKFs[i];
        const int nIDi = pKF->id_;
        Sophus::Sim3d Swi;

        KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);
        if(iti != NonCorrectedSim3.end())
            Swi = iti->second.inverse();
        else
            Swi = mvSim3[nIDi].inverse();

        //todo add orb-slam spanning tree edges

        // Loop edges
        const std::set<KeyFrame::Ptr> sLoopEdges = pKF->getLoopEdges();
        for(std::set<KeyFrame::Ptr>::const_iterator sit=sLoopEdges.begin(), send=sLoopEdges.end(); sit!=send; sit++)
        {
            KeyFrame::Ptr pLKF = *sit;
            if(pLKF->id_<pKF->id_)
            {
                Sophus::Sim3d Slw;

                KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                if(itl!=NonCorrectedSim3.end())
                    Slw = itl->second;
                else
                    Slw = mvSim3[pLKF->id_];


                double *Sim3_Address = new double[7];
                Sophus::Sim3d Sli = Slw * Swi;
                Sim3_Address = Sli.data();
                tDeltaSim3Address.push_back(Sim3_Address);

                ceres::LossFunction *lossfunc = new ceres::HuberLoss(0.5);
                ceres::CostFunction *costfunc = ceres_slover::RelativeSim3Error::Create(tDeltaSim3Address.back());
                problem.AddResidualBlock(costfunc, lossfunc, vScw[nIDi].data(), mvSim3[pLKF->id_].data());

                edge_num++;
            }
        }

//        std::cout<<"Loop edge--->: "<<edge_num<<std::endl;

//        edge_num = 0;
        // Covisibility graph edges
        const std::set<KeyFrame::Ptr> vpConnectedKFs = pKF->getConnectedKeyFrames(-1,minFeat);
        for(std::set<KeyFrame::Ptr>::const_iterator vit=vpConnectedKFs.begin(); vit!=vpConnectedKFs.end(); vit++)
        {
            KeyFrame::Ptr pKFn = *vit;
            if(pKFn /*&& pKFn!=pParentKF */&& !sLoopEdges.count(pKFn))
            {
                //todo child check of orb slam
                if(!pKFn->isBad() && pKFn->getParent()!=pKF && pKFn->id_<pKF->id_)
                {
                    if(sInsertedEdges.count(std::make_pair(std::min(pKF->id_,pKFn->id_),std::max(pKF->id_,pKFn->id_))))
                        continue;

                    Sophus::Sim3d Snw;

                    KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                    if(itn!=NonCorrectedSim3.end())
                        Snw = itn->second;
                    else
                        Snw = mvSim3[pKFn->id_];

                    double *Sim3_Address = new double[7];

                    Sophus::Sim3d Sni = Snw * Swi;
                    Sim3_Address = Sni.data();
                    tDeltaSim3Address.push_back(Sim3_Address);

                    ceres::LossFunction *lossfunc = new ceres::HuberLoss(0.5);
                    ceres::CostFunction *costfunc = ceres_slover::RelativeSim3Error::Create(tDeltaSim3Address.back());
                    problem.AddResidualBlock(costfunc, lossfunc, mvSim3[nIDi].data(), mvSim3[pKFn->id_].data());
                    edge_num++;
                }
            }
        }
    }

//    std::cout<<"normal edges --->: "<<edge_num<<std::endl;

    //solve problem
    ceres::Solver::Options options;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 20;
    //options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<"OptimizeEssentialGraph FullReport()"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

    //todo delete
    /*
    std::cout<<"Begin to delete all double*!"<<std::endl;
    std::cout<<"tDeltaSim3Address size:"<<tDeltaSim3Address.size()<<std::endl;
    for(std::vector<double*>::iterator lit = tDeltaSim3Address.begin(), lend = tDeltaSim3Address.end(); lit!=lend; lit++)
    {
        std::cout<<"Delete"<<std::endl;
        delete *lit;
    }
    tDeltaSim3Address.clear();
    std::cout<<"Finish to delete all double*!"<<std::endl;
     */

    std::vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>> vCorrectedSwc(N);

    LOG(WARNING) << "[LoopClosure] Begin to correct kf pose!";
    std::unique_lock<std::mutex > lock(pMap->mutex_update_);
    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for(size_t i=0;i<vpKFs.size();i++)
    {
        KeyFrame::Ptr pKFi = vpKFs[i];

        const int nIDi = pKFi->id_;

        vCorrectedSwc[nIDi]=mvSim3[nIDi].inverse();
        Eigen::Matrix3d eigR = mvSim3[nIDi].rotationMatrix();
        Eigen::Vector3d eigt = mvSim3[nIDi].translation();
        double s = mvSim3[nIDi].scale();

        eigt *=(1.0/s); //[R t/s;0 1]

        SE3d Tiw = SE3d(eigR,eigt);

        pKFi->setTcw(Tiw);
    }
    LOG(WARNING) << "[LoopClosure] Finish to correct kf pose!";

    LOG(WARNING) << "[LoopClosure] Begin to correct point pose!";
    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint::Ptr pMP = vpMPs[i];

        if(pMP->isBad())
            continue;

        int nIDr;
        // 经过sim3矫正的点
        if(pMP->mnCorrectedByKF==pCurKF->id_)
        {
            nIDr = pMP->mnCorrectedReference;
        }
        else
        {
            KeyFrame::Ptr pRefKF = pMP->getReferenceKeyFrame();
            nIDr = pRefKF->id_;
        }

        Sophus::Sim3d Srw = vScw[nIDr];
        SE3d Srw_se3 = SE3d(Srw.rotationMatrix(),Srw.translation());
        Sophus::Sim3d correctedSwr = vCorrectedSwc[nIDr];
        SE3d  correctedSwr_se3 = SE3d(correctedSwr.rotationMatrix(),correctedSwr.translation());

        Vector3d eigP3Dw = pMP->pose();
//        Eigen::Vector3d eigCorrectedP3Dw = correctedSwr_se3 * (Srw_se3 * eigP3Dw);
        Eigen::Vector3d eigCorrectedP3Dw = correctedSwr * (Srw * eigP3Dw);

        pMP->setPose(eigCorrectedP3Dw);
        pMP->updateViewAndDepth();

    }
    LOG(WARNING) << "[LoopClosure] Correct all kf and mappoint pose!";


}

}