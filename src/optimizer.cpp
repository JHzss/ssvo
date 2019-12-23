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

    std::vector<KeyFrame::Ptr> all_kfs;
    if(nLoopKF ==0)
    {
        all_kfs = map->getAllKeyFrames();
    }
    else
    {
        all_kfs = map->getAllKeyFrames(nLoopKF);
    }

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
        kf->beforeUpdate_Tcw_ = kf->Tcw();
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


//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------
/**
 * 计算陀螺仪bias
 * @param vTwc [in]
 * @param vImuPreInt [in]
 * @return 计算得到的bg
 */
Vector3d Optimizer::OptimizeInitialGyroBias(const std::vector<SE3d,Eigen::aligned_allocator<SE3d>>& vTwc, const std::vector<IMUPreintegrator>& vImuPreInt)
{
    int N = vTwc.size();
    LOG_ASSERT(vTwc.size()==vImuPreInt.size())<<"vTwc.size()!=vImuPreInt.size()"<<std::endl;

    Matrix4d Tbc = ImuConfigParam::GetEigTbc();
    Matrix3d Rcb = Tbc.topLeftCorner(3,3).transpose();

    ceres::Problem problem;

    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(7.815);//降低外点的影响

    //!仅有这一个优化变量
    Vector3d vBiasg;
    vBiasg.setZero();
    problem.AddParameterBlock(vBiasg.data(),3);

    for (int i = 0; i < N; ++i)
    {
        // 第一帧的预积分没有意义，忽略
        if(i==0)
            continue;
        //上一个关键帧的位姿
        const SE3d Twi = vTwc[i-1];
        Matrix3d Rwci = Twi.rotationMatrix();
        //下一帧关键帧的位姿
        const SE3d Twj = vTwc[i];
        Matrix3d Rwcj = Twj.rotationMatrix();

        const IMUPreintegrator& imupreint = vImuPreInt[i];

        Matrix3d dRbij = imupreint.getDeltaR();
        Matrix3d J_dR_bg = imupreint.getJRBiasg();
        Matrix3d Rwbi = Rwci*Rcb;
        Matrix3d Rwbj = Rwcj*Rcb;

        ceres::CostFunction* costFunction = ceres_slover::GyrBiasError::Creat(dRbij,J_dR_bg,Rwbi,Rwbj);
        problem.AddResidualBlock(costFunction,loss_function,vBiasg.data());
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    std::cout<<"OptimizeInitialGyroBias FullReport()"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

    return vBiasg;
}

/**
 * 计算尺度、重力分量
 * @param
 * @param
 * @return
 */
void Optimizer::OptimizeInitialScaleGravity(const std::vector<SE3d, Eigen::aligned_allocator<SE3d>> &vE_Twc, const std::vector<std::shared_ptr<KeyFrameInit>> &vKFInit)
{
//    ceres::Problem problem;
//    ceres::LossFunction *loss_function;
//    loss_function = new ceres::CauchyLoss(1.0);//降低外点的影响
//    double X[4];
//
//    int N = vKFInit.size();
//
//    Eigen::MatrixXd A{3*(N-2),4};
//    Eigen::VectorXd B{3*(N-2)};
//    Eigen::Matrix3d I3;
//
//    A.setZero();
//    B.setZero();
//    I3.setIdentity();
//
//    for(int i = 0; i<N-2; i++)
//    {
//        //KeyFrameInit* pKF1 = vKFInit[i];//vScaleGravityKF[i];
//        KeyFrameInit::Ptr pKF2 = vKFInit[i+1];
//        KeyFrameInit::Ptr pKF3 = vKFInit[i+2];
//        // Delta time between frames
//        double dt12 = pKF2->mIMUPreInt.getDeltaTime();
//        double dt23 = pKF3->mIMUPreInt.getDeltaTime();
//        // Pre-integrated measurements
//        Vector3d dp12 = pKF2->mIMUPreInt.getDeltaP();
//        Vector3d dv12 = pKF2->mIMUPreInt.getDeltaV();
//        Vector3d dp23 = pKF3->mIMUPreInt.getDeltaP();
//
//        SE3d Twc1 = vE_Twc[i];
//        SE3d Twc2 = vE_Twc[i+1];
//        SE3d Twc3 = vE_Twc[i+2];
//
//        Vector3d pc1 = Twc1.translation();
//        Vector3d pc2 = Twc2.translation();
//        Vector3d pc3 = Twc3.translation();
//
//        Matrix3d Rc1 = Twc1.rotationMatrix();
//        Matrix3d Rc2 = Twc2.rotationMatrix();
//        Matrix3d Rc3 = Twc3.rotationMatrix();
//
//        // Stack to A/B matrix
//        // lambda*s + beta*g = gamma
//        Vector3d lambda = (pc2-pc1)*dt23 + (pc2-pc3)*dt12;
//        Matrix3d beta = 0.5*I3*(dt12*dt12*dt23 + dt12*dt23*dt23);
//        Vector3d gamma = (Rc3-Rc2)*E_pcb*dt12 + (Rc1-Rc2)*E_pcb*dt23 + Rc1*E_Rcb*dp12*dt23 - Rc2*E_Rcb*dp23*dt12 - Rc1*E_Rcb*dv12*dt12*dt23;
//
//        A.block(3*i,0,3,1) = lambda;
//        A.block(3*i,1,3,3) = beta;
//        B.block(3*i,0,3,1) = gamma;
//        // Tested the formulation in paper, -gamma. Then the scale and gravity vector is -xx
//
//         Matrix<double,3,4> A_;
//         Vector3d B_;
//         A_.block(0,0,3,1) = lambda;
//         A_.block(0,1,3,3) = beta;
//         B_ = gamma;
//
//         ceres::CostFunction* costFunction = ceres_slover::AlignmentError::Creat(A_,B_);
//         problem.AddResidualBlock(costFunction,loss_function,X);
//
//    }

}

void Optimizer::GlobalBundleAdjustmentNavStatePRV(Map::Ptr pMap, const Vector3d &gw, int nIterations,const uint64_t nLoopKF, bool report, bool verbose)
{

    if (pMap->KeyFramesInMap() < 2)
        return;

    // Extrinsics
    Matrix4d Tbc = ImuConfigParam::GetEigTbc();
    Sophus::SE3d Tbc_ = Sophus::SE3d(Tbc);
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);

    // Gravity vector in world frame
    Vector3d GravityVec = gw;

    std::vector<KeyFrame::Ptr> vpKFs = pMap->getAllKeyFrames(/*nLoopKF*/);
    std::vector<MapPoint::Ptr> vpMP = pMap->getAllMapPoints();
    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    std::sort(vpKFs.begin(),vpKFs.end(),[](KeyFrame::Ptr kf1,KeyFrame::Ptr kf2)->bool{ return kf1->id_ > kf2->id_;});


    static double focus_length = MIN(vpKFs.back()->cam_->fx(), vpKFs.back()->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma() / focus_length;

    ceres::Problem problem;

    // 添加关键帧的变量
    for (const KeyFrame::Ptr &kf : vpKFs)
    {
        //todo  NavState 中 V的状态在初始化的时候已经处理好了，所以这里就不对V进行更新，但是之后如果再用到位姿优化的话需要考虑一下。
        kf->UpdateNavStatePRFromTcw(kf->Tcw(),Tbc_);
        kf->setOptimizationState();
        ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();
        problem.AddParameterBlock(kf->optimal_PR_.data(), 6, local_parameterization_pr);
        problem.AddParameterBlock(kf->optimal_v_.data(),3);
        problem.AddParameterBlock(kf->optimal_detla_bias_.data(),6);
        problem.SetParameterBlockConstant(kf->optimal_detla_bias_.data());

        if(kf->id_ == 0)
        {
            problem.SetParameterBlockConstant(kf->optimal_PR_.data());
//            problem.SetParameterBlockConstant(kf->optimal_detla_bias_.data());
        }
        if(kf->id_ == 1)
        {
            problem.SetParameterBlockConstant(kf->optimal_PR_.data());
        }

    }

    const double thHuberNavStatePRV = sqrt(100*21.666);
    const double thHuberNavStateBias = sqrt(100*16.812);

    double scale = sqrt(3.81) /*pixel_usigma * sqrt(4)*/;
    ceres::LossFunction* lossfunctionPRV = new ceres::HuberLoss(thHuberNavStatePRV);
    ceres::LossFunction* lossfunctionBias = new ceres::HuberLoss(thHuberNavStateBias);
    ceres::LossFunction* lossfunctionMpt = new ceres::HuberLoss(scale);

    //todo 设置协方差矩阵
    //! 视觉误差-重投影误差（公式5）

    int vision_item = 0;
    for (const MapPoint::Ptr &mpt : vpMP)
    {
        if(mpt->isBad())
            continue;

        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for (const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;

            ceres::CostFunction* cost_function1 = ceres_slover::PointReprojectionError::Creat( Tbc_ ,ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);//, 1.0/(1<<ft->level_));
            problem.AddResidualBlock(cost_function1, lossfunctionMpt, kf->optimal_PR_.data(), mpt->optimal_pose_.data());
            vision_item++;
        }
    }

    std::cout<<"Finish add visiual error,sum : "<<vision_item<<std::endl;


    int imu_item = 0;
    //! IMU误差 bias误差
    Eigen::Matrix<double,6,6> InvCovBgaRW = Eigen::Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();   // Accelerometer bias random walk, covariance INVERSE

    std::vector<ceres::ResidualBlockId> res_id;
    for (int i = 0; i < vpKFs.size()-1; ++i)
    {
        KeyFrame::Ptr pKF1 = vpKFs[i];
        if(pKF1->isBad())
        {
            std::cout<<"pKF is bad in gBA, id "<<pKF1->id_<<std::endl;   //Debug log
            continue;
        }

        KeyFrame::Ptr pKF0 = pKF1->GetPrevKeyFrame();

//        std::cout<<"pkf0----pkf1: "<<pKF0->id_<<"---"<<pKF1->id_<<std::endl;
        if(!pKF0)
        {
            LOG_ASSERT(pKF1->id_ == 0)<<"Previous KeyFrame is NULL?"<<std::endl;  //Test log
        }
        std::cout<<"pkf0----pkf1: "<<pKF0->id_<<"---"<<pKF1->id_<<std::endl;

        //todo

        Matrix9d CovPRV = pKF1->GetIMUPreInt().getCovPVPhi();
        CovPRV.col(3).swap(CovPRV.col(6));
        CovPRV.col(4).swap(CovPRV.col(7));
        CovPRV.col(5).swap(CovPRV.col(8));
        CovPRV.row(3).swap(CovPRV.row(6));
        CovPRV.row(4).swap(CovPRV.row(7));
        CovPRV.row(5).swap(CovPRV.row(8));

        ceres::CostFunction* costFunction = ceres_slover::NavStatePRVError::Create(CovPRV,pKF1->GetIMUPreInt(),GravityVec);
        ceres::ResidualBlockId residualBlockId = problem.AddResidualBlock(costFunction,NULL,pKF0->optimal_PR_.data(),pKF0->optimal_v_.data(),pKF0->optimal_detla_bias_.data(),
                                 pKF1->optimal_PR_.data(),pKF1->optimal_v_.data());

        res_id.push_back(residualBlockId);


//        std::cout<<"add imu error success"<<std::endl;

        Vector3d bg_0 = pKF0->GetNavState().Get_BiasGyr();
        Vector3d ba_0 = pKF0->GetNavState().Get_BiasAcc();
        Vector3d bg_1 = pKF1->GetNavState().Get_BiasGyr();
        Vector3d ba_1 = pKF1->GetNavState().Get_BiasAcc();

        ceres::CostFunction* costFunction1 = ceres_slover::NavStateBiasError::Creat((InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime()).inverse(),
                                                                                    bg_0,ba_0,bg_1,ba_1);
        problem.AddResidualBlock(costFunction1,lossfunctionBias,pKF0->optimal_detla_bias_.data(),pKF1->optimal_detla_bias_.data());
//        std::cout<<"add bias error success"<<std::endl<<std::endl;
        imu_item++;
    }
    std::cout<<"Finish add imu error. sum: "<<imu_item<<std::endl;

//    std::abort();

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;
//    options.max_num_iterations = nIterations;
//    options_.gradient_tolerance = 1e-4;
//    options_.function_tolerance = 1e-4;
    //options_.max_solver_time_in_seconds = 0.2;

    std::cout<<"========= Begin solve GlobalBundleAdjustmentNavStatePRV ==========="<<std::endl;
    ceres::Solve(options, &problem, &summary);

    for(auto &item1:res_id)
    {
        double residual = evaluateResidual<9>(problem,item1).squaredNorm();
        std::cout<<"residual id: "<<residual<<std::endl;
    }
    std::cout<<"GlobalBundleAdjustmentNavStatePRV FullReport()"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

    //! Report
//    reportInfo<2>(problem, summary, report, verbose);

    //todo 根据误差去除误匹配点
    for(auto kf:vpKFs)
    {
        kf->GBA_KF_ = nLoopKF;kf->beforeGBA_Tcw_ = kf->Tcw();
    }
    for(auto mpt:vpMP)
    {
        mpt->GBA_KF_ = nLoopKF;
    }

//    cv::waitKey(0);
//    std::abort();
    //! 根据优化参数设置状态


    for(KeyFrame::Ptr &kf:vpKFs)
    {
        NavState ns_recov;

        Vector3d optimal_Pwb_(kf->optimal_PR_[0],kf->optimal_PR_[1],kf->optimal_PR_[2]);
        Vector3d optimal_PHIwb_(kf->optimal_PR_[3],kf->optimal_PR_[4],kf->optimal_PR_[5]);
        Sophus::SO3d optimal_Rwb_ = Sophus::SO3d::exp(optimal_PHIwb_);

        ns_recov.Set_Pos(optimal_Pwb_);
        ns_recov.Set_Rot(optimal_Rwb_);
        ns_recov.Set_Vel(kf->optimal_v_);
        ns_recov.Set_BiasAcc(kf->GetNavState().Get_BiasAcc()+kf->optimal_detla_bias_.block(3,0,3,1));
        ns_recov.Set_BiasGyr(kf->GetNavState().Get_BiasGyr()+kf->optimal_detla_bias_.block(0,0,3,1));
        ns_recov.Set_DeltaBiasGyr(Vector3d::Zero());
        ns_recov.Set_DeltaBiasAcc(Vector3d::Zero());

        kf->SetNavState(ns_recov);

        SE3d Twb_ = SE3d(optimal_Rwb_,optimal_Pwb_);
        SE3d Tbc_ = SE3d(Tbc);
        kf->optimal_Tcw_ = (Twb_ * Tbc_).inverse();
        kf->setTcw(kf->optimal_Tcw_);
    }


}
/**
 * @brief ORB localBA的变形，只优化窗口内的关键帧，与这些关键帧有公式关系的那些关键帧fix位姿
 */
void Optimizer::LocalBAPRVIDP(Map::Ptr pMap, const KeyFrame::Ptr pCurKF,const std::vector<KeyFrame::Ptr> &actived_keyframes, std::list<MapPoint::Ptr> &bad_mpts, int size, int min_shared_fts, const Vector3d& gw, bool report, bool verbose)
{
    static double focus_length = MIN(pCurKF->cam_->fx(), pCurKF->cam_->fy());
    static double pixel_usigma = Config::imagePixelSigma()/*/focus_length*/;
    LOG_ASSERT(pCurKF == actived_keyframes.back())<<"pCurKF != actived_keyframes.back. check"<<std::endl;

    Matrix4d Tbc = ImuConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    SE3d Tbc_ = SE3d(Tbc);
    // Gravity vector in world frame
    Vector3d GravityVec = gw;

    // All KeyFrames in Local window are optimized
    for(const KeyFrame::Ptr &pKFi:actived_keyframes)
        pKFi->mnBALocalForKF = pCurKF->id_;

    // Local MapPoints seen in Local KeyFrames
    std::unordered_set<MapPoint::Ptr> local_mappoints;
    std::set<KeyFrame::Ptr> fixed_keyframe;

    // Add the KeyFrame before local window.
    KeyFrame::Ptr pKFPrevLocal = actived_keyframes.front()->GetPrevKeyFrame();
    LOG_ASSERT(!pKFPrevLocal->isBad())<<"pKFPrevLocal is bad!"<<std::endl;
    pKFPrevLocal->mnBAFixedForKF = pCurKF->id_;
    fixed_keyframe.insert(pKFPrevLocal);

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        for(const MapPoint::Ptr &mpt : mpts)
        {
            if(mpt->isBad())
                continue;
            local_mappoints.insert(mpt);
//            mpt->mnBALocalForKF=pCurKF->id_;
        }
    }

    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();
        for(const auto &item : obs)
        {
            if(item.first->mnBALocalForKF == pCurKF->id_ || item.first->mnBAFixedForKF == pCurKF->id_)
                continue;
            fixed_keyframe.insert(item.first);

//            std::cout<<"fixed kf id: "<<item.first->id_<<std::endl;
            item.first->mnBAFixedForKF = pCurKF->id_;
        }
    }
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();

    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        std::cout<<"actived kf->id : "<<kf->id_<<std::endl;
//        kf->UpdateNavStatePVRFromTcw(Tbc_);
        kf->setOptimizationState();
        ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();
//        ceres::LocalParameterization* local_parameterization_bias = new ceres_slover::BiasParameterization();
        problem.AddParameterBlock(kf->optimal_PR_.data(), 6, local_parameterization_pr);
        problem.AddParameterBlock(kf->optimal_v_.data(),3);
        problem.AddParameterBlock(kf->optimal_detla_bias_.data(),6);
        problem.SetParameterBlockConstant(kf->optimal_detla_bias_.data());


        if(kf->id_ <= 1)
        {
            problem.SetParameterBlockConstant(kf->optimal_PR_.data());
        }
    }

    for(const KeyFrame::Ptr &kf : fixed_keyframe)
    {
        std::cout<<"fixed: kf->id : "<<kf->id_<<std::endl;
//        kf->UpdateNavStatePVRFromTcw(Tbc_);
        kf->setOptimizationState();
        ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();
        problem.AddParameterBlock(kf->optimal_PR_.data(), 6, local_parameterization_pr);
        problem.AddParameterBlock(kf->optimal_v_.data(),3);
        problem.AddParameterBlock(kf->optimal_detla_bias_.data(),6);
        problem.SetParameterBlockConstant(kf->optimal_PR_.data());

        if(kf==pKFPrevLocal)
        {
            problem.AddParameterBlock(kf->optimal_v_.data(),3);
            problem.AddParameterBlock(kf->optimal_detla_bias_.data(),6);
            problem.SetParameterBlockConstant(kf->optimal_v_.data());
            problem.SetParameterBlockConstant(kf->optimal_detla_bias_.data());
        }
    }

    double scale = pixel_usigma * 2;
    ceres::LossFunction* lossfunctionMpt = new ceres::HuberLoss(scale);
    const double thHuberNavStatePRV = sqrt(100*21.666);
    const double thHuberNavStateBias = sqrt(100*16.812);
    ceres::LossFunction* lossfunctionPRV = new ceres::HuberLoss(thHuberNavStatePRV);
    ceres::LossFunction* lossfunctionBias = new ceres::HuberLoss(thHuberNavStateBias);

    std::vector<ceres::ResidualBlockId> res_ids_vision;
    std::vector<ceres::ResidualBlockId> res_ids_prv;
    std::vector<ceres::ResidualBlockId> res_ids_bias;

    for(const MapPoint::Ptr &mpt : local_mappoints)
    {
        mpt->optimal_pose_ = mpt->pose();
        const std::map<KeyFrame::Ptr, Feature::Ptr> obs = mpt->getObservations();

        for(const auto &item : obs)
        {
            const KeyFrame::Ptr &kf = item.first;
            const Feature::Ptr &ft = item.second;
            ceres::CostFunction* cost_function1 = ceres_slover::PointReprojectionError::Creat( Tbc_ ,ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);
            ceres::ResidualBlockId residualBlockId_vision = problem.AddResidualBlock(cost_function1, lossfunctionMpt, kf->optimal_PR_.data(), mpt->optimal_pose_.data());
            res_ids_vision.push_back(residualBlockId_vision);
        }
    }

    Eigen::Matrix<double,6,6> InvCovBgaRW = Eigen::Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();

    for (std::vector<KeyFrame::Ptr>::const_iterator lit=actived_keyframes.begin(), lend=actived_keyframes.end(); lit!=lend; lit++)
    {
        KeyFrame::Ptr pKF1 = *lit;
        if(pKF1->isBad())
        {
            std::cout<<"pKF is bad in gBA, id "<<pKF1->id_<<std::endl;   //Debug log
            continue;
        }

        KeyFrame::Ptr pKF0 = pKF1->GetPrevKeyFrame();

        LOG_ASSERT(pKF0 != NULL &&pKF1 != NULL)<<"pKF0 != NULL &&pKF1 != NULL"<<std::endl;

        if(!pKF0)
        {
            LOG_ASSERT(pKF1->id_ == 0)<<"Previous KeyFrame is NULL?"<<std::endl;  //Test log
        }
        std::cout<<"pkf0----pkf1: "<<pKF0->id_<<"---"<<pKF1->id_<<std::endl;
        Matrix9d CovPRV = pKF1->GetIMUPreInt().getCovPVPhi();
        CovPRV.col(3).swap(CovPRV.col(6));
        CovPRV.col(4).swap(CovPRV.col(7));
        CovPRV.col(5).swap(CovPRV.col(8));
        CovPRV.row(3).swap(CovPRV.row(6));
        CovPRV.row(4).swap(CovPRV.row(7));
        CovPRV.row(5).swap(CovPRV.row(8));

        ceres::CostFunction* costFunction = ceres_slover::NavStatePRVError::Create(CovPRV,pKF1->GetIMUPreInt(),GravityVec);
        ceres::ResidualBlockId residualBlockId_prv = problem.AddResidualBlock(costFunction,NULL,pKF0->optimal_PR_.data(),pKF0->optimal_v_.data(),pKF0->optimal_detla_bias_.data(),
                                 pKF1->optimal_PR_.data(),pKF1->optimal_v_.data());
        res_ids_prv.push_back(residualBlockId_prv);

        Vector3d bg_0 = pKF0->GetNavState().Get_BiasGyr();
        Vector3d ba_0 = pKF0->GetNavState().Get_BiasAcc();
        Vector3d bg_1 = pKF1->GetNavState().Get_BiasGyr();
        Vector3d ba_1 = pKF1->GetNavState().Get_BiasAcc();

        ceres::CostFunction* costFunction1 = ceres_slover::NavStateBiasError::Creat((InvCovBgaRW/pKF1->GetIMUPreInt().getDeltaTime()).inverse(),
                                                                                    bg_0,ba_0,bg_1,ba_1);
        ceres::ResidualBlockId residualBlockId_bias = problem.AddResidualBlock(costFunction1,lossfunctionBias,pKF0->optimal_detla_bias_.data(),pKF1->optimal_detla_bias_.data());
        res_ids_bias.push_back(residualBlockId_bias);

    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = report & verbose;
//    options.max_num_iterations = 10;

    ceres::Solve(options, &problem, &summary);

    std::cout<<"LocalBAPRVIDP FullReport()->"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;
//    std::abort();

        std::cout<<"res_ids_vision number:"<<res_ids_vision.size()<<std::endl;
        std::cout<<"res_ids_prv number:"<<res_ids_prv.size()<<std::endl;
        std::cout<<"res_ids_bias number:"<<res_ids_bias.size()<<std::endl;


    for(const KeyFrame::Ptr &kf:actived_keyframes)
    {
        NavState ns_recov;

        Vector3d optimal_Pwb_(kf->optimal_PR_[0],kf->optimal_PR_[1],kf->optimal_PR_[2]);
        Vector3d optimal_PHIwb_(kf->optimal_PR_[3],kf->optimal_PR_[4],kf->optimal_PR_[5]);
        Sophus::SO3d optimal_Rwb_ = Sophus::SO3d::exp(optimal_PHIwb_);

        ns_recov.Set_Pos(optimal_Pwb_);
        ns_recov.Set_Rot(optimal_Rwb_);
        ns_recov.Set_Vel(kf->optimal_v_);
        ns_recov.Set_BiasAcc(kf->GetNavState().Get_BiasAcc()+kf->optimal_detla_bias_.block(3,0,3,1));
        ns_recov.Set_BiasGyr(kf->GetNavState().Get_BiasGyr()+kf->optimal_detla_bias_.block(0,0,3,1));
        ns_recov.Set_DeltaBiasGyr(Vector3d::Zero());
        ns_recov.Set_DeltaBiasAcc(Vector3d::Zero());
        kf->SetNavState(ns_recov);

        SE3d Twb_ = SE3d(optimal_Rwb_,optimal_Pwb_);
        SE3d Tbc_ = SE3d(Tbc);
        kf->optimal_Tcw_ = (Twb_ * Tbc_).inverse();
    }

    //! update pose
    for(const KeyFrame::Ptr &kf : actived_keyframes)
    {
        kf->beforeUpdate_Tcw_ = kf->Tcw();
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

//    reportInfo<2>(problem, summary, report, verbose);
}

// 跟踪上一帧普通帧有先验误差
int Optimizer::PoseOptimization(Frame::Ptr pFrame, Frame::Ptr pLastFrame, const IMUPreintegrator &imupreint, const Vector3d &gw, const bool &bComputeMarg)
{
    const double focus_length = MIN(pFrame->cam_->fx(), pFrame->cam_->fy());
    const double pixel_usigma = Config::imagePixelSigma();

    Matrix4d Tbc = ImuConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    SE3d Tbc_ = SE3d(Tbc);
    Vector3d GravityVec = gw;

    ceres::Problem problem;
//    pLastFrame->UpdateNavStatePVRFromTcw(Tbc_);
    pLastFrame->setOptimizationState();
    //因为经过了图像匹配的过程，所以应该是需要调整的,否则图像匹配的作用就没有了
//    pFrame->UpdateNavStatePVRFromTcw(Tbc_);
    pFrame->setOptimizationState();
    ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();
    problem.AddParameterBlock(pFrame->optimal_PR_.data(), 6, local_parameterization_pr);
    problem.AddParameterBlock(pFrame->optimal_v_.data(),3);
    problem.AddParameterBlock(pFrame->optimal_detla_bias_.data(),6);
    problem.AddParameterBlock(pLastFrame->optimal_PR_.data(), 6, local_parameterization_pr);
    problem.AddParameterBlock(pLastFrame->optimal_v_.data(),3);
    problem.AddParameterBlock(pLastFrame->optimal_detla_bias_.data(),6);
    //todo viorb中这个是固定的
    problem.SetParameterBlockConstant(pLastFrame->optimal_PR_.data());
    problem.SetParameterBlockConstant(pLastFrame->optimal_v_.data());
    problem.SetParameterBlockConstant(pLastFrame->optimal_detla_bias_.data());

    problem.SetParameterBlockConstant(pFrame->optimal_detla_bias_.data());


        static const double scale = pixel_usigma * std::sqrt(3.81);
    const double thHuberNavStatePRV = sqrt(100*21.666);
    const double thHuberNavStateBias = sqrt(100*16.812);
    const double thHuberNavStatePrior = sqrt(30.5779);
    ceres::LossFunction* lossfunctionMpt = new ceres::HuberLoss(scale);
    ceres::LossFunction* lossfunctionPRV = new ceres::HuberLoss(thHuberNavStatePRV);
    ceres::LossFunction* lossfunctionBias = new ceres::HuberLoss(thHuberNavStateBias);
    ceres::LossFunction* lossfunctionPrior = new ceres::HuberLoss(thHuberNavStatePrior);


    //先验误差
    {

    }

    std::vector<Feature::Ptr> fts = pFrame->getFeatures();
    const size_t N = fts.size();
    std::vector<ceres::ResidualBlockId> res_ids(N);
    for(size_t i = 0; i < N; ++i)
    {
        Feature::Ptr ft = fts[i];
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr || mpt->isBad())
            continue;
        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::PointReprojectionError::Creat( Tbc_ ,ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);
        res_ids[i] = problem.AddResidualBlock(cost_function, lossfunctionMpt, pFrame->optimal_PR_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(mpt->optimal_pose_.data());
    }
    //是否使用种子点
    /*
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
    */

    Eigen::Matrix<double,6,6> InvCovBgaRW = Eigen::Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();

    Matrix9d CovPRV = imupreint.getCovPVPhi();
    CovPRV.col(3).swap(CovPRV.col(6));
    CovPRV.col(4).swap(CovPRV.col(7));
    CovPRV.col(5).swap(CovPRV.col(8));
    CovPRV.row(3).swap(CovPRV.row(6));
    CovPRV.row(4).swap(CovPRV.row(7));
    CovPRV.row(5).swap(CovPRV.row(8));

    ceres::CostFunction* costFunction = ceres_slover::NavStatePRVError::Create(CovPRV,imupreint,GravityVec);
    ceres::ResidualBlockId residualBlockId_prv = problem.AddResidualBlock(costFunction,NULL,pLastFrame->optimal_PR_.data(),pLastFrame->optimal_v_.data(),pLastFrame->optimal_detla_bias_.data(),
                                                                          pFrame->optimal_PR_.data(),pFrame->optimal_v_.data());
    Vector3d bg_0 = pLastFrame->GetNavState().Get_BiasGyr();
    Vector3d ba_0 = pLastFrame->GetNavState().Get_BiasAcc();
    Vector3d bg_1 = pFrame->GetNavState().Get_BiasGyr();
    Vector3d ba_1 = pFrame->GetNavState().Get_BiasAcc();

    ceres::CostFunction* costFunction1 = ceres_slover::NavStateBiasError::Creat((InvCovBgaRW/imupreint.getDeltaTime()).inverse(),
                                                                                bg_0,ba_0,bg_1,ba_1);
    ceres::ResidualBlockId residualBlockId_bias = problem.AddResidualBlock(costFunction1,lossfunctionBias,pLastFrame->optimal_detla_bias_.data(),pFrame->optimal_detla_bias_.data());


    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
//    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);
    std::cout<<"PoseOptimization to Last Frame FullReport()->"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

//    std::abort();

    bool reject = true;
    if(reject)
    {
        int remove_count = 0;

        static const double TH_REPJ = 3.81 * pixel_usigma * pixel_usigma;
        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];

            if(!res_ids[i])
                continue;

            if(evaluateResidual<2>(problem, res_ids[i]).squaredNorm() > TH_REPJ * (1 << ft->level_))
            {
                remove_count++;
                problem.RemoveResidualBlock(res_ids[i]);
                pFrame->removeFeature(ft);
            }
        }

        ceres::Solve(options, &problem, &summary);
        std::cout<<"PoseOptimization to Last Frame after remove outliners FullReport()->"<<std::endl;
        std::cout<<summary.FullReport()<<std::endl;

    }

    NavState ns_recov;
    Vector3d optimal_Pwb_(pFrame->optimal_PR_[0],pFrame->optimal_PR_[1],pFrame->optimal_PR_[2]);
    Vector3d optimal_PHIwb_(pFrame->optimal_PR_[3],pFrame->optimal_PR_[4],pFrame->optimal_PR_[5]);
    Sophus::SO3d optimal_Rwb_ = Sophus::SO3d::exp(optimal_PHIwb_);

    ns_recov.Set_Pos(optimal_Pwb_);
    ns_recov.Set_Rot(optimal_Rwb_);
    ns_recov.Set_Vel(pFrame->optimal_v_);
    ns_recov.Set_BiasAcc(pFrame->GetNavState().Get_BiasAcc()+pFrame->optimal_detla_bias_.block(3,0,3,1));
    ns_recov.Set_BiasGyr(pFrame->GetNavState().Get_BiasGyr()+pFrame->optimal_detla_bias_.block(0,0,3,1));
    ns_recov.Set_DeltaBiasGyr(Vector3d::Zero());
    ns_recov.Set_DeltaBiasAcc(Vector3d::Zero());

    pFrame->SetNavState(ns_recov);
//    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    SE3d Twb_ = SE3d(optimal_Rwb_,optimal_Pwb_);
//    SE3d Tbc_ = SE3d(Tbc);
    pFrame->optimal_Tcw_ = (Twb_ * Tbc_).inverse();

    //! update pose
    pFrame->setTcw(pFrame->optimal_Tcw_);

    if(bComputeMarg)
    {

    }

}

// 跟踪上一帧关键帧
int Optimizer::PoseOptimization(Frame::Ptr pFrame, KeyFrame::Ptr pLastKF, const IMUPreintegrator &imupreint, const Vector3d &gw, const bool &bComputeMarg)
{
    const double focus_length = MIN(pFrame->cam_->fx(), pFrame->cam_->fy());
    const double pixel_usigma = Config::imagePixelSigma();
    static const size_t OPTIMAL_MPTS = 150;

    Matrix4d Tbc = ImuConfigParam::GetEigTbc();
    Matrix3d Rbc = Tbc.topLeftCorner(3,3);
    Vector3d Pbc = Tbc.topRightCorner(3,1);
    SE3d Tbc_ = SE3d(Tbc);
    // Gravity vector in world frame
    Vector3d GravityVec = gw;

    ceres::Problem problem;

//    pLastKF->UpdateNavStatePVRFromTcw(Tbc_);
    pLastKF->setOptimizationState();
//    pFrame->UpdateNavStatePVRFromTcw(Tbc_);
    pFrame->setOptimizationState();

    ceres::LocalParameterization* local_parameterization_pr = new ceres_slover::PRParameterization();
    problem.AddParameterBlock(pFrame->optimal_PR_.data(), 6, local_parameterization_pr);
    problem.AddParameterBlock(pFrame->optimal_v_.data(),3);
    problem.AddParameterBlock(pFrame->optimal_detla_bias_.data(),6);
    problem.AddParameterBlock(pLastKF->optimal_PR_.data(), 6, local_parameterization_pr);
    problem.AddParameterBlock(pLastKF->optimal_v_.data(),3);
    problem.AddParameterBlock(pLastKF->optimal_detla_bias_.data(),6);
    problem.SetParameterBlockConstant(pLastKF->optimal_PR_.data());
    problem.SetParameterBlockConstant(pLastKF->optimal_v_.data());
    problem.SetParameterBlockConstant(pLastKF->optimal_detla_bias_.data());
//    problem.SetParameterBlockConstant(pFrame->optimal_PR_.data());
//    problem.SetParameterBlockConstant(pFrame->optimal_detla_bias_.data());
    problem.SetParameterBlockConstant(pFrame->optimal_detla_bias_.data());



        static const double scale = pixel_usigma * std::sqrt(3.81);
    const double thHuberNavStatePRV = sqrt(100*21.666);
    const double thHuberNavStateBias = sqrt(100*16.812);
    ceres::LossFunction* lossfunctionMpt = new ceres::HuberLoss(scale);
    ceres::LossFunction* lossfunctionPRV = new ceres::HuberLoss(thHuberNavStatePRV);
    ceres::LossFunction* lossfunctionBias = new ceres::HuberLoss(thHuberNavStateBias);

    std::vector<Feature::Ptr> fts = pFrame->getFeatures();
    const size_t N = fts.size();
    std::vector<ceres::ResidualBlockId> res_ids(N);
    for(size_t i = 0; i < N; ++i)
    {
        Feature::Ptr ft = fts[i];
        MapPoint::Ptr mpt = ft->mpt_;
        if(mpt == nullptr || mpt->isBad())
            continue;
        mpt->optimal_pose_ = mpt->pose();
        ceres::CostFunction* cost_function = ceres_slover::PointReprojectionError::Creat( Tbc_ ,ft->fn_[0] / ft->fn_[2], ft->fn_[1] / ft->fn_[2]);
        res_ids[i] = problem.AddResidualBlock(cost_function, lossfunctionMpt, pFrame->optimal_PR_.data(), mpt->optimal_pose_.data());
        problem.SetParameterBlockConstant(mpt->optimal_pose_.data());
    }
    //是否使用种子点
    /*
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
    */

    Eigen::Matrix<double,6,6> InvCovBgaRW = Eigen::Matrix<double,6,6>::Identity();
    InvCovBgaRW.topLeftCorner(3,3) = Matrix3d::Identity()/IMUData::getGyrBiasRW2();       // Gyroscope bias random walk, covariance INVERSE
    InvCovBgaRW.bottomRightCorner(3,3) = Matrix3d::Identity()/IMUData::getAccBiasRW2();

    Matrix9d CovPRV = imupreint.getCovPVPhi();
    CovPRV.col(3).swap(CovPRV.col(6));
    CovPRV.col(4).swap(CovPRV.col(7));
    CovPRV.col(5).swap(CovPRV.col(8));
    CovPRV.row(3).swap(CovPRV.row(6));
    CovPRV.row(4).swap(CovPRV.row(7));
    CovPRV.row(5).swap(CovPRV.row(8));

    ceres::CostFunction* costFunction = ceres_slover::NavStatePRVError::Create(CovPRV,imupreint,GravityVec);
    ceres::ResidualBlockId residualBlockId_prv = problem.AddResidualBlock(costFunction,NULL,pLastKF->optimal_PR_.data(),pLastKF->optimal_v_.data(),pLastKF->optimal_detla_bias_.data(),
                                                                          pFrame->optimal_PR_.data(),pFrame->optimal_v_.data());
    Vector3d bg_0 = pLastKF->GetNavState().Get_BiasGyr();
    Vector3d ba_0 = pLastKF->GetNavState().Get_BiasAcc();
    Vector3d bg_1 = pFrame->GetNavState().Get_BiasGyr();
    Vector3d ba_1 = pFrame->GetNavState().Get_BiasAcc();

    ceres::CostFunction* costFunction1 = ceres_slover::NavStateBiasError::Creat((InvCovBgaRW/imupreint.getDeltaTime()).inverse(),
                                                                                bg_0,ba_0,bg_1,ba_1);
    ceres::ResidualBlockId residualBlockId_bias = problem.AddResidualBlock(costFunction1,lossfunctionBias,pLastKF->optimal_detla_bias_.data(),pFrame->optimal_detla_bias_.data());


    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
//    options.max_linear_solver_iterations = 20;

    ceres::Solve(options, &problem, &summary);

    std::cout<<"PoseOptimization to Last KeyFrame FullReport()->"<<std::endl;
    std::cout<<summary.FullReport()<<std::endl;

        std::cout<<"CovPRV:"<<std::endl<<CovPRV<<std::endl;

    bool reject = true;
    if(reject)
    {
        int remove_count = 0;

        static const double TH_REPJ = 3.81 * pixel_usigma * pixel_usigma;
        for(size_t i = 0; i < N; ++i)
        {
            Feature::Ptr ft = fts[i];
//            std::cout<<"evaluateResidual:"<<evaluateResidual<2>(problem, res_ids[i]).squaredNorm()<< "---"<<TH_REPJ * (1 << ft->level_)<<std::endl;
            if(!res_ids[i])
                continue;
            if(evaluateResidual<2>(problem, res_ids[i]).squaredNorm() > TH_REPJ * (1 << ft->level_))
            {
                remove_count++;
                problem.RemoveResidualBlock(res_ids[i]);
                pFrame->removeFeature(ft);
            }
        }

        ceres::Solve(options, &problem, &summary);
        std::cout<<"PoseOptimization to Last KeyFrame after remove outliners FullReport()->"<<std::endl;
        std::cout<<summary.FullReport()<<std::endl;

    }

    NavState ns_recov;
    Vector3d optimal_Pwb_(pFrame->optimal_PR_[0],pFrame->optimal_PR_[1],pFrame->optimal_PR_[2]);
    Vector3d optimal_PHIwb_(pFrame->optimal_PR_[3],pFrame->optimal_PR_[4],pFrame->optimal_PR_[5]);
    Sophus::SO3d optimal_Rwb_ = Sophus::SO3d::exp(optimal_PHIwb_);

    ns_recov.Set_Pos(optimal_Pwb_);
    ns_recov.Set_Rot(optimal_Rwb_);
    ns_recov.Set_Vel(pFrame->optimal_v_);
    ns_recov.Set_BiasAcc(pFrame->GetNavState().Get_BiasAcc()+pFrame->optimal_detla_bias_.block(3,0,3,1));
    ns_recov.Set_BiasGyr(pFrame->GetNavState().Get_BiasGyr()+pFrame->optimal_detla_bias_.block(0,0,3,1));
    ns_recov.Set_DeltaBiasGyr(Vector3d::Zero());
    ns_recov.Set_DeltaBiasAcc(Vector3d::Zero());

    pFrame->SetNavState(ns_recov);
//    pFrame->UpdatePoseFromNS(ConfigParam::GetMatTbc());

    SE3d Twb_ = SE3d(optimal_Rwb_,optimal_Pwb_);
//    SE3d Tbc_ = SE3d(Tbc);
    pFrame->optimal_Tcw_ = (Twb_ * Tbc_).inverse();

    //! update pose
    pFrame->setTcw(pFrame->optimal_Tcw_);
//    std::abort();

    if(bComputeMarg)
    {

    }
    //! Report
//    reportInfo<2>(problem, summary, report, verbose);
}





}




