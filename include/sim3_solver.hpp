#ifndef _SSVO_SIM3_SOLVER_HPP_
#define _SSVO_SIM3_SOLVER_HPP_

#include "global.hpp"
#include "feature.hpp"
#include "map_point.hpp"
#include "optimizer.hpp"

namespace ssvo{

class Sim3Solver{
public:

    Sim3Solver(const SE3d Tcw1, const SE3d Tcw2, const std::vector<Feature::Ptr> &fts1, const std::vector<Feature::Ptr> &fts2, const bool scale_fixed) :
        scale_fixed_(scale_fixed)
    {
        const size_t N = fts1.size();
        LOG_ASSERT(N == fts2.size()) << "fts1(" << N << ") != fts2(" << fts2.size() << ")!";

        mpts1_.reserve(N);
        mpts2_.reserve(N);
        pxls1_.reserve(N);
        pxls2_.reserve(N);
        max_err1_.reserve(N);
        max_err2_.reserve(N);
        indices_.reserve(N);

        const Matrix3d Rcw1 = Tcw1.rotationMatrix();
        const Matrix3d Rcw2 = Tcw2.rotationMatrix();
        const Vector3d tcw1 = Tcw1.translation();
        const Vector3d tcw2 = Tcw2.translation();

        for(size_t i = 0; i < N; i++)
        {
            const Feature::Ptr &ft1 = fts1[i];
            const Feature::Ptr &ft2 = fts2[i];

            if(ft1->mpt_ == nullptr || ft2->mpt_ == nullptr)
                continue;
            if(ft1->mpt_->isBad() || ft2->mpt_->isBad())
                continue;

            mpts1_.push_back(Rcw1 * ft1->mpt_->pose() + tcw1);
            mpts2_.push_back(Rcw2 * ft2->mpt_->pose() + tcw2);

            pxls1_.push_back(ft1->fn_.head<2>()/ft1->fn_[2]);
            pxls1_.push_back(ft2->fn_.head<2>()/ft2->fn_[2]);

            const double sigm_square1 = 1 << ft1->level_;
            const double sigm_square2 = 1 << ft2->level_;

            max_err1_.push_back(9.210 * sigm_square1);
            max_err2_.push_back(9.210 * sigm_square2);

            indices_.push_back(i);
        }

        N_ = N;
        noMore_ = false;
    }

    bool runRANSAC(const int max_iterations = 5);

    void getEstimateSim3(Matrix3d &R, Vector3d &t, double &s, std::vector<bool> &inliers);

    static void computeSim3(const std::vector<Vector3d> &pts1, const std::vector<Vector3d> &pts2, Matrix3d &R12, Vector3d &t12, double &s12, bool scale_fixed = false);

    void SetRansacParameters(double probability = 0.99, int minInliers = 20 , int maxIterations = 300);


private:

    int checkInliers(Matrix3d R12, Vector3d t12, double s12, std::vector<bool>& inliers);

private:

    const bool scale_fixed_;

    size_t N_;
    std::vector<size_t> indices_;
    std::vector<Vector3d> mpts1_;
    std::vector<Vector3d> mpts2_;
    std::vector<Vector2d> pxls1_; //归一化平面中前两维
    std::vector<Vector2d> pxls2_;

    std::vector<double> max_err1_;
    std::vector<double> max_err2_;

    std::vector<bool> inliers_;

    Matrix3d R12_best_;
    Vector3d t12_best_;
    double s12_best_;
    int inliers_count_best_;

    int maxIterations_;
    double probability_;
    int min_inliers_;
    int iterations_;

public:
    bool noMore_;

};//! Sim3Solver


}


#endif //_SSVO_SIM3_SOLVER_HPP_
