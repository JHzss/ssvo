#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace ssvo{

using std::string;

class Config
{
public:

    static int imageNLevel(){return getInstance().image_nlevel_;}

    static double imagePixelSigma(){return getInstance().image_sigma_;}

    static int initMinCorners(){return getInstance().init_min_corners_;}

    static int initMinTracked(){return getInstance().init_min_tracked_;}

    static int initMinDisparity(){return getInstance().init_min_disparity_;}

    static int initMinInliers(){return getInstance().init_min_inliers_;}

    static int initMaxRansacIters(){return getInstance().init_max_iters_;}

    static int gridSize(){return getInstance().grid_size_;}

    static int gridMinSize(){return getInstance().grid_min_size_;}

    static int fastMaxThreshold(){return getInstance().fast_max_threshold_;}

    static int fastMinThreshold(){return getInstance().fast_min_threshold_;}

    static double fastMinEigen(){return getInstance().fast_min_eigen_;}

    static double mapScale(){return getInstance().mapping_scale_;}

    static int minConnectionObservations(){return getInstance().mapping_min_connection_observations_;}

    static int minCornersPerKeyFrame(){return getInstance().mapping_min_corners_;}

    static int maxReprojectKeyFrames(){return getInstance().mapping_max_reproject_kfs_;}

    static int maxLocalBAKeyFrames(){return getInstance().mapping_max_local_ba_kfs_;}

    static int minLocalBAConnectedFts(){return getInstance().mapping_min_local_ba_connected_fts_;}

    static int alignTopLevel(){return getInstance().align_top_level_;}

    static int alignBottomLevel(){return getInstance().align_bottom_level_;}

    static int alignPatchSize(){return getInstance().align_patch_size_;}

    static int maxTrackKeyFrames(){return getInstance().max_local_kfs_;}

    static int minQualityFts(){return getInstance().min_quality_fts_;}

    static int maxQualityDropFts(){return getInstance().max_quality_drop_fts_;}

    static int maxSeedsBuffer(){return getInstance().max_seeds_buffer_;}

    static int maxPerprocessKeyFrames(){return getInstance().max_perprocess_kfs_;}

    static string timeTracingDirectory(){return getInstance().time_trace_dir_;}

    static std::string DBoWDirectory(){return getInstance().dbow_dir_;}

//    static std::string IMU_TOPIC(){return getInstance().imu_topic_;}
//
//    static std::string IMAGE_TOPIC(){return getInstance().image_topic_;}


private:
    static Config& getInstance()
    {
        static Config instance(file_name_);
        return instance;
    }

    Config(string& file_name)
    {
        cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
        LOG_ASSERT(fs.isOpened()) << "Failed to open settings file at: " << file_name;

        //! Image
        image_nlevel_ = (int)fs["Image.nlevels"];
        image_sigma_ = (double)fs["Image.sigma"];
        image_sigma2_ = image_sigma_*image_sigma_;

        //! FAST detector parameters
        grid_size_ = (int)fs["FastDetector.grid_size"];
        grid_min_size_ = (int)fs["FastDetector.grid_min_size"];
        fast_max_threshold_ = (int)fs["FastDetector.fast_max_threshold"];
        fast_min_threshold_ = (int)fs["FastDetector.fast_min_threshold"];
        fast_min_eigen_ = (double)fs["FastDetector.fast_min_eigen"];

        //! initializer parameters
        init_min_corners_ = (int)fs["Initializer.min_corners"];
        init_min_tracked_ = (int)fs["Initializer.min_tracked"];
        init_min_disparity_ = (int)fs["Initializer.min_disparity"];
        init_min_inliers_ = (int)fs["Initializer.min_inliers"];
        init_max_iters_ = (int)fs["Initializer.ransac_max_iters"];

        //! map
        mapping_scale_ = (double)fs["Mapping.scale"];
        mapping_min_connection_observations_ = (int)fs["Mapping.min_connection_observations"];
        mapping_min_corners_ = (int)fs["Mapping.min_corners"];
        mapping_max_reproject_kfs_ = (int)fs["Mapping.max_reproject_kfs"];
        mapping_max_local_ba_kfs_ = (int)fs["Mapping.max_local_ba_kfs"];
        mapping_min_local_ba_connected_fts_ = (int)fs["Mapping.min_local_ba_connected_fts"];

        //! Align
        align_top_level_ = (int)fs["Align.top_level"];
        align_top_level_ = MIN(align_top_level_, image_nlevel_-1);
        align_bottom_level_ = (int)fs["Align.bottom_level"];
        align_bottom_level_ = MAX(align_bottom_level_, 0);
        align_patch_size_ = (int)fs["Align.patch_size"];

        //! Tracking
        max_local_kfs_ = (int)fs["Tracking.max_local_kfs"];
        min_quality_fts_ = (int)fs["Tracking.min_quality_fts"];
        max_quality_drop_fts_ = (int)fs["Tracking.max_quality_drop_fts"];

        //! ros topic
//        imu_topic_ = (std::string)fs["Ros.imu_topic"];
//        image_topic_ = (std::string)fs["Ros.image_topic"];


        max_seeds_buffer_ = (int)fs["DepthFilter.max_seeds_buffer"];
        max_perprocess_kfs_ = (int)fs["DepthFilter.max_perprocess_kfs"];

        //! glog
        if(!fs["Glog.alsologtostderr"].empty())
            fs["Glog.alsologtostderr"] >> FLAGS_alsologtostderr;

        if(!fs["Glog.colorlogtostderr"].empty())
            fs["Glog.colorlogtostderr"] >> FLAGS_colorlogtostderr;

        if(!fs["Glog.stderrthreshold"].empty())
            fs["Glog.stderrthreshold"] >> FLAGS_stderrthreshold;

        if(!fs["Glog.minloglevel"].empty())
            fs["Glog.minloglevel"] >> FLAGS_minloglevel;

        if(!fs["Glog.log_prefix"].empty())
            fs["Glog.log_prefix"] >> FLAGS_log_prefix;

        if(!fs["Glog.log_dir"].empty())
            fs["Glog.log_dir"] >> FLAGS_log_dir;

        //! Time Trace
        if(!fs["Trace.log_dir"].empty())
            fs["Trace.log_dir"] >> time_trace_dir_;

        //! DBoW
        if(!fs["DBoW.voc_dir"].empty())
            fs["DBoW.voc_dir"] >> dbow_dir_;

        fs.release();
    }

public:
    //! config file's name
    static string file_name_;

private:

    int image_nlevel_;
    double image_sigma_;
    double image_sigma2_;
    double image_unsigma_;
    double image_unsigma2_;

    //! FAST detector parameters
    int grid_size_;
    int grid_min_size_;
    int fast_max_threshold_;
    int fast_min_threshold_;
    double fast_min_eigen_;

    //! initializer parameters
    int init_min_corners_;
    int init_min_tracked_;
    int init_min_disparity_;
    int init_min_inliers_;
    int init_max_iters_;

    //! map
    double mapping_scale_;
    int mapping_min_connection_observations_;
    int mapping_min_corners_;
    int mapping_max_reproject_kfs_;
    int mapping_max_local_ba_kfs_;
    int mapping_min_local_ba_connected_fts_;

    //! Align
    int align_top_level_;
    int align_bottom_level_;
    int align_patch_size_;

    //! Tracking
    int max_local_kfs_;
    int min_quality_fts_;
    int max_quality_drop_fts_;

    //! DepthFilter
    int max_seeds_buffer_;
    int max_perprocess_kfs_;

    //! TimeTrace
    string time_trace_dir_;
    
    //! DBoW
    std::string dbow_dir_;

    //! ros topic
    std::string imu_topic_,image_topic_;

};

class ImuConfigParam
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuConfigParam(std::string configfile);

    // For good initialization (no movement at the beginning for some bag)
    double _testDiscardTime;

    static string imu_topic_;

    static string image_topic_;

    static int image_frequency_;

    static int imu_frequency_;

    static Eigen::Matrix4d GetEigTbc();

    static Eigen::Matrix4d GetEigTcb();

    static int GetLocalWindowSize();

    static double GetImageDelayToIMU();

    static bool GetAccMultiply9p8();

    static double GetG(){return _g;}

    std::string _bagfile;
//    std::string _imageTopic;
//    std::string _imuTopic;

    static std::string getTmpFilePath();
    static std::string _tmpFilePath;

    static double GetVINSInitTime(){return _nVINSInitTime;}
    static bool GetRealTimeFlag() {return _bRealTime;}

private:
    static Eigen::Matrix4d _EigTbc;

    static Eigen::Matrix4d _EigTcb;

    static int _LocalWindowSize;
    static double _ImageDelayToIMU;
    static bool _bAccMultiply9p8;

    static double _g;
    static double _nVINSInitTime;
    static bool _bRealTime;
};

}

#endif