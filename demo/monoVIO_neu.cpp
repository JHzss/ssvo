#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <chrono>


#include "system.hpp"
using namespace ssvo;

std::list<ssvo::IMUData> vImus;
std::list<ssvo::IMAGEData> vImages;

std::condition_variable condition_variable;
std::mutex imu_lock;

struct Measurement
{
    Measurement(ssvo::IMAGEData imageData,std::vector<ssvo::IMUData> imuDatas):m_image(imageData),m_imus(imuDatas){}
    ssvo::IMAGEData m_image;
    std::vector<ssvo::IMUData> m_imus;
};

void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg)
{
    Eigen::Vector3d gyr(imu_msg->angular_velocity.x,imu_msg->angular_velocity.y,imu_msg->angular_velocity.z);
    Eigen::Vector3d acc(imu_msg->linear_acceleration.x,imu_msg->linear_acceleration.y,imu_msg->linear_acceleration.z);
    ssvo::IMUData imuData(gyr,acc,imu_msg->header.stamp.toSec()); //一定要用toSec()这个函数，不能直接使用.sec，uint32t强制转double会出现错误
    imu_lock.lock();
    vImus.push_back(imuData);
//    cout<<"vImus size():"<<vImus.size()<<endl;
    imu_lock.unlock();
    condition_variable.notify_one();
}

void image_callback(const sensor_msgs::ImageConstPtr& image_msg)
{
    cv_bridge::CvImageConstPtr image_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
    cv::Mat image = image_ptr->image;
    ssvo::IMAGEData imageData(image,image_ptr->header.stamp.toSec()); //一定要用toSec()这个函数，不能直接使用.sec，uint32t强制转double会出现错误
    vImages.push_back(imageData);
//    cout<<"vImages size():"<<vImages.size()<<endl;
}


std::vector<Measurement> getMeasurements()
{
    std::vector<Measurement> measurements;

    while(true)
    {
//        cout<<"measurements: "<<measurements.size()<<endl;

        if (vImus.empty() || vImages.empty())
            return measurements;

        if (vImus.back()._t <= vImages.front().timestamps_)
            return measurements;

        if (vImus.front()._t > vImages.front().timestamps_)
        {
            ROS_WARN("throw img, only should happen at the beginning");
            vImages.pop_front();
            continue;
        }
        ssvo::IMAGEData img_msg = vImages.front();
        vImages.pop_front();

        std::vector<ssvo::IMUData> IMUs;

        //! 第一帧对应的IMU不用额外处理，因为这部分不用计算积分的
        while (vImus.front()._t <= img_msg.timestamps_)
        {
            IMUs.emplace_back(vImus.front());

//            cout<<std::fixed<<std::setprecision(6)<<img_msg.timestamps_<<"----"<<vImus.front()._t<<endl;
            if((img_msg.timestamps_-vImus.front()._t)>1e-6)
                vImus.pop_front();
            else
                break;
        }

        measurements.push_back(Measurement(img_msg,IMUs));
    }

    return measurements;
}

void main_loop(const string& config_file, const string& calib_file )
{
    std::shared_ptr<System> vo = std::shared_ptr<System>(new System(config_file, calib_file));

    while(1)
    {
        std::vector<Measurement> measurements;
        std::unique_lock<std::mutex> lock(imu_lock);

        condition_variable.wait(lock,[&]
        {
            return (measurements = getMeasurements()).size() != 0;
        });

        lock.unlock();

        for(const auto& measurement:measurements)
        {
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

            cout<<"measurement.m_imus: "<<measurement.m_imus.size()<<endl;

            vo->process(measurement.m_image.image_,measurement.m_image.timestamps_,measurement.m_imus);

            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); //单位是秒

            double deltat = 1.0/ImuConfigParam::image_frequency_-ttrack;

            if(deltat>0)
                usleep(deltat*1e6);
        }
    }
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
//    LOG_ASSERT(argc == 4) << "\n Usage : ./monoVO_live config_file calib_file imu_config_file";

    ros::init(argc,argv,"ssvio");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string config_file,calib_file,imu_config_file;
    if(!n.getParam("config_file",config_file))
        ROS_ERROR_STREAM("Failed to load " << "config_file");
    if(!n.getParam("calib_file",calib_file))
        ROS_ERROR_STREAM("Failed to load " << "calib_file");
    if(!n.getParam("imu_config_file",imu_config_file))
        ROS_ERROR_STREAM("Failed to load " << "imu_config_file");


    ssvo::ImuConfigParam imuConfigParam(imu_config_file);

    cout<<"ImuConfigParam::imu_topic_: "<<ImuConfigParam::imu_topic_<<endl;
    cout<<"ImuConfigParam::image_topic_: "<<ImuConfigParam::image_topic_<<endl;


    ros::Subscriber sub_imu = n.subscribe(ImuConfigParam::imu_topic_,2000,imu_callback);
    ros::Subscriber sub_image = n.subscribe(ImuConfigParam::image_topic_,2000,image_callback);

    std::thread track_thread = std::thread(main_loop,config_file,calib_file);

    ros::spin();

    return 0;
}