#include "config.hpp"
#include "imudata.hpp"
#include <global.hpp>

namespace ssvo
{
//! 重力值
double ImuConfigParam::_g = 9.810;
//! imu和相机的外参
Eigen::Matrix4d ImuConfigParam::_EigTbc = Eigen::Matrix4d::Identity();
Eigen::Matrix4d ImuConfigParam::_EigTcb = Eigen::Matrix4d::Identity();
//! 滑动窗口大小
int ImuConfigParam::_LocalWindowSize = 20;
//! 相机和imu时间延迟
double ImuConfigParam::_ImageDelayToIMU = 0;
//! acc=acc*9.8, if below is 1
bool ImuConfigParam::_bAccMultiply9p8 = false;
//! Path to save tmp files/results
std::string ImuConfigParam::_tmpFilePath = "/home/jh/ssvo_imu";
//! imu 相机初始化时间
double ImuConfigParam::_nVINSInitTime = 15.0;
//! 是否实时
bool ImuConfigParam::_bRealTime = true;

string ImuConfigParam::imu_topic_;

string ImuConfigParam::image_topic_;

int ImuConfigParam::imu_frequency_;
int ImuConfigParam::image_frequency_;


    ImuConfigParam::ImuConfigParam(std::string configfile)
{
    cv::FileStorage fSettings(configfile, cv::FileStorage::READ);

    std::cout<<std::endl<<std::endl<<"Parameters: "<<std::endl;

    _testDiscardTime = fSettings["test.DiscardTime"];

    _nVINSInitTime = fSettings["test.VINSInitTime"];

    std::cout<<"================ Parameter with IMU ===================="<<std::endl;
    std::cout<<"VINS initialize time: "<<_nVINSInitTime<<std::endl;
    std::cout<<std::endl;
    std::cout<<"Discart time in test data: "<<_testDiscardTime<<std::endl;
    std::cout<<std::endl;
    fSettings["test.InitVIOTmpPath"] >> _tmpFilePath;
    std::cout<<"save tmp file in "<<_tmpFilePath<<std::endl;
    std::cout<<std::endl;
    fSettings["bagfile"] >> _bagfile;
    std::cout<<"open rosbag: "<<_bagfile<<std::endl;
    std::cout<<std::endl;
    fSettings["imu_topic"] >> imu_topic_;
    fSettings["image_topic"] >> image_topic_;
    fSettings["imu_topic"] >> imu_topic_;
    fSettings["image_topic"] >> image_topic_;
    fSettings["imu_frequency"] >> imu_frequency_;
    fSettings["image_frequency"] >> image_frequency_;
//    std::cout<<"imu topic: "<<_imuTopic<<std::endl;
//    std::cout<<"image topic: "<<_imageTopic<<std::endl;

    _LocalWindowSize = fSettings["LocalMapping.LocalWindowSize"];
    std::cout<<"local window size: "<<_LocalWindowSize<<std::endl;
    std::cout<<std::endl;
    _ImageDelayToIMU = fSettings["Camera.delaytoimu"];
    std::cout<<"timestamp image delay to imu: "<<_ImageDelayToIMU<<std::endl;
    std::cout<<std::endl;
    {
        cv::FileNode Tbc_ = fSettings["Camera.Tbc"];
        Eigen::Matrix3d R;
        R <<   Tbc_[0], Tbc_[1], Tbc_[2],
                Tbc_[4], Tbc_[5], Tbc_[6],
                Tbc_[8], Tbc_[9], Tbc_[10];
        Eigen::Quaterniond qr(R);
        R = qr.normalized().toRotationMatrix();


        Eigen::Matrix<double,3,1> t( Tbc_[3], Tbc_[7], Tbc_[11]);
        _EigTbc = Eigen::Matrix4d::Identity();
        _EigTbc.block<3,3>(0,0) = R;
        _EigTbc.block<3,1>(0,3) = t;

        _EigTcb = Eigen::Matrix4d::Identity();
        _EigTcb.block<3,3>(0,0) = R.transpose();
        _EigTcb.block<3,1>(0,3) = -R.transpose()*t;

        // Tbc_[0], Tbc_[1], Tbc_[2], Tbc_[3], Tbc_[4], Tbc_[5], Tbc_[6], Tbc_[7], Tbc_[8], Tbc_[9], Tbc_[10], Tbc_[11], Tbc_[12], Tbc_[13], Tbc_[14], Tbc_[15];
        std::cout<<"Tbc inited:"<<std::endl<<_EigTbc<<std::endl;
        std::cout<<"Tcb inited:"<<std::endl<<_EigTcb<<std::endl;
        std::cout<<"Tbc*Tcb:"<<std::endl<<_EigTbc*_EigTcb<<std::endl;
    }
    std::cout<<std::endl;
    {
        int tmpBool = fSettings["IMU.multiplyG"];
        _bAccMultiply9p8 = (tmpBool != 0);
        std::cout<<"whether acc*9.8? 0/1: "<<_bAccMultiply9p8<<std::endl;
    }
    std::cout<<std::endl;
    {
        int tmpBool = fSettings["test.RealTime"];
        _bRealTime = (tmpBool != 0);
        std::cout<<"whether run realtime? 0/1: "<<_bRealTime<<std::endl;
    }

    double gyrBiasRw = fSettings["IMU.gyrBiasRW"];
    double accBiasRw = fSettings["IMU.accBiasRW"];
    double gyrMeasCov = fSettings["IMU.gyrNoise"];
    double accMeasCov = fSettings["IMU.accNoise"];
    double frequency = fSettings["IMU.frequency"];

    int gyrBiasRw_p = fSettings["IMU.gyrBiasRW_p"];
    int accBiasRw_p = fSettings["IMU.accBiasRW_p"];
    int gyrMeasCov_p = fSettings["IMU.gyrNoise_p"];
    int accMeasCov_p = fSettings["IMU.accNoise_p"];
    cout<<"/********"<<endl
        <<"* IMU.frequency: "<<frequency<<endl
        <<"* IMU.gyrBiasRW: "<<gyrBiasRw<<endl<<"* IMU.accBiasRW: "<<accBiasRw<<endl
        <<"* IMU.gyrMeasCov: "<<gyrMeasCov<<endl<<"* IMU.accMeasCov: "<<accMeasCov<<endl
        <<"* IMU.gyrBiasRW_p: "<<gyrBiasRw_p<<endl<<"* IMU.accBiasRW_p: "<<accBiasRw_p<<endl
        <<"* IMU.gyrMeasCov_p: "<<gyrMeasCov_p<<endl<<"* IMU.accMeasCov_p: "<<accMeasCov_p<<endl
        <<"********/"<<endl;

    IMUData::_gyrBiasRw2 = gyrBiasRw * gyrBiasRw/**10*/;  //2e-12*1e3
    IMUData::_accBiasRw2 = accBiasRw * accBiasRw/**10*/;  //4.5e-8*1e2

    //MH01 *200 *1000  --- 0.039/ *1 *500 0.042
    //MH02 *200 *10000
    //MH03 *200 *10000
    //MH04 *1 *10000   /  *200 *200 ---0.16 /*200 *10000 ---0.16/ *1 *1 ---0.63 / *1 *100 ---0.27

    // 方差+离散->sigma_gw * sigma_gw / dt * (调参->标定的IMU不准确，可能相差数量级，因此需要加上一个参数)
    IMUData::_gyrMeasCov = Eigen::Matrix3d::Identity() * gyrMeasCov * gyrMeasCov * frequency * gyrMeasCov_p;       // sigma_g * sigma_g / dt, ~6e-6*10
    IMUData::_accMeasCov = Eigen::Matrix3d::Identity() * accMeasCov * accMeasCov * frequency * accMeasCov_p;       // sigma_a * sigma_a / dt, ~8e-4*10

    // covariance of bias random walk
    // 方差+离散->sigma_gw * sigma_gw * dt
    IMUData::_gyrBiasRWCov = Eigen::Matrix3d::Identity()*IMUData::_gyrBiasRw2 / frequency * gyrBiasRw_p;     // sigma_gw * sigma_gw * dt, ~2e-12
    IMUData::_accBiasRWCov = Eigen::Matrix3d::Identity()*IMUData::_accBiasRw2 / frequency * accBiasRw_p;     // sigma_aw * sigma_aw * dt, ~4.5e-8

    std::cout<<"================ END ===================="<<std::endl;
}

std::string ImuConfigParam::getTmpFilePath()
{
    return _tmpFilePath;
}

Eigen::Matrix4d ImuConfigParam::GetEigTbc()
{
    return _EigTbc;
}

Eigen::Matrix4d ImuConfigParam::GetEigTcb()
{
    return _EigTcb;
}



int ImuConfigParam::GetLocalWindowSize()
{
    return _LocalWindowSize;
}

double ImuConfigParam::GetImageDelayToIMU()
{
    return _ImageDelayToIMU;
}

bool ImuConfigParam::GetAccMultiply9p8()
{
    return _bAccMultiply9p8;
}

}
