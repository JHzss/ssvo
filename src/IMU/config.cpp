#include "config.hpp"

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
//    fSettings["imutopic"] >> _imuTopic;
//    fSettings["imagetopic"] >> _imageTopic;
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
