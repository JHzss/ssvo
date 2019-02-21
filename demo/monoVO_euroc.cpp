#include "system.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"
#include "src/IMU/imudata.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 5) << "\n Usage : ./monoVO_dataset config_file calib_file dataset_path imu_config_file";

    System vo(argv[1], argv[2]);
    ssvo::ImuConfigParam ImuConfigParam(argv[4]);

    EuRocDataReader dataset(argv[3]);

    std::vector<ssvo::IMUData> vImus;

    for(auto it:dataset.imu_)
    {
        vImus.push_back(IMUData(it.gyro[0],it.gyro[1],it.gyro[2],it.acc[0],it.acc[1],it.acc[2],it.timestamp));
    }
    int nImus = vImus.size();
    std::cout << "Imus in data: " << nImus << std::endl;
    if(nImus<=0)
    {
        std::cerr << "ERROR: Failed to load imus" << std::endl;
        return 1;
    }


    ssvo::Timer<std::micro> timer;
    const size_t N = dataset.leftImageSize();
    bool test_delay = true;
    long imuindex = 0;
    for(size_t i = 0; i < N; i++)
    {
//        cv::waitKey(0);
        const EuRocDataReader::Image image_data = dataset.leftImage(i);
        LOG(INFO) << "=== Load Image " << i << ": " << image_data.path << ", time: " << std::fixed <<std::setprecision(7)<< image_data.timestamp << std::endl;
        cv::Mat image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
        if(image.empty())
            continue;

        if(test_delay)
        {
            const double startimutime = vImus[0]._t;
            if(startimutime <= image_data.timestamp)
            {
                test_delay = false;
            }
            else
            {
                continue;
            }
        }
        double tframe = image_data.timestamp;

        std::vector<ssvo::IMUData> vimu;
        while(1)
        {
            const ssvo::IMUData& imudata = vImus[imuindex];
            if(imudata._t > tframe)
                break;
            vimu.push_back(imudata);
            if(imudata._t == tframe)
                break;
            imuindex++;
        }

//        for(int i = 0;i<vimu.size();i++)
//        {
//            ssvo::IMUData imu = vimu[i];
//            std::cout<<std::setprecision(6);
//            std::cout<<std::fixed<<imu._t<<std::endl;
//        }
//        std::cout<<std::fixed<<"image: "<<image_data.timestamp<<std::endl;

        timer.start();
        bool use_vi = true;
        if(use_vi)
            vo.process(image, image_data.timestamp,vimu);
        else
            vo.process(image, image_data.timestamp);
        timer.stop();

        double time_process = timer.duration();

        double time_wait = 0;
        if(i < N -1)
            time_wait = (dataset.leftImage(i+1).timestamp - image_data.timestamp)*1e6;
        else
            time_wait = (image_data.timestamp - dataset.leftImage(i-1).timestamp)*1e6;

        if(time_process < time_wait)
            std::this_thread::sleep_for(std::chrono::microseconds((int)(time_wait - time_process)));
    }

    vo.saveTrajectoryTUM("trajectory.txt");
//    getchar();

    return 0;
}