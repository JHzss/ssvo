#include "system.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 5) << "\n Usage : ./monoVO_tum config_file calib_file dataset_path association_file ";

    System vo(argv[1], argv[2]);

    TUMDataReader dataset(argv[3],argv[4]);

    ssvo::Timer<std::micro> timer;
    const size_t N = dataset.N;
    for(size_t i = 0; i < N; i++)
    {
        string rgb_,depth_;
        double timestamp;
        if(!dataset.readItemByIndex(i,rgb_,depth_,timestamp))
            std::cerr<<"NO image to load"<<std::endl;

        LOG(INFO) << "=== Load Image " << i << ": " << rgb_ << ", time: " << std::fixed <<std::setprecision(7)<< timestamp << std::endl;
        cv::Mat image = cv::imread(rgb_, CV_LOAD_IMAGE_UNCHANGED);
        if(image.empty())
            continue;

        timer.start();
        vo.process(image, timestamp);
        timer.stop();

        double time_process = timer.duration();

        double time_wait = 0;
        if(i < N -1)
            time_wait = (dataset.timestamps_[i+1] - dataset.timestamps_[i])*1e6;
        else
            time_wait = (dataset.timestamps_[i] - dataset.timestamps_[i-1])*1e6;

        if(time_process < time_wait)
            std::this_thread::sleep_for(std::chrono::microseconds((int)(time_wait - time_process)));


    }

    vo.saveTrajectoryTUM("trajectory.txt");
    getchar();

    return 0;
}