# ssvo

Semi-direct sparse odometry

### 1.Prerequisites

#### 1.1 [OpenCV](http://opencv.org)
OpenCV 3.1.0 is used in this code.

#### 1.2 [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```shell
sudo apt-get install libeigen3-dev
```

#### 1.3 [Sophus](https://github.com/strasdat/Sophus)
This code use the template implement of Sophus. Recommend the latest released version [v1.0.0](https://github.com/strasdat/Sophus/tree/v1.0.0)

#### 1.4 [glog](https://github.com/google/glog)
This code use glog for logging.
```shell
sudo apt-get install libgoogle-glog-dev
```

#### 1.5 [Ceres](http://ceres-solver.org/installation.html)
Use Ceres-Slover to slove bundle adjustment. Please follow the [installation page](http://ceres-solver.org/installation.html#section-customizing) to install.

#### 1.6 [Pangolin](https://github.com/stevenlovegrove/Pangolin)
This code use Pangolin to display the map reconstructed. When install, just follow the [README.md](https://github.com/stevenlovegrove/Pangolin/blob/master/README.md) file.

#### 1.7 [DBow3](https://github.com/kokerf/DBow3)
After build and install, copy the file `FindDBoW3.cmake` to the directory `cmake_modules`
