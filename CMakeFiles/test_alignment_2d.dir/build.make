# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jh/clion-2016.3.4/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/jh/clion-2016.3.4/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jh/ssvo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jh/ssvo

# Include any dependencies generated for this target.
include CMakeFiles/test_alignment_2d.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_alignment_2d.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_alignment_2d.dir/flags.make

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o: CMakeFiles/test_alignment_2d.dir/flags.make
CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o: test/test_alignment_2d.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/ssvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o -c /home/jh/ssvo/test/test_alignment_2d.cpp

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/ssvo/test/test_alignment_2d.cpp > CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.i

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/ssvo/test/test_alignment_2d.cpp -o CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.s

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.requires:

.PHONY : CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.requires

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.provides: CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_alignment_2d.dir/build.make CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.provides.build
.PHONY : CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.provides

CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.provides.build: CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o


CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o: CMakeFiles/test_alignment_2d.dir/flags.make
CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o: src/feature_alignment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/ssvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o -c /home/jh/ssvo/src/feature_alignment.cpp

CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/ssvo/src/feature_alignment.cpp > CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.i

CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/ssvo/src/feature_alignment.cpp -o CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.s

CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.requires:

.PHONY : CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.requires

CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.provides: CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_alignment_2d.dir/build.make CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.provides.build
.PHONY : CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.provides

CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.provides.build: CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o


# Object files for target test_alignment_2d
test_alignment_2d_OBJECTS = \
"CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o" \
"CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o"

# External object files for target test_alignment_2d
test_alignment_2d_EXTERNAL_OBJECTS =

bin/test_alignment_2d: CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o
bin/test_alignment_2d: CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o
bin/test_alignment_2d: CMakeFiles/test_alignment_2d.dir/build.make
bin/test_alignment_2d: lib/libssvo.a
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_shape.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_stitching.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_objdetect.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_superres.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_videostab.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_calib3d.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_features2d.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_flann.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_highgui.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_ml.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_photo.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_video.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_videoio.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_imgcodecs.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_imgproc.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_viz.so.3.2.0
bin/test_alignment_2d: /home/jh/opencv-3.2.0/build/lib/libopencv_core.so.3.2.0
bin/test_alignment_2d: /usr/local/lib/libceres.a
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libglog.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libgflags.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/test_alignment_2d: /usr/lib/libtbb.so
bin/test_alignment_2d: /usr/lib/libtbbmalloc.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libamd.so
bin/test_alignment_2d: /usr/lib/liblapack.so
bin/test_alignment_2d: /usr/lib/libf77blas.so
bin/test_alignment_2d: /usr/lib/libatlas.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/librt.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/test_alignment_2d: /usr/lib/libtbb.so
bin/test_alignment_2d: /usr/lib/libtbbmalloc.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libamd.so
bin/test_alignment_2d: /usr/lib/liblapack.so
bin/test_alignment_2d: /usr/lib/libf77blas.so
bin/test_alignment_2d: /usr/lib/libatlas.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/librt.so
bin/test_alignment_2d: /usr/local/lib/libpangolin.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGL.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libSM.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libICE.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libX11.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libXext.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGLEW.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGL.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libSM.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libICE.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libX11.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libXext.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libGLEW.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libpython2.7.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libdc1394.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libavcodec.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libavformat.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libavutil.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libswscale.so
bin/test_alignment_2d: /usr/lib/libOpenNI.so
bin/test_alignment_2d: /usr/lib/libOpenNI2.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libpng.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libz.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libjpeg.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libtiff.so
bin/test_alignment_2d: /usr/lib/x86_64-linux-gnu/libIlmImf.so
bin/test_alignment_2d: Thirdparty/fast/build/libfast.a
bin/test_alignment_2d: CMakeFiles/test_alignment_2d.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jh/ssvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bin/test_alignment_2d"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_alignment_2d.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_alignment_2d.dir/build: bin/test_alignment_2d

.PHONY : CMakeFiles/test_alignment_2d.dir/build

CMakeFiles/test_alignment_2d.dir/requires: CMakeFiles/test_alignment_2d.dir/test/test_alignment_2d.cpp.o.requires
CMakeFiles/test_alignment_2d.dir/requires: CMakeFiles/test_alignment_2d.dir/src/feature_alignment.cpp.o.requires

.PHONY : CMakeFiles/test_alignment_2d.dir/requires

CMakeFiles/test_alignment_2d.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_alignment_2d.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_alignment_2d.dir/clean

CMakeFiles/test_alignment_2d.dir/depend:
	cd /home/jh/ssvo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo/CMakeFiles/test_alignment_2d.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_alignment_2d.dir/depend
