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
include CMakeFiles/test_timer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_timer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_timer.dir/flags.make

CMakeFiles/test_timer.dir/test/test_timer.cpp.o: CMakeFiles/test_timer.dir/flags.make
CMakeFiles/test_timer.dir/test/test_timer.cpp.o: test/test_timer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jh/ssvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_timer.dir/test/test_timer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_timer.dir/test/test_timer.cpp.o -c /home/jh/ssvo/test/test_timer.cpp

CMakeFiles/test_timer.dir/test/test_timer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_timer.dir/test/test_timer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jh/ssvo/test/test_timer.cpp > CMakeFiles/test_timer.dir/test/test_timer.cpp.i

CMakeFiles/test_timer.dir/test/test_timer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_timer.dir/test/test_timer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jh/ssvo/test/test_timer.cpp -o CMakeFiles/test_timer.dir/test/test_timer.cpp.s

CMakeFiles/test_timer.dir/test/test_timer.cpp.o.requires:

.PHONY : CMakeFiles/test_timer.dir/test/test_timer.cpp.o.requires

CMakeFiles/test_timer.dir/test/test_timer.cpp.o.provides: CMakeFiles/test_timer.dir/test/test_timer.cpp.o.requires
	$(MAKE) -f CMakeFiles/test_timer.dir/build.make CMakeFiles/test_timer.dir/test/test_timer.cpp.o.provides.build
.PHONY : CMakeFiles/test_timer.dir/test/test_timer.cpp.o.provides

CMakeFiles/test_timer.dir/test/test_timer.cpp.o.provides.build: CMakeFiles/test_timer.dir/test/test_timer.cpp.o


# Object files for target test_timer
test_timer_OBJECTS = \
"CMakeFiles/test_timer.dir/test/test_timer.cpp.o"

# External object files for target test_timer
test_timer_EXTERNAL_OBJECTS =

bin/test_timer: CMakeFiles/test_timer.dir/test/test_timer.cpp.o
bin/test_timer: CMakeFiles/test_timer.dir/build.make
bin/test_timer: CMakeFiles/test_timer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jh/ssvo/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/test_timer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_timer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_timer.dir/build: bin/test_timer

.PHONY : CMakeFiles/test_timer.dir/build

CMakeFiles/test_timer.dir/requires: CMakeFiles/test_timer.dir/test/test_timer.cpp.o.requires

.PHONY : CMakeFiles/test_timer.dir/requires

CMakeFiles/test_timer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_timer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_timer.dir/clean

CMakeFiles/test_timer.dir/depend:
	cd /home/jh/ssvo && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo /home/jh/ssvo/CMakeFiles/test_timer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_timer.dir/depend

