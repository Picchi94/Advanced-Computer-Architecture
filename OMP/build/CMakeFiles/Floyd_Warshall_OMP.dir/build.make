# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build"

# Include any dependencies generated for this target.
include CMakeFiles/Floyd_Warshall_OMP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Floyd_Warshall_OMP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Floyd_Warshall_OMP.dir/flags.make

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/main.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/main.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/main.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o: ../src/FloydWarshall.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshall.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshall.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshall.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o: ../src/FloydWarshallOMP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshallOMP.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshallOMP.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/FloydWarshallOMP.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o: ../src/Graph/GraphBase.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphBase.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphBase.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphBase.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o: ../src/Graph/GraphStd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStd.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStd.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStd.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o: ../src/Graph/GraphStdRead.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStdRead.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStdRead.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphStdRead.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o: ../src/Graph/GraphWeight.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeight.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeight.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeight.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o: ../src/Graph/GraphWeightRead.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeightRead.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeightRead.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Graph/GraphWeightRead.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o: ../src/Host/FileUtil.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/FileUtil.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/FileUtil.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/FileUtil.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o


CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o: CMakeFiles/Floyd_Warshall_OMP.dir/flags.make
CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o: ../src/Host/PrintExt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o"
	/storage/fpicchirallo/clang/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o -c "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/PrintExt.cpp"

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.i"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/PrintExt.cpp" > CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.i

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.s"
	/storage/fpicchirallo/clang/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/src/Host/PrintExt.cpp" -o CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.s

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.requires:

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.requires

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.provides: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.requires
	$(MAKE) -f CMakeFiles/Floyd_Warshall_OMP.dir/build.make CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.provides.build
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.provides

CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.provides.build: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o


# Object files for target Floyd_Warshall_OMP
Floyd_Warshall_OMP_OBJECTS = \
"CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o" \
"CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o"

# External object files for target Floyd_Warshall_OMP
Floyd_Warshall_OMP_EXTERNAL_OBJECTS =

Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/build.make
Floyd_Warshall_OMP: CMakeFiles/Floyd_Warshall_OMP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable Floyd_Warshall_OMP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Floyd_Warshall_OMP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Floyd_Warshall_OMP.dir/build: Floyd_Warshall_OMP

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/build

CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/main.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshall.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/FloydWarshallOMP.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphBase.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStd.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphStdRead.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeight.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Graph/GraphWeightRead.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/FileUtil.cpp.o.requires
CMakeFiles/Floyd_Warshall_OMP.dir/requires: CMakeFiles/Floyd_Warshall_OMP.dir/src/Host/PrintExt.cpp.o.requires

.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/requires

CMakeFiles/Floyd_Warshall_OMP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Floyd_Warshall_OMP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/clean

CMakeFiles/Floyd_Warshall_OMP.dir/depend:
	cd "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP" "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP" "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build" "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build" "/storage/fpicchirallo/Progetto Architetture Avanzate/OMP/build/CMakeFiles/Floyd_Warshall_OMP.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/Floyd_Warshall_OMP.dir/depend

