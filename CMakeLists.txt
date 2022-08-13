# CMake version check
cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE Release)
# Main Project Name
project(framediff)
#project(magnification)

# set path of OpenCV and libtorch
set(OpenCV_DIR "/home/zengruilin/opencv/lib/cmake/opencv4")
set(Torch_DIR "/home/zengruilin/libtorch/share/cmake/Torch")
set(TORCH_INCLUDE_DIRS "/home/zengruilin/libtorch/include")

# Dependent packages
find_package(OpenCV REQUIRED core imgproc highgui)
find_package(Torch REQUIRED)

# Includes
include_directories( ${OpenCV_INCLUDE_DIRS}
	${PROJECT_BINARY_DIR}/include/
	${TORCH_INCLUDE_DIRS}
)
# Executable file directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
#Executable for main target
add_executable(framediff
	src/framediff.cpp
)
# Link libraries
target_link_libraries(${PROJECT_NAME}
	${OpenCV_LIBS}
	${TORCH_LIBRARIES}
	${PROJECT_SOURCE_DIR}
)
# Executable for main target
#add_executable(magnification
#	src/outputframe.cpp
#	src/getframe.cpp
#	src/main.cpp
#	src/setting.cpp
#    src/EVM.cpp
#	src/flow.cpp
#)
# Link libraries
#target_link_libraries(${PROJECT_NAME}
#	${OpenCV_LIBS}
#	${TORCH_LIBRARIES}
#	${PROJECT_SOURCE_DIR}
#)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
