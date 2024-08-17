cmake_minimum_required(VERSION 3.5)

project(LaneDetectorInfer LANGUAGES CXX)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(BoldRed "${Esc}[1;31m")
set(BoldGreen "${Esc}[1;32m")
set(BoldYellow "${Esc}[1;33m")
set(BoldMagenta "${Esc}[1;35m")
set(BoldCyan "${Esc}[1;36m")
set(BoldWhite "${Esc}[1;37m")

# CMAKE_C_COMPILER
if(NOT CMAKE_C_COMPILER)
    if(APPLE)
        set(CMAKE_C_COMPILER /usr/bin/clang)
    elseif(UNIX AND NOT APPLE)
        set(CMAKE_C_COMPILER /usr/bin/gcc)
    elseif(WIN32)
        set(CMAKE_C_COMPILER C:/mingw64/bin/x86_64-w64-mingw32-gcc.exe)
    else()
        message(FATAL_ERROR "Unsupported platform, please set CMAKE_C_COMPILER")
    endif()
endif()

message(STATUS "[INFO] C compiler: ${CMAKE_C_COMPILER}")

# CMAKE_CXX_COMPILER
if(NOT CMAKE_CXX_COMPILER)
    if(APPLE)
        set(CMAKE_CXX_COMPILER /usr/bin/clang++)
    elseif(UNIX AND NOT APPLE)
        set(CMAKE_CXX_COMPILER /usr/bin/g++)
    elseif(WIN32)
        set(CMAKE_CXX_COMPILER C:/mingw64/bin/x86_64-w64-mingw32-g++.exe)
    else()
        message(FATAL_ERROR "Unsupported platform, please set CMAKE_CXX_COMPILER")
    endif()
endif()

message(STATUS "[INFO] CXX compiler: ${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------
# [dependencies] gflags
# ------------------------------------------
find_package(gflags REQUIRED COMPONENTS static)
include_directories(${GFLAGS_INCLUDE_DIR})
link_directories(${gflags_LIBS})
message(STATUS "[INFO] gflags status:")
message(STATUS "       include  : ${GFLAGS_INCLUDE_DIR}")
message(STATUS "       lib      : ${GFLAGS_LIBRARIES}")

# ------------------------------------------
# [dependencies] opencv
# ------------------------------------------
# find_package(OpenCV 4 REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# link_directories(${OpenCV_LIBS})
# message(STATUS "[INFO] OpenCV status:")
# message(STATUS "       version  : ${OpenCV_VERSION}")
# message(STATUS "       include  : ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "       libs     : ${OpenCV_LIBS}")

# ------------------------------------------
# [dependencies] onnxruntime
# ------------------------------------------
set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_HOME}) # 如果环境变量没有设置，需要指定路径或者设置环境变量 ONNXRUNTIME_HOME
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib) # 目录下有 libonnxruntime.a
set(ONNXRUNTIME_LIBS onnxruntime) # 在 ${ONNXRUNTIME_DIR}/lib 目录下有 libonnxruntime.a

# ------------------------------------------
# set output path
# ------------------------------------------
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(EXECUTABLE_NAME infer-onnx-image)
add_executable(${EXECUTABLE_NAME} infer-onnx-image.cpp)
target_link_libraries(${EXECUTABLE_NAME}
    ${GFLAGS_LIBRARIES}
    ${ONNXRUNTIME_LIBS}
)
unset(EXECUTABLE_NAME)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/weights
    DESTINATION ${EXECUTABLE_OUTPUT_PATH}
)