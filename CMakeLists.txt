cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(HighQualityEllipseDetector)

set(Matlab_ROOT_DIR "c:/Program Files (x86)/MATLAB/R2016a")
find_package(Matlab)

#matlab_add_mex(NAME mex_file_name SRC source_file.cpp)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package(OpenCV REQUIRED)

include_directories("../Eigen")
include_directories(${OpenCV_INCLUDE_DIRS})
set(SOURCES
    main.cpp
    generateEllipseCandidates.cpp
    )

set(HEADERS
    )

SOURCE_GROUP("Source Files" FILES ${SOURCES})
SOURCE_GROUP("Header Files" FILES ${HEADERS})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set_target_properties( ${PROJECT_NAME} PROPERTIES  VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROJECT_NAME}>")
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(MEX1_SOURCES
    generateEllipseCandidates_mex.cpp
    generateEllipseCandidates.cpp
    )

set(MEX1_HEADERS
    )
SOURCE_GROUP("Mex Source Files" FILES ${MEX1_SOURCES})
SOURCE_GROUP("Mex Header Files" FILES ${MEX1_HEADERS})
matlab_add_mex(NAME generateEllipseCandidates SRC ${MEX1_SOURCES})
target_link_libraries(generateEllipseCandidates ${OpenCV_LIBS})

set(MEX2_SOURCES
    fitEllipse_mex.cpp
    )

set(MEX2_HEADERS
    )
SOURCE_GROUP("Mex Source Files" FILES ${MEX1_SOURCES})
SOURCE_GROUP("Mex Header Files" FILES ${MEX1_HEADERS})
matlab_add_mex(NAME fitEllipse SRC ${MEX2_SOURCES})
target_link_libraries(fitEllipse ${OpenCV_LIBS})