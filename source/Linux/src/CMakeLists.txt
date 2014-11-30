#find_package(MySQL REQUIRED)
find_package(OpenCV REQUIRED)

MESSAGE(STATUS "OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
MESSAGE(STATUS "OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})

find_path(BOINC_INCLUDE_DIR config.h PATHS /boinc/src/boinc/ /boinc/)

include_directories(
    ${PROJECT_SOURCE_DIR}
    /usr/local/include
    ${OpenCV_INCLUDE_DIRS}
    #	${MYSQL_INCLUDE_DIR}
)

add_executable(frogmatcher
    frogmatcher
)

target_link_libraries(frogmatcher ${OpenCV_LIBRARIES})
