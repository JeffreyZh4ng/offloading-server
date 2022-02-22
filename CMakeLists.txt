cmake_minimum_required(VERSION 3.0)
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

project(simple_server)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(eCAL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(kimera_vio REQUIRED)

set(source_files
  simple_server.cpp
)

set(protobuf_files
    ${CMAKE_CURRENT_SOURCE_DIR}/proto_messages/vio_input.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/proto_messages/vio_output.proto
)

add_executable(${PROJECT_NAME} ${source_files})

PROTOBUF_TARGET_CPP(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/proto_messages/ ${protobuf_files})

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  eCAL::core
  protobuf::libprotobuf
  kimera_vio
)