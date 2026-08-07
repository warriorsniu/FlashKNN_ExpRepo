# All rights reserved by 
# Durga Keerthi Mandarapu, Vani Nagarajan, Artem Pelenitsyn, and Milind Kulkarni. 2024. 
# Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing. 

add_compile_definitions(KN=${KN})
add_compile_definitions(NORM=${NORM})
add_definitions(-DENHANCE)

embed_ptx(
  OUTPUT_TARGET
    sample01-knn-ptx
  PTX_LINK_LIBRARIES
    owl::owl
  SOURCES
    deviceCode.cu
)

add_executable(sample01-knn hostCode.cpp)
target_link_libraries(sample01-knn
  PRIVATE
    sample01-knn-ptx
    owl::owl
)
