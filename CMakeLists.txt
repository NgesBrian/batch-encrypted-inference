cmake_minimum_required(VERSION 3.5.1)
project(FHEON LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_STATIC "Build OpenFHE static libraries" OFF)

# -------------------------
# OpenFHE
# -------------------------
find_package(OpenFHE CONFIG REQUIRED)

message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
message(STATUS "OpenFHE include: ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE libdir: ${OpenFHE_LIBDIR}")

include_directories(
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
)

link_directories(${OpenFHE_LIBDIR})

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(EXTRA_LIBS stdc++fs)
endif()

if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    set(OPENFHE_LIBS ${OpenFHE_STATIC_LIBRARIES})
else()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    set(OPENFHE_LIBS ${OpenFHE_SHARED_LIBRARIES})
endif()

link_libraries(${OPENFHE_LIBS} ${EXTRA_LIBS})

# -------------------------
# Optional HEXL
# -------------------------
find_package(HEXL)
if (TARGET HEXL::hexl)
    message(STATUS "HEXL found")
    link_libraries(HEXL::hexl)
endif()

# -------------------------
# Common sources
# -------------------------
set(COMMON_SRCS
    src/Utils.h
    src/UtilsData.h
    src/FHEController.h src/FHEController.cpp
    src/ANNController.h src/ANNController.cpp
    src/ANNBatchController.h src/ANNBatchController.cpp
)

# -------------------------
# Helper function
# -------------------------
function(add_tresnet TARGET SRC DEFAULT_ARG)
    add_executable(${TARGET} ${SRC} ${COMMON_SRCS})
    target_compile_definitions(${TARGET} PRIVATE DEFAULT_ARG=${DEFAULT_ARG})
endfunction()

# -------------------------
# ResNet-20
# -------------------------
add_tresnet(tresnet20N16  ThroughputModels/TResNet20N16.cpp  100)
add_tresnet(tresnet20N32  ThroughputModels/TResNet20N32.cpp  100)
add_tresnet(tresnet20N64  ThroughputModels/TResNet20N64.cpp  100)
add_tresnet(tresnet20N128 ThroughputModels/TResNet20N128.cpp 1000)
add_tresnet(tresnet20N256 ThroughputModels/TResNet20N256.cpp 1000)
add_tresnet(tresnet20N512 ThroughputModels/TResNet20N512.cpp 1000)

# -------------------------
# ResNet-34
# -------------------------
add_tresnet(tresnet34N16  ThroughputModels/TResNet34N16.cpp  100)
add_tresnet(tresnet34N32  ThroughputModels/TResNet34N32.cpp  100)
add_tresnet(tresnet34N64  ThroughputModels/TResNet34N64.cpp  100)
add_tresnet(tresnet34N128 ThroughputModels/TResNet34N128.cpp 1000)
add_tresnet(tresnet34N256 ThroughputModels/TResNet34N256.cpp 1000)
