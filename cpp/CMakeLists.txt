cmake_minimum_required(VERSION 3.25)
project(mir_drum_isolator VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++20 standard — required for std::atomic, std::jthread, std::span, etc.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# vcpkg-provided packages (PortAudio, libsndfile, FFTW3)
# vcpkg toolchain must be passed via:
#   -DCMAKE_TOOLCHAIN_FILE=<vcpkg_root>/scripts/buildsystems/vcpkg.cmake
# ---------------------------------------------------------------------------
find_package(portaudio CONFIG REQUIRED)
find_package(SndFile CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)  # single-precision fftwf_* used in inference_engine.cpp

# ---------------------------------------------------------------------------
# ONNX Runtime — fetched from official GitHub prebuilt release.
# vcpkg's onnxruntime port lags behind and has inconsistent Windows CPU
# prebuilt support, so we fetch the binary directly.
# ---------------------------------------------------------------------------
include(FetchContent)

set(ORT_VERSION "1.18.1")

if(WIN32)
  set(ORT_ARCHIVE_NAME "onnxruntime-win-x64-${ORT_VERSION}.zip")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")
  set(ORT_SHA256 "")   # Set a known SHA256 after first download for reproducibility
elseif(APPLE)
  set(ORT_ARCHIVE_NAME "onnxruntime-osx-x86_64-${ORT_VERSION}.tgz")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")
else()
  set(ORT_ARCHIVE_NAME "onnxruntime-linux-x64-${ORT_VERSION}.tgz")
  set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE_NAME}")
endif()

FetchContent_Declare(
  onnxruntime
  URL      ${ORT_URL}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

# Build an IMPORTED target so the rest of the CMake file treats ORT
# identically to a find_package result.
add_library(OnnxRuntime::OnnxRuntime SHARED IMPORTED)

if(WIN32)
  set(ORT_BASE_DIR "${onnxruntime_SOURCE_DIR}")
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_BASE_DIR}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB               "${ORT_BASE_DIR}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_BASE_DIR}/include"
  )
else()
  set(ORT_BASE_DIR "${onnxruntime_SOURCE_DIR}")
  set_target_properties(OnnxRuntime::OnnxRuntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_BASE_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_BASE_DIR}/include"
  )
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(MIR_SOURCES
  src/main.cpp
  src/audio_reader.cpp
  src/inference_engine.cpp
)

set(MIR_HEADERS
  include/constants.h
  include/ring_buffer.h
  include/audio_reader.h
  include/inference_engine.h
)

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------
add_executable(mir_engine ${MIR_SOURCES} ${MIR_HEADERS})

target_include_directories(mir_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mir_engine PRIVATE
  portaudio                   # vcpkg: PortAudio
  SndFile::sndfile            # vcpkg: libsndfile
  FFTW3::fftw3f               # vcpkg: single-precision FFTW3 (matches fftwf_* in code)
  OnnxRuntime::OnnxRuntime    # FetchContent: ONNX Runtime
)

# ---------------------------------------------------------------------------
# Windows: copy the ONNX Runtime DLL next to the executable after build
# ---------------------------------------------------------------------------
if(WIN32)
  add_custom_command(TARGET mir_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_BASE_DIR}/lib/onnxruntime.dll"
      $<TARGET_FILE_DIR:mir_engine>
    COMMENT "Copying onnxruntime.dll to output directory"
  )
endif()

# ---------------------------------------------------------------------------
# Compiler warnings & optimisations
# ---------------------------------------------------------------------------
if(MSVC)
  # Do not force /O2 on Debug — it conflicts with /RTC1 from the default Debug flags.
  target_compile_options(mir_engine PRIVATE
    /W4
    $<$<CONFIG:Release>:/O2 /fp:fast>
  )
else()
  target_compile_options(mir_engine PRIVATE -Wall -Wextra -O3 -ffast-math)
endif()

# ---------------------------------------------------------------------------
# Install rules (optional convenience)
# ---------------------------------------------------------------------------
install(TARGETS mir_engine RUNTIME DESTINATION bin)
