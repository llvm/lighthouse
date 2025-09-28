# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

# LLVM requires CMP0116 for tblgen: https://reviews.llvm.org/D101083
# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
set(CMAKE_POLICY_DEFAULT_CMP0116 OLD)
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(LIGHTHOUSE C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

#-------------------------------------------------------------------------------
# Input Dialects
#-------------------------------------------------------------------------------

option(LIGHTHOUSE_INPUT_TORCH "Builds support for compiling Torch MLIR programs" ON)

#-------------------------------------------------------------------------------
# Python
# If building features that require Python development, find them early in
# one invocation (some CMake versions are sensitive to resolving out of order).
#-------------------------------------------------------------------------------
option(LIGHTHOUSE_BUILD_PYTHON_BINDINGS "Builds the Lighthouse python bindings" ON)

if (LIGHTHOUSE_BUILD_PYTHON_BINDINGS)
  # After CMake 3.18, we are able to limit the scope of the search to just
  # Development.Module. Searching for Development will fail in situations where
  # the Python libraries are not available. When possible, limit to just
  # Development.Module.
  # See https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode
  #
  # Configuring the Development.Module is flaky in multi-project setups.
  # "Bootstrapping" by first looking for the optional Development component
  # seems to be robust generally.
  # See: https://reviews.llvm.org/D118148
  # If building Python packages, we have a hard requirement on 3.9+.
  find_package(Python3 3.9 COMPONENTS Interpreter Development)
  find_package(Python3 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  # Some parts of the build use FindPython instead of FindPython3. Why? No
  # one knows, but they are different. So make sure to bootstrap this one too.
  # Not doing this here risks them diverging, which on multi-Python systems,
  # can be troublesome. Note that nanobind requires FindPython.
  set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

if(NOT "${Python_EXECUTABLE}" STREQUAL "${Python3_EXECUTABLE}")
  message(WARNING "FindPython and FindPython3 found different executables. You may need to pin -DPython_EXECUTABLE and -DPython3_EXECUTABLE (${Python_EXECUTABLE} vs ${Python3_EXECUTABLE})")
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Dependency
#-------------------------------------------------------------------------------

# Both the Lighthouse and MLIR Python bindings require nanobind. We initialize
# it here at the top level so that everything uses ours consistently.
if(LIGHTHOUSE_BUILD_PYTHON_BINDINGS)
  # TODO: Configure a flag to use system deps directly.
  include(FetchContent)
  FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG        0f9ce749b257fdfe701edb3cf6f7027ba029434a # v2.4.0
  )
  FetchContent_MakeAvailable(nanobind)
endif()

# Get the main LLVM deps.
# TODO: Add a flag to use installed llvm instead.
# TODO: There are a lot of default flags that are missing here. The build
# system was only configured enough to build mlir python bindings.

# Enable MLIR Python bindings if requested.
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "")
if (LIGHTHOUSE_BUILD_PYTHON_BINDINGS)
  set(MLIR_ENABLE_BINDINGS_PYTHON ON)
endif()
# Disable MLIR attempting to configure Python dev packages. We take care of
# that in Lighthouse as a super-project.
set(MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES ON CACHE BOOL "" FORCE)

# LLVM defaults to building all targets. We always enable targets that we need
# as we need them, so default to none. The user can override this as needed,
# which is fine.
set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "")

# These defaults are moderately important to us, but the user *can*
# override them (enabling some of these brings in deps that will conflict,
# so ymmv).
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "")
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "")
set(LLVM_ENABLE_IDE ON CACHE BOOL "")
set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "")

# Force LLVM to avoid dependencies, which we don't ever really want in our
# limited builds.
set(LLVM_ENABLE_LIBEDIT OFF CACHE BOOL "Default disable")
set(LLVM_ENABLE_LIBXML2 OFF CACHE BOOL "Default disable")
set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "Default disable")
set(LLVM_ENABLE_ZLIB OFF CACHE BOOL "Default disable")
set(LLVM_ENABLE_ZSTD OFF CACHE BOOL "Default disable")
set(LLVM_FORCE_ENABLE_STATS ON CACHE BOOL "Default enable")

# Unconditionally enable mlir.
set(LLVM_ENABLE_PROJECTS "" CACHE STRING "")
list(APPEND LLVM_ENABLE_PROJECTS "mlir")

# Setup LLVM lib and bin directories.
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/llvm-project/lib")
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/llvm-project/bin")

set(_BUNDLED_LLVM_CMAKE_SOURCE_SUBDIR "third_party/llvm-project/llvm")
add_subdirectory("${_BUNDLED_LLVM_CMAKE_SOURCE_SUBDIR}" "llvm-project" EXCLUDE_FROM_ALL)

# Set some CMake variables that mirror things exported in the find_package
# world. Source of truth for these is in an installed LLVMConfig.cmake,
# MLIRConfig.cmake, LLDConfig.cmake (etc) and in the various standalone
# build segments of each project's top-level CMakeLists.
set(LLVM_CMAKE_DIR "${CMAKE_BINARY_DIR}/llvm-project/lib/cmake/llvm")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
# TODO: Fix MLIR upstream so it doesn't spew into the containing project
# binary dir. See mlir/cmake/modules/CMakeLists.txt
# (and other LLVM sub-projects).
set(MLIR_CMAKE_DIR "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")

set(LLVM_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/${_BUNDLED_LLVM_CMAKE_SOURCE_SUBDIR}/include
  ${CMAKE_BINARY_DIR}/llvm-project/include
)
set(MLIR_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir/include
  ${CMAKE_BINARY_DIR}/llvm-project/tools/mlir/include
)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# Torch-MLIR Dependency
#-------------------------------------------------------------------------------

if (LIGHTHOUSE_INPUT_TORCH)
  set(TORCH_MLIR_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/torch-mlir" CACHE PATH "")
endif()

#-------------------------------------------------------------------------------
# Top-level libraries
#-------------------------------------------------------------------------------

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(lib)
add_subdirectory(python)
