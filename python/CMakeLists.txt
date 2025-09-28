# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# This directly uses the LLVM build system in order to create bundled API
# binaries that are consistent with LLVM. Consult upstream CMake macros if you
# don't understand what this does.
################################################################################

include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under lighthouse.
# TODO: Currently, we are building this under lighthouse. , but we should be
# ideally building this under lighthouse.compiler. , when we have a proper
# seperation between the compiler and the runtime. The runtime bindings have no
# reason to ship with the mlir python bindings.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=lighthouse.")

set(_PYTHON_BUILD_PREFIX "${CMAKE_BINARY_DIR}/python")
set(_PYTHON_INSTALL_PREFIX "python_packages/lighthouse")

set(CMAKE_PLATFORM_NO_VERSIONED_SONAME 1)

declare_mlir_python_sources(LighthousePythonSources)

# The Python bindings are monolithic and we don't have a good way for the
# torch plugin to contribute Python sources, so we just gate it here
# versus having more complicated indirection. May want to rethink this
# if others need it.
if(LIGHTHOUSE_INPUT_TORCH)
  declare_mlir_python_sources(LighthousePythonSources.Torch.Importers
    ADD_TO_PARENT LighthousePythonSources
    ROOT_DIR "${TORCH_MLIR_ROOT_DIR}/python/torch_mlir"
    SOURCES
      extras/fx_importer.py
      extras/onnx_importer.py
  )
endif()

set(_SOURCE_COMPONENTS
  # Local sources.
  LighthousePythonSources

  MLIRPythonSources.Core

  # TODO: We currently include func, because that's what the torch importer
  # needs. We can include more as we need.
  # MLIR Dialects.
  MLIRPythonSources.Dialects.func
)

add_mlir_python_modules(LighthousePythonModules
  ROOT_PREFIX "${_PYTHON_BUILD_PREFIX}/lighthouse"
  INSTALL_PREFIX "${_PYTHON_INSTALL_PREFIX}/lighthouse"
  DECLARED_SOURCES ${_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
    Lighthouse-C
)
