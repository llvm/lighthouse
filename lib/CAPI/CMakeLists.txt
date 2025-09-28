# Based on IREE's CAPI CMakeLists.txt :
# https://github.com/iree-org/iree/blob/39c572deb0d388eb86d78295896a6e38835ccc8c/compiler/src/iree/compiler/API/CMakeLists.txt
set(C_LIBS
  MLIRCAPIIR
  MLIRCAPIDebug
  MLIRCAPIFunc
  MLIRCAPIInterfaces
  MLIRCAPITransforms
)

foreach (lib ${C_LIBS})
  list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  # GENEX_EVAL is not recursive: it evaluates one level of generator
  # expressions. There is a way to hold it just right so that GENEX_EVAL is
  # used at the call site, but it is really tricky and people can't be expected
  # to get it right. In fact, at the time of this writing, some misc usage
  # in LLVM causes a second level of generator expressions in some base
  # libraries. For good measure, we do GENEX_EVAL four times. If you get errors
  # that show like generator expressions are showing up in link lines, this is
  # the culprit. Look at the export_objects_debug.txt to confirm. Then, add
  # another level of fix upstream if you like pain.
  list(APPEND _DEPS "$<GENEX_EVAL:$<GENEX_EVAL:$<GENEX_EVAL:$<GENEX_EVAL:$<TARGET_PROPERTY:${lib},LINK_LIBRARIES>>>>>")
endforeach ()

set(CMAKE_CXX_VISIBILITY_PRESET "hidden")
set(CMAKE_VISIBILITY_INLINES_HIDDEN "ON")

add_mlir_library(Lighthouse-C
  CAPI.cpp
  SHARED
  EXCLUDE_FROM_LIBMLIR
)

target_sources(Lighthouse-C PRIVATE
  ${_OBJECTS}
)

target_link_libraries(Lighthouse-C PRIVATE
  ${_DEPS}
)

target_link_options(Lighthouse-C PRIVATE "-Wl,-exclude-libs,ALL")
