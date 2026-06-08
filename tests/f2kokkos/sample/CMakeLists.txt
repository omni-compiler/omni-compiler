cmake_minimum_required(VERSION 3.16)
project(kokkos-fortran LANGUAGES Fortran CXX)

set(FLCL_DIR   $ENV{WK}/flcl)

find_package(Kokkos 4.6 REQUIRED CONFIG)
find_package(flcl REQUIRED PATHS ${FLCL_DIR})

#----------------------------
# Configuration for F2Kokkos
#----------------------------
#set(PP_EXECUTABLE PP)

set(SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR})
set(BIN_DIR   ${CMAKE_CURRENT_BINARY_DIR})
set(GEN_DIR   ${BIN_DIR}/generated)

file(MAKE_DIRECTORY ${GEN_DIR})

set(INPUT_F90 ${SRC_DIR}/sub.f90)
# set(XMOD_FILES ${SRC_DIR}/flcl_dualview_mod.xmod
#   ${SRC_DIR}/flcl_mod.xmod
#   ${SRC_DIR}/flcl_ndarray_mod.xmod
#   ${SRC_DIR}/flcl_types_f_mod.xmod
#   ${SRC_DIR}/flcl_util_kokkos_mod.xmod
#   ${SRC_DIR}/flcl_util_strings_mod.xmod
#   ${SRC_DIR}/flcl_view_mod.xmod)

set(OUT_KKS_F90 ${GEN_DIR}/sub_kks.f90)
set(OUT_KKS_CC  ${GEN_DIR}/sub_kks.cc)

#----------------------------
# Rules for F2Kokkos
#----------------------------
add_custom_command(
    OUTPUT
        ${OUT_KKS_F90}
        ${OUT_KKS_CC}
    COMMAND
        F_Front -fxmp
        ${INPUT_F90} | F_Trans -M ${SRC_DIR} | F_Back -l > ${OUT_KKS_F90}
    COMMAND
        mv ${SRC_DIR}/sub_kks.cc ${GEN_DIR}
    DEPENDS
        ${INPUT_F90}
#	${XMOD_FILES}
    COMMENT
        "Running PP on sub.f90"
    VERBATIM
)

#set_source_files_properties(
#    ${OUT_KKS_F90}
#    ${OUT_KKS_CC}
#    PROPERTIES GENERATED TRUE
#)

#----------------------------
# Defining the executable
#----------------------------
add_executable(omni-test
  omni-test.f90
#  sub_kokkos_mod.f90
  ${OUT_KKS_F90}
  ${OUT_KKS_CC})

set_target_properties(
    omni-test
    PROPERTIES
        Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mod"
)

target_include_directories(omni-test
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/mod>
        $<INSTALL_INTERFACE:mod>
)

target_link_libraries(omni-test flcl::flcl)

if (CMAKE_Fortran_COMPILER_ID STREQUAL "XL")
    target_link_options(omni-test PRIVATE LINKER:-lxlf90_r)
endif()

if (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel" OR CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set_target_properties(omni-test PROPERTIES LINKER_LANGUAGE Fortran)
endif()

set(RPATHS "$ENV{LD_LIBRARY_PATH}")
set_target_properties(omni-test PROPERTIES 
                      BUILD_WITH_INSTALL_RPATH True
                      INSTALL_RPATH "${RPATHS}")
