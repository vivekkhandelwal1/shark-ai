cmake_minimum_required(VERSION 3.13)
project(HIP_KERNELS LANGUAGES C CXX)

# Set ROCm and IREE build paths
set(IREE_BUILD_DIR $ENV{IREE_BUILD_DIR})
set(ROCM_PATH /opt/rocm)
set(TARGET_ARCH gfx942)

# Find all kernel source files in the kernels subdirectory
file(GLOB KERNEL_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.c")

# Paths to ROCm and IREE bitcode libraries
set(ROCM_BC
  ${ROCM_PATH}/amdgcn/bitcode/opencl.bc
  ${ROCM_PATH}/amdgcn/bitcode/hip.bc
  ${ROCM_PATH}/amdgcn/bitcode/oclc_isa_version_1100.bc
)
set(IREE_BC
  ${IREE_BUILD_DIR}/lib/iree_platform_libs/rocm/ockl.bc
  ${IREE_BUILD_DIR}/lib/iree_platform_libs/rocm/ocml.bc
)

set(HSACO_TARGETS "")

# Define the output directory for .hsaco files
set(CUSTOM_KERNEL_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled_kernels")
file(MAKE_DIRECTORY ${CUSTOM_KERNEL_OUTPUT_DIR})

foreach(KERNEL_SRC ${KERNEL_SRCS})
  get_filename_component(KERNEL_NAME ${KERNEL_SRC} NAME)
  set(BC_FILE "${KERNEL_NAME}.bc")
  set(LINKED_BC_FILE "${KERNEL_NAME}.linked.bc")
  set(O_FILE "${KERNEL_NAME}.o")
  set(HSACO_FILE "${KERNEL_NAME}.hsaco")
  set(HSACO_OUTPUT_FILE "${CUSTOM_KERNEL_OUTPUT_DIR}/${KERNEL_NAME}.hsaco")

  # Step 1: Compile HIP kernel to LLVM IR
  add_custom_command(
    OUTPUT ${BC_FILE}
    COMMAND ${IREE_BUILD_DIR}/llvm-project/bin/clang
      -x hip --offload-arch=${TARGET_ARCH} --offload-device-only -nogpulib
      -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -O3 -fvisibility=protected
      -emit-llvm -c ${KERNEL_SRC} -o ${BC_FILE}
    DEPENDS ${KERNEL_SRC}
    COMMENT "Compiling ${KERNEL_NAME} to LLVM IR"
  )

  # Step 2: Link with ROCm/IREE bitcode
  add_custom_command(
    OUTPUT ${LINKED_BC_FILE}
    COMMAND ${IREE_BUILD_DIR}/llvm-project/bin/llvm-link
      ${IREE_BC} ${ROCM_BC} ${BC_FILE} -o ${LINKED_BC_FILE}
    DEPENDS ${BC_FILE}
    COMMENT "Linking ${KERNEL_NAME} with ROCm/IREE bitcode"
  )

  # Step 3: Compile to AMDGPU object file
  add_custom_command(
    OUTPUT ${O_FILE}
    COMMAND ${IREE_BUILD_DIR}/llvm-project/bin/clang
      -target amdgcn-amd-amdhsa -mcpu=${TARGET_ARCH}
      -c ${LINKED_BC_FILE} -o ${O_FILE}
    DEPENDS ${LINKED_BC_FILE}
    COMMENT "Compiling ${KERNEL_NAME} linked bitcode to AMDGPU object"
  )

  # Step 4: Link to produce .hsaco
  add_custom_command(
    OUTPUT ${HSACO_FILE}
    COMMAND ${IREE_BUILD_DIR}/llvm-project/bin/lld
      -flavor gnu -shared ${O_FILE} -o ${HSACO_FILE}
    DEPENDS ${O_FILE}
    COMMENT "Linking ${KERNEL_NAME} object to produce .hsaco"
  )

  # Step 5: Copy .hsaco to output directory
  add_custom_command(
    OUTPUT ${HSACO_OUTPUT_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${HSACO_FILE} ${HSACO_OUTPUT_FILE}
    DEPENDS ${HSACO_FILE}
    COMMENT "Copying ${HSACO_FILE} to ${HSACO_OUTPUT_FILE}"
  )

  # Collect all .hsaco output targets
  list(APPEND HSACO_TARGETS ${HSACO_OUTPUT_FILE})

  # Optional: Clean up intermediates
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${BC_FILE}
    ${LINKED_BC_FILE}
    ${O_FILE}
    ${HSACO_FILE}
  )
endforeach()

add_custom_command(
  OUTPUT topk_test
  COMMAND hipcc ${CMAKE_CURRENT_SOURCE_DIR}/tests/topk.cpp -o topk_test
  DEPENDS topk_test
  COMMENT "Compiling topk test"
)

# Add a custom target to build all .hsaco files in the output directory
add_custom_target(
  tests ALL
  DEPENDS topk_test
  COMMENT "Building test deps"
)

# Add a custom target to build all .hsaco files in the output directory
add_custom_target(
  all_hsaco_kernels ALL
  DEPENDS ${HSACO_TARGETS}
  COMMENT "Building and copying all .hsaco kernels to ${CUSTOM_KERNEL_OUTPUT_DIR}"
)
