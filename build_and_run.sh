#!/bin/bash

# Add MSVC and OpenCV DLLs to PATH
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:$PATH"
export PATH="/c/vcpkg/installed/x64-windows/bin:$PATH"

# Compile
echo "Compiling..."
nvcc main.cu -o compressible.exe \
  -I"C:/vcpkg/installed/x64-windows/include/opencv4" \
  -L"C:/vcpkg/installed/x64-windows/lib" \
  -lopencv_core4 -lopencv_highgui4 -lopencv_imgcodecs4 -lopencv_imgproc4 \
  -arch=compute_120 -allow-unsupported-compiler

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "GPU compilation successful!"

# Compile CPU version
echo "Compiling CPU version..."
cl.exe /EHsc /O2 /Fe:compressible_cpu.exe main_cpu.cpp \
  /I"C:/vcpkg/installed/x64-windows/include/opencv4" \
  /link /LIBPATH:"C:/vcpkg/installed/x64-windows/lib" \
  opencv_core4.lib opencv_highgui4.lib opencv_imgcodecs4.lib opencv_imgproc4.lib

if [ $? -ne 0 ]; then
  echo "CPU compilation failed!"
  exit 1
fi

echo "CPU compilation successful!"

# Create output directories and run
mkdir -p resultatsim resultatsim_cpu
echo "Running GPU simulation..."
./compressible.exe "$@"
