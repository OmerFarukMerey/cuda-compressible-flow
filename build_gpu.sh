#!/bin/bash

# Add MSVC and OpenCV DLLs to PATH
export PATH="/c/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64:$PATH"
export PATH="/c/vcpkg/installed/x64-windows/bin:$PATH"

# Compile GPU version
echo "Compiling GPU version..."
nvcc main.cu -o compressible.exe \
  -I"C:/vcpkg/installed/x64-windows/include/opencv4" \
  -L"C:/vcpkg/installed/x64-windows/lib" \
  -lopencv_core4 -lopencv_highgui4 -lopencv_imgcodecs4 -lopencv_imgproc4 \
  -arch=compute_120 -allow-unsupported-compiler

if [ $? -ne 0 ]; then
  echo "GPU compilation failed!"
  exit 1
fi

echo "GPU compilation successful!"
mkdir -p resultatsim
echo "Running GPU simulation..."
./compressible.exe "$@"
