# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -fopenmp   # <-- Bật OpenMP

# Directories
SRC_DIR = src
BUILD_DIR = build

# OpenCL and OpenCV Paths
OPENCL_INC = /c/OpenCL-SDK/include
OPENCL_LIB = /c/OpenCL-SDK/lib

OPENCV_INC = /ucrt64/include/opencv4
OPENCV_LIB = /ucrt64/lib

# Linker Flags
LDFLAGS = -L$(OPENCL_LIB) -lOpenCL \
          -L$(OPENCV_LIB) \
          -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc \
          -fopenmp                    # <-- Link OpenMP

# Files
SRC = $(SRC_DIR)/edge_detect.cpp
EXE = $(BUILD_DIR)/edge_detect.exe
INPUT_DIR = input
OUTPUT_DIR = output

# Default target
all: $(EXE)

$(EXE): $(SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(OPENCL_INC) -I$(OPENCV_INC) $(SRC) -o $@ $(LDFLAGS)
	@echo "Build successful!"

run: all
	@echo "Running program..."
	@./$(EXE)
	@echo "Run successfully!"

clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory!"

.PHONY: all run clean
