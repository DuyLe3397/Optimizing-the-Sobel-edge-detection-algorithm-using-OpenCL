#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

#define KERNEL_FILE "kernels/edge_filter.cl"

void checkError(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        std::cerr << "❌ Error: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::string loadKernel(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "❌ Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    return std::string(std::istreambuf_iterator<char>(file), {});
}

int main()
{
    cv::Mat input = cv::imread("input/input.jpg", cv::IMREAD_GRAYSCALE);
    if (input.empty())
    {
        std::cerr << "❌ Cannot load input image!" << std::endl;
        return -1;
    }

    int width = input.cols;
    int height = input.rows;
    size_t image_size = width * height;

    // === CPU xử lý bằng OpenCV filter2D ===
    cv::Mat cpu_output;
    cv::Mat kernelX = (cv::Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat grad_x, grad_y;
    cv::filter2D(input, grad_x, CV_16S, kernelX);
    cv::filter2D(input, grad_y, CV_16S, kernelY);
    cv::Mat abs_x, abs_y;
    cv::convertScaleAbs(grad_x, abs_x);
    cv::convertScaleAbs(grad_y, abs_y);
    cv::addWeighted(abs_x, 0.5, abs_y, 0.5, 0, cpu_output);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    cv::imwrite("output/cpu_output.jpg", cpu_output);

    // === GPU xử lý bằng OpenCL ===
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    checkError(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    std::string kernel_code = loadKernel(KERNEL_FILE);
    const char *source = kernel_code.c_str();
    size_t source_size = kernel_code.length();

    cl_program program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
    checkError(err, "clCreateProgramWithSource");
    checkError(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr), "clBuildProgram");

    cl_kernel kernel = clCreateKernel(program, "edge_filter", &err);
    checkError(err, "clCreateKernel");

    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size, input.data, &err);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_size, nullptr, &err);

    checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf), "SetArg0");
    checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf), "SetArg1");
    checkError(clSetKernelArg(kernel, 2, sizeof(int), &width), "SetArg2");
    checkError(clSetKernelArg(kernel, 3, sizeof(int), &height), "SetArg3");

    size_t globalSize[2] = {(size_t)width, (size_t)height};

    cv::Mat opencl_output(height, width, CV_8UC1);
    auto t3 = std::chrono::high_resolution_clock::now();
    checkError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkError(clFinish(queue), "clFinish");
    checkError(clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, image_size, opencl_output.data, 0, nullptr, nullptr), "clEnqueueReadBuffer");
    auto t4 = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    cv::imwrite("output/opencl_output.jpg", opencl_output);

    // === Đo độ khác biệt giữa 2 ảnh ===
    cv::Mat diff;
    cv::absdiff(cpu_output, opencl_output, diff);
    double diff_percent = (cv::countNonZero(diff) * 100.0) / (width * height);

    std::cout << "✅ CPU Time (OpenCV): " << cpu_time << " µs\n";
    std::cout << "⚡ GPU Time (OpenCL): " << gpu_time << " µs\n";
    std::cout << "📊 Pixel Difference: " << diff_percent << " %\n";

    // Cleanup
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
