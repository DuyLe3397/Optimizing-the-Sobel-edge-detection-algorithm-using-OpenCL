// Updated edge_detect.cpp with GPU (OpenCL) + CPU(OpenCV) + CPU(OpenMP)
// AUTO ITERATE MULTI IMAGES + SAVE CSV RESULTS

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <filesystem>

#define KERNEL_FILE "kernels/edge_filter.cl"

namespace fs = std::filesystem;

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

// =============================================
// GPU Sobel (OpenCL) as Function
// =============================================
long long sobelGPU(const cv::Mat &input, cv::Mat &output)
{
    int width = input.cols;
    int height = input.rows;
    size_t image_size = width * height;

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

    cl_mem input_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size, (void *)input.data, &err);
    cl_mem output_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_size, nullptr, &err);

    checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buf), "SetArg0");
    checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buf), "SetArg1");
    checkError(clSetKernelArg(kernel, 2, sizeof(int), &width), "SetArg2");
    checkError(clSetKernelArg(kernel, 3, sizeof(int), &height), "SetArg3");

    size_t globalSize[2] = {(size_t)width, (size_t)height};

    auto t1 = std::chrono::high_resolution_clock::now();
    checkError(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
    checkError(clFinish(queue), "clFinish");
    checkError(clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, image_size, output.data, 0, nullptr, nullptr), "clEnqueueReadBuffer");
    auto t2 = std::chrono::high_resolution_clock::now();

    long long time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return time;
}

// =============================================
// CPU Sequential (OpenCV filter2D)
// =============================================
long long sobelCPU(const cv::Mat &input, cv::Mat &output)
{
    cv::setNumThreads(1);

    cv::Mat kernelX = (cv::Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat gx, gy, ax, ay;
    cv::filter2D(input, gx, CV_16S, kernelX);
    cv::filter2D(input, gy, CV_16S, kernelY);
    cv::convertScaleAbs(gx, ax);
    cv::convertScaleAbs(gy, ay);
    cv::addWeighted(ax, 0.5, ay, 0.5, 0, output);
    auto t2 = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

// =============================================
// CPU Parallel OpenMP
// =============================================
long long sobelOMP(const cv::Mat &input, cv::Mat &output)
{
    int width = input.cols;
    int height = input.rows;

    auto t1 = std::chrono::high_resolution_clock::now();

    // Parallel theo trục Y
#pragma omp parallel for
    for (int y = 1; y < height - 1; y++)
    {
        // Vector hóa theo trục X
#pragma omp simd
        for (int x = 1; x < width - 1; x++)
        {
            int gx =
                -input.at<uchar>(y - 1, x - 1) + input.at<uchar>(y - 1, x + 1) +
                -2 * input.at<uchar>(y, x - 1) + 2 * input.at<uchar>(y, x + 1) +
                -input.at<uchar>(y + 1, x - 1) + input.at<uchar>(y + 1, x + 1);

            int gy =
                input.at<uchar>(y - 1, x - 1) + 2 * input.at<uchar>(y - 1, x) + input.at<uchar>(y - 1, x + 1) +
                -input.at<uchar>(y + 1, x - 1) - 2 * input.at<uchar>(y + 1, x) - input.at<uchar>(y + 1, x + 1);

            int mag = std::min(255, (abs(gx) + abs(gy)) / 2);

            output.at<uchar>(y, x) = mag;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

int main()
{
    fs::create_directories("output");

    // ===========================
    //  CSV header (5 runs x 3 methods)
    // ===========================
    std::ofstream csv("output/results.csv");
    csv << "RESOLUTION,"
        << "GPU_1,GPU_2,GPU_3,GPU_4,GPU_5,"
        << "CPU_1,CPU_2,CPU_3,CPU_4,CPU_5,"
        << "OMP_1,OMP_2,OMP_3,OMP_4,OMP_5\n";

    // ===========================
    //    PROCESS EACH IMAGE
    // ===========================
    for (auto &file : fs::directory_iterator("input"))
    {
        if (file.path().extension() != ".jpg")
            continue;

        std::string filename = file.path().filename().string();
        std::cout << "\n🔍 Processing " << filename << std::endl;

        cv::Mat input = cv::imread(file.path().string(), cv::IMREAD_GRAYSCALE);
        if (input.empty())
        {
            std::cerr << "❌ Cannot load " << filename << std::endl;
            continue;
        }

        int w = input.cols;
        int h = input.rows;
        std::string res = std::to_string(w) + "x" + std::to_string(h);

        // Output buffers
        cv::Mat gpu_out(h, w, CV_8UC1);
        cv::Mat cpu_out(h, w, CV_8UC1);
        cv::Mat omp_out(h, w, CV_8UC1);

        // ==============================
        //      RUN 5 TIMES EACH
        // ==============================
        long long GPU[5], CPU[5], OMP[5];

        for (int i = 0; i < 5; i++)
        {
            GPU[i] = sobelGPU(input, gpu_out);
            CPU[i] = sobelCPU(input, cpu_out);
            OMP[i] = sobelOMP(input, omp_out);
        }

        // ==============================
        //     SAVE IMAGES (1 sample only)
        // ==============================
        cv::imwrite("output/GPU_" + filename, gpu_out);
        cv::imwrite("output/CPU_" + filename, cpu_out);
        cv::imwrite("output/OMP_" + filename, omp_out);

        // ==============================
        //     WRITE TO CSV
        // ==============================
        csv << res << ",";
        for (int i = 0; i < 5; i++)
            csv << GPU[i] << ",";
        for (int i = 0; i < 5; i++)
            csv << CPU[i] << ",";
        for (int i = 0; i < 5; i++)
            csv << OMP[i] << (i == 4 ? "\n" : ",");
    }

    csv.close();
    std::cout << "\n📄 Saved results: output/results.csv\n";

    // ======================================================
    //            LOAD CSV FOR PLOTTING
    // ======================================================
    std::ifstream fin("output/results.csv");
    std::string line;
    getline(fin, line); // skip header

    std::vector<std::string> labels;
    std::vector<std::vector<double>> gpu(5), cpu(5), omp(5);

    while (getline(fin, line))
    {
        std::stringstream ss(line);
        std::string res;
        getline(ss, res, ',');
        labels.push_back(res);

        for (int i = 0; i < 5; i++)
        {
            std::string v;
            getline(ss, v, ',');
            gpu[i].push_back(stod(v) / 1000.0);
        }
        for (int i = 0; i < 5; i++)
        {
            std::string v;
            getline(ss, v, ',');
            cpu[i].push_back(stod(v) / 1000.0);
        }
        for (int i = 0; i < 5; i++)
        {
            std::string v;
            getline(ss, v, ',');
            omp[i].push_back(stod(v) / 1000.0);
        }
    }
    fin.close();

    // ======================================================
    //         Chart drawing function (OpenCV)
    // ======================================================
    auto draw_chart = [&](std::string filename,
                          std::vector<std::vector<double>> &data,
                          std::string title)
    {
        int width = 1400, height = 700;
        cv::Mat chart(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        int margin = 100;
        int base_y = height - margin;

        // Find max
        double maxv = 0;
        for (auto &run : data)
            for (double v : run)
                if (v > maxv)
                    maxv = v;

        double scale = (height - 2 * margin) / maxv;

        // Draw axes
        cv::line(chart, {margin, base_y}, {width - margin, base_y}, {0, 0, 0}, 2);
        cv::line(chart, {margin, base_y}, {margin, margin}, {0, 0, 0}, 2);

        // ---- Draw Y ticks every 50ms ----
        int step_ms = 50;
        for (int t = 0; t <= (int)maxv + 50; t += step_ms)
        {
            int y = base_y - t * scale;
            cv::line(chart, {margin - 5, y}, {margin + 5, y}, {0, 0, 0}, 2);

            cv::putText(chart, std::to_string(t),
                        {margin - 60, y + 5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        {0, 0, 0}, 1);
        }

        // Draw title
        cv::putText(chart, title, {width / 2 - 150, 50},
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 0, 0}, 3);

        int n = labels.size();
        int step_x = (width - 2 * margin) / (n + 1);

        cv::Scalar colors[5] = {
            {0, 0, 255}, {0, 128, 255}, {0, 200, 0}, {255, 128, 0}, {128, 0, 255}};

        for (int run = 0; run < 5; run++)
        {
            for (int i = 0; i < n; i++)
            {
                int x = margin + (i + 1) * step_x;
                int y = base_y - data[run][i] * scale;

                cv::circle(chart, {x, y}, 5, colors[run], -1);

                if (i > 0)
                {
                    int x_prev = margin + (i)*step_x;
                    int y_prev = base_y - data[run][i - 1] * scale;

                    cv::line(chart, {x_prev, y_prev}, {x, y}, colors[run], 2);
                }
            }
        }

        cv::imwrite("output/" + filename, chart);
        std::cout << "📊 Saved chart: output/" << filename << "\n";
    };

    // ======================================================
    //         GENERATE 3 SEPARATE CHARTS
    // ======================================================
    draw_chart("chart_gpu.png", gpu, "GPU (OpenCL) - 5 Runs");
    draw_chart("chart_cpu.png", cpu, "CPU (OpenCV) - 5 Runs");
    draw_chart("chart_omp.png", omp, "CPU (OpenMP) - 5 Runs");

    return 0;
}
