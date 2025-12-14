// edge_detect.cpp
// GPU (OpenCL) + CPU (OpenCV) + CPU (OpenMP)
// AUTO ITERATE MULTI IMAGES + SAVE CSV RESULTS + SPEEDUP & SCALING

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <filesystem>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

#define KERNEL_FILE "kernels/edge_filter.cl"

namespace fs = std::filesystem;

// =========================
//  OpenCL Helper
// =========================
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

// =========================
//  OpenCL Sobel Context
// =========================
struct OpenCLSobelContext
{
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_device_id device = nullptr;
    int width = 0;
    int height = 0;
    cl_mem input_buf = nullptr;
    cl_mem output_buf = nullptr;
};

OpenCLSobelContext initSobelGPU(int width, int height)
{
    OpenCLSobelContext ctx;
    ctx.width = width;
    ctx.height = height;
    size_t image_size = static_cast<size_t>(width) * height;

    cl_int err;
    cl_platform_id platform;

    checkError(clGetPlatformIDs(1, &platform, nullptr), "clGetPlatformIDs");
    checkError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &ctx.device, nullptr), "clGetDeviceIDs");

    ctx.context = clCreateContext(nullptr, 1, &ctx.device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    ctx.queue = clCreateCommandQueue(ctx.context, ctx.device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    std::string kernel_code = loadKernel(KERNEL_FILE);
    const char *source = kernel_code.c_str();
    size_t source_size = kernel_code.length();

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &source, &source_size, &err);
    checkError(err, "clCreateProgramWithSource");
    checkError(clBuildProgram(ctx.program, 1, &ctx.device, nullptr, nullptr, nullptr), "clBuildProgram");

    ctx.kernel = clCreateKernel(ctx.program, "edge_filter", &err);
    checkError(err, "clCreateKernel");

    ctx.input_buf = clCreateBuffer(ctx.context,
                                   CL_MEM_READ_ONLY,
                                   image_size,
                                   nullptr,
                                   &err);
    checkError(err, "clCreateBuffer input");

    ctx.output_buf = clCreateBuffer(ctx.context,
                                    CL_MEM_WRITE_ONLY,
                                    image_size,
                                    nullptr,
                                    &err);
    checkError(err, "clCreateBuffer output");

    // width, height: set 1 lần
    checkError(clSetKernelArg(ctx.kernel, 2, sizeof(int), &ctx.width), "SetArg2");
    checkError(clSetKernelArg(ctx.kernel, 3, sizeof(int), &ctx.height), "SetArg3");

    return ctx;
}

void releaseSobelGPU(OpenCLSobelContext &ctx)
{
    if (ctx.input_buf)
        clReleaseMemObject(ctx.input_buf);
    if (ctx.output_buf)
        clReleaseMemObject(ctx.output_buf);
    if (ctx.kernel)
        clReleaseKernel(ctx.kernel);
    if (ctx.program)
        clReleaseProgram(ctx.program);
    if (ctx.queue)
        clReleaseCommandQueue(ctx.queue);
    if (ctx.context)
        clReleaseContext(ctx.context);
}

// =============================================
// GPU Sobel (OpenCL) – sử dụng context đã init
// =============================================
long long sobelGPU(const cv::Mat &input, cv::Mat &output, OpenCLSobelContext &ctx)
{
    int width = ctx.width;
    int height = ctx.height;
    size_t image_size = static_cast<size_t>(width) * height;

    // đảm bảo output có đúng kích thước
    if (output.empty() || output.rows != height || output.cols != width)
        output.create(height, width, CV_8UC1);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Ghi input
    checkError(
        clEnqueueWriteBuffer(ctx.queue, ctx.input_buf, CL_TRUE,
                             0, image_size, input.data,
                             0, nullptr, nullptr),
        "clEnqueueWriteBuffer");

    // Set buffer args
    checkError(clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.input_buf), "SetArg0");
    checkError(clSetKernelArg(ctx.kernel, 1, sizeof(cl_mem), &ctx.output_buf), "SetArg1");

    size_t globalSize[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};

    // Chạy kernel
    checkError(clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 2, nullptr,
                                      globalSize, nullptr, 0, nullptr, nullptr),
               "clEnqueueNDRangeKernel");

    checkError(clFinish(ctx.queue), "clFinish");

    // Đọc back output
    checkError(
        clEnqueueReadBuffer(ctx.queue, ctx.output_buf, CL_TRUE,
                            0, image_size, output.data,
                            0, nullptr, nullptr),
        "clEnqueueReadBuffer");

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

// =============================================
// CPU Sequential (OpenCV filter2D)
// =============================================
long long sobelCPU(const cv::Mat &input, cv::Mat &output)
{
    cv::setNumThreads(1); // tránh OpenCV tự đa luồng

    cv::Mat kernelX = (cv::Mat_<char>(3, 3)
                           << -1,
                       0, 1,
                       -2, 0, 2,
                       -1, 0, 1);
    cv::Mat kernelY = (cv::Mat_<char>(3, 3)
                           << 1,
                       2, 1,
                       0, 0, 0,
                       -1, -2, -1);

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
// CPU Parallel OpenMP – có tham số num_threads
// =============================================
long long sobelOMP(const cv::Mat &input, cv::Mat &output, int num_threads)
{
    cv::setNumThreads(1);

    int width = input.cols;
    int height = input.rows;

    if (output.empty() || output.rows != height || output.cols != width)
        output.create(height, width, CV_8UC1);

    omp_set_num_threads(num_threads);

    auto t1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (int y = 1; y < height - 1; y++)
    {
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

            output.at<uchar>(y, x) = static_cast<uchar>(mag);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

// =============================================
//               MAIN
// =============================================
int main(int argc, char **argv)
{
    fs::create_directories("output");

    // CSV chi tiết
    std::ofstream csv("output/results.csv");
    csv << "RESOLUTION,"
        << "GPU_1,GPU_2,GPU_3,GPU_4,GPU_5,"
        << "CPU_1,CPU_2,CPU_3,CPU_4,CPU_5,"
        << "OMP4_1,OMP4_2,OMP4_3,OMP4_4,OMP4_5\n";

    // CSV tổng hợp (mean + speedup + scaling)
    std::ofstream csv_summary("output/summary.csv");
    csv_summary << "RESOLUTION,"
                << "CPU_mean_ms,GPU_mean_ms,"
                << "OMP_1_mean_ms,OMP_2_mean_ms,OMP_4_mean_ms,OMP_8_mean_ms,"
                << "Speedup_GPU,"
                << "Speedup_OMP_1,Speedup_OMP_2,Speedup_OMP_4,Speedup_OMP_8\n";

    // Duyệt toàn bộ ảnh trong thư mục input/
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

        cv::Mat gpu_out(h, w, CV_8UC1);
        cv::Mat cpu_out(h, w, CV_8UC1);
        cv::Mat omp_out(h, w, CV_8UC1);

        // === Init OpenCL cho độ phân giải này ===
        OpenCLSobelContext gpu_ctx = initSobelGPU(w, h);

        long long GPU[5], CPU[5], OMP4[5];

        for (int i = 0; i < 5; i++)
        {
            GPU[i] = sobelGPU(input, gpu_out, gpu_ctx);
            CPU[i] = sobelCPU(input, cpu_out);
            OMP4[i] = sobelOMP(input, omp_out, 4); // cố định 4 luồng cho bảng chi tiết
        }

        // Lưu ảnh kết quả (GPU / CPU / OMP 4 threads)
        cv::imwrite("output/GPU_" + filename, gpu_out);
        cv::imwrite("output/CPU_" + filename, cpu_out);
        cv::imwrite("output/OMP4_" + filename, omp_out);

        // Ghi CSV chi tiết
        csv << res << ",";
        for (int i = 0; i < 5; i++)
            csv << GPU[i] << ",";
        for (int i = 0; i < 5; i++)
            csv << CPU[i] << ",";
        for (int i = 0; i < 5; i++)
            csv << OMP4[i] << (i == 4 ? "\n" : ",");

        // ======= Tính trung bình cho CPU & GPU =======
        auto avg5 = [](long long a[5])
        {
            long long s = 0;
            for (int i = 0; i < 5; ++i)
                s += a[i];
            return static_cast<double>(s) / 5.0; // microseconds
        };

        double cpu_mean_us = avg5(CPU);
        double gpu_mean_us = avg5(GPU);

        // ======= Đo OMP với 1,2,4,8 threads (3 lần mỗi cấu hình) =======
        int omp_threads[4] = {1, 2, 4, 8};
        double omp_mean_us[4];

        for (int k = 0; k < 4; ++k)
        {
            long long tmp[3];
            for (int i = 0; i < 3; ++i)
                tmp[i] = sobelOMP(input, omp_out, omp_threads[k]);
            long long s = tmp[0] + tmp[1] + tmp[2];
            omp_mean_us[k] = static_cast<double>(s) / 3.0;
        }

        // ======= Speedup =======
        double speedup_gpu = cpu_mean_us / gpu_mean_us;
        double speedup_omp[4];
        for (int k = 0; k < 4; ++k)
            speedup_omp[k] = cpu_mean_us / omp_mean_us[k];

        // Ghi CSV summary (đổi ra ms)
        csv_summary << res << ","
                    << cpu_mean_us / 1000.0 << "," << gpu_mean_us / 1000.0 << ","
                    << omp_mean_us[0] / 1000.0 << "," // 1 thread
                    << omp_mean_us[1] / 1000.
                    << ","                            // 2 threads
                    << omp_mean_us[2] / 1000.0 << "," // 4 threads
                    << omp_mean_us[3] / 1000.0 << "," // 8 threads
                    << speedup_gpu << ","
                    << speedup_omp[0] << ","
                    << speedup_omp[1] << ","
                    << speedup_omp[2] << ","
                    << speedup_omp[3] << "\n";

        // Giải phóng OpenCL resource cho ảnh này
        releaseSobelGPU(gpu_ctx);
    }

    csv.close();
    csv_summary.close();
    std::cout << "\n📄 Saved results: output/results.csv & output/summary.csv\n";

    // ======================================================
    //            LOAD results.csv FOR PLOTTING
    // ======================================================
    std::ifstream fin("output/results.csv");
    if (!fin.is_open())
    {
        std::cerr << "⚠️ Cannot open output/results.csv for chart drawing\n";
        return 0;
    }

    std::string line;
    getline(fin, line); // skip header

    std::vector<std::string> labels;
    std::vector<std::vector<double>> gpu(5), cpu(5), omp4(5);

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
            gpu[i].push_back(std::stod(v) / 1000.0); // us -> ms
        }
        for (int i = 0; i < 5; i++)
        {
            std::string v;
            getline(ss, v, ',');
            cpu[i].push_back(std::stod(v) / 1000.0);
        }
        for (int i = 0; i < 5; i++)
        {
            std::string v;
            getline(ss, v, ',');
            omp4[i].push_back(std::stod(v) / 1000.0);
        }
    }
    fin.close();

    // ======================================================
    //         Chart drawing function (OpenCV)
    // ======================================================
    auto draw_chart = [&](const std::string &filename,
                          std::vector<std::vector<double>> &data,
                          const std::string &title)
    {
        int width = 1600, height = 800;
        cv::Mat chart(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

        int margin_left = 150;
        int margin_bottom = 120;
        int margin_top = 80;
        int margin_right = 80;

        int base_y = height - margin_bottom;

        // Find max value
        double maxv = 0;
        for (auto &run : data)
            for (double v : run)
                maxv = std::max(maxv, v);
        if (maxv <= 0)
            maxv = 1.0;

        // Giới hạn số tick trên trục Y (ví dụ tối đa 8–10 tick)
        const int max_ticks = 8;
        double rough_step = maxv / max_ticks; // bước thô (ms)

        // Làm tròn bước thô thành 1, 2, 5 * 10^k để trông đẹp
        double pow10 = std::pow(10.0, std::floor(std::log10(rough_step)));
        double norm = rough_step / pow10; // trong [1,10)
        double factor;
        if (norm <= 1.0)
            factor = 1.0;
        else if (norm <= 2.0)
            factor = 2.0;
        else if (norm <= 5.0)
            factor = 5.0;
        else
            factor = 10.0;

        double step_ms = factor * pow10; // bước chuẩn cho trục Y (ms)

        // scale Y
        double y_scale = (height - margin_top - margin_bottom) / (maxv);

        // Y axis
        cv::line(chart,
                 {margin_left, margin_top},
                 {margin_left, base_y},
                 {0, 0, 0}, 2);

        // Vẽ tick trục Y với bước đã tối ưu
        for (double t = 0; t <= maxv + 0.5 * step_ms; t += step_ms)
        {
            int y = base_y - static_cast<int>(t * y_scale);
            if (y < margin_top)
                break;

            cv::line(chart, {margin_left - 5, y}, {margin_left + 5, y}, {0, 0, 0}, 1);

            std::string label = std::to_string(static_cast<int>(std::round(t)));
            cv::putText(chart, label + " ms",
                        {margin_left - 120, y + 5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        {0, 0, 0}, 1);
        }

        // X axis
        int n = static_cast<int>(labels.size());
        if (n == 0)
        {
            std::cerr << "⚠️ No data to draw chart: " << filename << "\n";
            return;
        }

        int step_x = (width - margin_left - margin_right) / (n + 1);

        cv::line(chart,
                 {margin_left, base_y},
                 {width - margin_right, base_y},
                 {0, 0, 0}, 2);

        for (int i = 0; i < n; i++)
        {
            int x = margin_left + (i + 1) * step_x;

            cv::line(chart, {x, base_y - 5}, {x, base_y + 5}, {0, 0, 0}, 1);

            cv::putText(chart, labels[i],
                        {x - 80, base_y + 30},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        {0, 0, 0}, 1);
        }

        // Title
        cv::putText(chart, title,
                    {width / 2 - 250, margin_top - 20},
                    cv::FONT_HERSHEY_SIMPLEX, 1.5,
                    {0, 0, 0}, 3);

        // Curves
        cv::Scalar colors[5] = {
            {0, 0, 255},
            {0, 128, 255},
            {0, 200, 0},
            {255, 128, 0},
            {128, 0, 255}};

        for (int run = 0; run < 5; run++)
        {
            for (int i = 0; i < n; i++)
            {
                int x = margin_left + (i + 1) * step_x;
                int y = base_y - static_cast<int>(data[run][i] * y_scale);

                cv::circle(chart, {x, y}, 4, colors[run], -1);

                if (i > 0)
                {
                    int px = margin_left + i * step_x;
                    int py = base_y - static_cast<int>(data[run][i - 1] * y_scale);
                    cv::line(chart, {px, py}, {x, y}, colors[run], 2);
                }
            }
        }

        cv::imwrite("output/" + filename, chart);
        std::cout << "📊 Saved chart: output/" << filename << "\n";
    };

    // Generate 3 charts: GPU / CPU / OMP4
    draw_chart("chart_gpu.png", gpu, "GPU (OpenCL) - 5 Runs");
    draw_chart("chart_cpu.png", cpu, "CPU (OpenCV) - 5 Runs");
    draw_chart("chart_omp4.png", omp4, "CPU (OpenMP, 4 threads) - 5 Runs");

    return 0;
}
