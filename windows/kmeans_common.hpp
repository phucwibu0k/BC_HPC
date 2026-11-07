#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace km {
using Count = unsigned long long;
using Clock = std::chrono::steady_clock;
constexpr double tolerance = 0.1; // Maximum Euclidean movement of any RGB center.
struct Options {
    std::string input, output;
    int k = 8, max_iterations = 100, threads = 4, block_size = 256;
    bool gui = true, help = false;
};
struct Image {
    int width = 0, height = 0;
    std::vector<unsigned char> rgb;
    int size() const { return static_cast<int>(rgb.size() / 3); }
};
struct Result {
    std::vector<int> labels;
    std::vector<double> centers;
    int iterations = 0;
    double milliseconds = 0, kernel_ms = -1;
};
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
inline std::vector<std::string> arguments(int argc, char** argv) {
#ifdef _WIN32
    // Read the original Unicode command line, including Vietnamese file names.
    int count = 0;
    LPWSTR* wide = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!wide) throw std::runtime_error("Cannot read command line.");
    std::vector<std::string> args;
    for (int i = 1; i < count; ++i) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, s.data(), n, nullptr, nullptr);
        s.pop_back();
        args.push_back(s);
    }
    LocalFree(wide);
    return args;
#else
    return {argv + 1, argv + argc};
#endif
}
inline int integer(const std::string& value, int low, int high) {
    std::size_t used = 0;
    long long n = std::stoll(value, &used);
    if (used != value.size() || n < low || n > high)
        throw std::runtime_error("Integer outside allowed range: " + value);
    return static_cast<int>(n);
}
inline Options parse(int argc, char** argv) {
    Options o;
    auto args = arguments(argc, argv);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        if (a == "--help" || a == "-h") o.help = true;
        else if (a == "--no-gui") o.gui = false;
        else if (a == "--k" || a == "--iterations" || a == "--threads" || a == "--block-size" || a == "--output") {
            if (++i == args.size()) throw std::runtime_error("Missing value for " + a);
            if (a == "--k") o.k = integer(args[i], 1, 256);
            else if (a == "--iterations") o.max_iterations = integer(args[i], 1, 10000);
            else if (a == "--threads") o.threads = integer(args[i], 1, 1024);
            else if (a == "--block-size") o.block_size = integer(args[i], 32, 1024);
            else o.output = args[i];
        } else if (o.input.empty() && a.rfind("--", 0) != 0) o.input = a;
        else throw std::runtime_error("Unknown or extra argument: " + a);
    }
    if (!o.help && o.input.empty())
        throw std::runtime_error("Usage: program image.jpg [--k 8] [--iterations 100] [--threads 4] [--block-size 256]");
    if (!o.output.empty()) {
        auto input = std::filesystem::weakly_canonical(std::filesystem::u8path(o.input));
        auto output = std::filesystem::weakly_canonical(std::filesystem::u8path(o.output));
        if (input == output || std::filesystem::exists(output))
            throw std::runtime_error("Output already exists or is the input. Choose a new output path.");
    }
    return o;
}
inline void usage() {
    std::cout << "Usage: program image.jpg [--k 8] [--iterations 100] [--threads 4]\n"
                 "       [--block-size 256] [--no-gui] [--output new_result.png]\n"
                 "Defaults: K=8 (clamped to pixel count), seed=42, tolerance=0.1 RGB.\n"
                 "--threads applies to OpenMP; --block-size applies to CUDA. MPI: mpiexec -n 4 program image.jpg\n"
                 "Close the window or press Esc to exit. Input is never overwritten.\n";
}
inline Image load(const std::string& name) {
    std::ifstream stream(std::filesystem::u8path(name), std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open input: " + name);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(stream)), {});
    if (stream.bad()) throw std::runtime_error("Failed to read input.");
    cv::Mat bgr = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("Unsupported or damaged image: " + name);
    if (bgr.total() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 3))
        throw std::runtime_error("Image too large for MPI byte counts.");
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return {rgb.cols, rgb.rows, {rgb.data, rgb.data + rgb.total() * 3}};
}

#ifdef __CUDACC__
#define KM_HD __host__ __device__
#else
#define KM_HD
#endif
KM_HD inline int nearest(const unsigned char* pixel, const double* centers, int k) {
    double best = 1.0e300;
    int label = 0;
    for (int c = 0; c < k; ++c) {
        double r = pixel[0] - centers[3*c];
        double g = pixel[1] - centers[3*c+1];
        double b = pixel[2] - centers[3*c+2];
        double d = r*r + g*g + b*b;
        if (d < best) { best = d; label = c; }
    }
    return label;
}
#undef KM_HD
inline std::vector<double> initialize(const Image& image, int k) {
    // Deterministic farthest-first initialization avoids identical starting colors when possible.
    std::mt19937 generator(42);
    int chosen = static_cast<int>(generator() % image.size());
    std::vector<double> centers(3*k), distance(image.size(), 1.0e300);
    for (int c = 0; c < k; ++c) {
        for (int d = 0; d < 3; ++d) centers[3*c+d] = image.rgb[3*chosen+d];
        int farthest = 0;
        for (int i = 0; i < image.size(); ++i) {
            double squared = 0;
            for (int d = 0; d < 3; ++d) {
                double delta = image.rgb[3*i+d] - centers[3*c+d];
                squared += delta * delta;
            }
            distance[i] = std::min(distance[i], squared);
            if (distance[i] > distance[farthest]) farthest = i;
        }
        chosen = farthest;
    }
    return centers;
}
inline bool update(std::vector<double>& centers, const std::vector<Count>& stats) {
    double movement = 0;
    for (std::size_t c = 0; c < centers.size()/3; ++c) {
        if (stats[4*c+3] == 0) continue; // Keep the previous center for an empty cluster.
        double squared = 0;
        for (int d = 0; d < 3; ++d) {
            double next = static_cast<double>(stats[4*c+d]) / stats[4*c+3];
            double delta = next - centers[3*c+d];
            squared += delta*delta;
            centers[3*c+d] = next;
        }
        movement = std::max(movement, squared);
    }
    return movement <= tolerance*tolerance;
}
inline void accumulate(const unsigned char* pixel, int label, Count* stats) {
    for (int d = 0; d < 3; ++d) stats[4*label+d] += pixel[d];
    ++stats[4*label+3];
}
inline void present(const Image& image, const Result& result, const Options& o, const char* version) {
    cv::Mat rgb(image.height, image.width, CV_8UC3);
    double sse = 0;
    for (int i = 0; i < image.size(); ++i) {
        int c = result.labels[i];
        if (c < 0 || 3*c+2 >= static_cast<int>(result.centers.size()))
            throw std::runtime_error("Invalid output cluster label.");
        for (int d = 0; d < 3; ++d) {
            double center = result.centers[3*c+d];
            double delta = image.rgb[3*i+d] - center;
            sse += delta*delta;
            rgb.data[3*i+d] = cv::saturate_cast<unsigned char>(std::lround(center));
        }
    }
    std::cout << std::fixed << std::setprecision(6) << "version=" << version
              << " pixels=" << image.size() << " k=" << result.centers.size()/3
              << " iterations=" << result.iterations << " compute_ms=" << result.milliseconds
              << " sse=" << sse;
    if (result.kernel_ms >= 0) std::cout << " kernel_ms=" << result.kernel_ms;
    std::cout << '\n';
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    if (!o.output.empty()) {
        auto path = std::filesystem::u8path(o.output);
        std::vector<unsigned char> bytes;
        if (!cv::imencode(path.extension().string(), bgr, bytes))
            throw std::runtime_error("Cannot encode output image.");
        std::ofstream stream(path, std::ios::binary);
        if (!stream || !stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()))
            throw std::runtime_error("Cannot write output image.");
    }
    if (!o.gui) return;
    cv::Mat view;
    double scale = std::min({1.0, 1500.0/image.width, 850.0/image.height});
    cv::Size size(std::max(1, static_cast<int>(image.width*scale)),
                  std::max(1, static_cast<int>(image.height*scale)));
    cv::resize(bgr, view, size, 0, 0, cv::INTER_NEAREST);
    std::string title = std::string(version) + " | K-means result | K=" + std::to_string(result.centers.size()/3);
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, view);
    cv::resizeWindow(title, view.cols, view.rows);
    for (;;) {
        if (cv::waitKey(30) == 27) break;
        try {
            if (cv::getWindowProperty(title, cv::WND_PROP_VISIBLE) < 1) break;
        } catch (const cv::Exception& e) {
            if (e.code == cv::Error::StsNullPtr) break; // Qt can destroy its receiver when the last window closes.
            throw;
        }
    }
    cv::destroyAllWindows();
}
inline int error(const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';

    return 1;
}
} // namespace km
