#include "kmeans_common.hpp"
#include <cuda_runtime.h>

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count*sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Events {
    cudaEvent_t start = nullptr, end = nullptr;
    Events() {
        check(cudaEventCreate(&start));
        auto status = cudaEventCreate(&end);
        if (status != cudaSuccess) { cudaEventDestroy(start); check(status); }
    }
    ~Events() { cudaEventDestroy(start); cudaEventDestroy(end); }
};
__global__ void assign_and_sum(const unsigned char* rgb, int n, const double* centers,
                               int k, km::Count* sums) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = km::nearest(rgb+3*i, centers, k);
    for (int d = 0; d < 3; ++d) atomicAdd(sums+4*c+d, static_cast<km::Count>(rgb[3*i+d]));
    atomicAdd(sums+4*c+3, 1ULL);
}
__global__ void update_centers(double* centers, const km::Count* sums, int k, int* changed) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= k || sums[4*c+3] == 0) return;
    double squared = 0;
    for (int d = 0; d < 3; ++d) {
        double next = static_cast<double>(sums[4*c+d])/sums[4*c+3];
        double delta = next-centers[3*c+d];
        squared += delta*delta;
        centers[3*c+d] = next;
    }
    if (squared > km::tolerance*km::tolerance) atomicExch(changed, 1);
}
__global__ void final_labels(const unsigned char* rgb, int n, const double* centers, int k, int* labels) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) labels[i] = km::nearest(rgb+3*i, centers, k);
}

km::Result segment(const km::Image& image, const km::Options& o) {
    km::Result result;
    result.centers = km::initialize(image, std::min(o.k, image.size()));
    int n = image.size(), k = static_cast<int>(result.centers.size()/3);
    result.labels.resize(n);
    check(cudaFree(nullptr)); // Initialize CUDA outside the measured interval.
    Events events;
    int block = o.block_size;
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    if (block > properties.maxThreadsPerBlock)
        throw std::runtime_error("Block size exceeds GPU maxThreadsPerBlock.");
    std::cout << "block_size=" << block << '\n';
    auto start = km::Clock::now();
    {
        DeviceBuffer<unsigned char> rgb(image.rgb.size());
        DeviceBuffer<double> centers(3*k);
        DeviceBuffer<int> labels(n), changed(1);
        DeviceBuffer<km::Count> sums(4*k);
        check(cudaMemcpy(rgb.data, image.rgb.data(), image.rgb.size(), cudaMemcpyHostToDevice));
        check(cudaMemcpy(centers.data, result.centers.data(), 3*k*sizeof(double), cudaMemcpyHostToDevice));
        int grid = (n+block-1)/block;
        result.kernel_ms = 0;
        for (int iteration = 0; iteration < o.max_iterations; ++iteration) {
            check(cudaMemset(sums.data, 0, 4*k*sizeof(km::Count)));
            check(cudaMemset(changed.data, 0, sizeof(int)));
            check(cudaEventRecord(events.start));
            assign_and_sum<<<grid, block>>>(rgb.data, n, centers.data, k, sums.data);
            check(cudaGetLastError());
            update_centers<<<(k+block-1)/block, block>>>(centers.data, sums.data, k, changed.data);
            check(cudaGetLastError());
            check(cudaEventRecord(events.end));
            check(cudaEventSynchronize(events.end));
            float ms = 0;
            check(cudaEventElapsedTime(&ms, events.start, events.end));
            result.kernel_ms += ms;
            int moved = 0;
            check(cudaMemcpy(&moved, changed.data, sizeof(int), cudaMemcpyDeviceToHost));
            ++result.iterations;
            if (!moved) break;
        }
        check(cudaEventRecord(events.start));
        final_labels<<<grid, block>>>(rgb.data, n, centers.data, k, labels.data);
        check(cudaGetLastError());
        check(cudaEventRecord(events.end));
        check(cudaEventSynchronize(events.end));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, events.start, events.end));
        result.kernel_ms += ms;
        check(cudaMemcpy(result.labels.data(), labels.data, n*sizeof(int), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(result.centers.data(), centers.data, 3*k*sizeof(double), cudaMemcpyDeviceToHost));
    }
    result.milliseconds = km::elapsed(start);
    return result;
}

int main(int argc, char** argv) {
    try {
        auto o = km::parse(argc, argv);
        if (o.help) { km::usage(); return 0; }
        auto image = km::load(o.input);
        km::present(image, segment(image, o), o, "CUDA");
        return 0;
    } catch (const std::exception& e) { return km::error(e); }
}
