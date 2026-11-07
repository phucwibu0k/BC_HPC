#include "kmeans_common.hpp"
#include <omp.h>

km::Result segment(const km::Image& image, const km::Options& o) {
    km::Result result;
    result.centers = km::initialize(image, std::min(o.k, image.size()));
    int k = static_cast<int>(result.centers.size()/3);
    result.labels.resize(image.size());
    omp_set_dynamic(0);
    omp_set_num_threads(o.threads);
    std::vector<km::Count> stats(4*k), local(static_cast<std::size_t>(o.threads)*4*k);
    auto start = km::Clock::now();
    for (int iteration = 0; iteration < o.max_iterations; ++iteration) {
        std::fill(local.begin(), local.end(), 0);
        // Each thread owns its sums; the implicit barrier precedes the merge.
        #pragma omp parallel
        {
            auto* sums = local.data() + static_cast<std::size_t>(omp_get_thread_num())*4*k;
            #pragma omp for schedule(static)
            for (int i = 0; i < image.size(); ++i) {
                int c = km::nearest(&image.rgb[3*i], result.centers.data(), k);
                km::accumulate(&image.rgb[3*i], c, sums);
            }
        }
        std::fill(stats.begin(), stats.end(), 0);
        for (int t = 0; t < o.threads; ++t)
            for (int j = 0; j < 4*k; ++j) stats[j] += local[static_cast<std::size_t>(t)*4*k+j];
        ++result.iterations;
        if (km::update(result.centers, stats)) break;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < image.size(); ++i)
        result.labels[i] = km::nearest(&image.rgb[3*i], result.centers.data(), k);
    result.milliseconds = km::elapsed(start);
    return result;
}

int main(int argc, char** argv) {
    try {
        auto options = km::parse(argc, argv);
        if (options.help) { km::usage(); return 0; }
        auto image = km::load(options.input);
        std::cout << "threads_requested=" << options.threads << '\n';
        km::present(image, segment(image, options), options, "OpenMP");
        return 0;
    } catch (const std::exception& e) { return km::error(e); }
}
