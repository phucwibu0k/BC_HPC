#include "kmeans_common.hpp"

km::Result segment(const km::Image& image, const km::Options& o) {
    km::Result result;
    result.centers = km::initialize(image, std::min(o.k, image.size()));
    int k = static_cast<int>(result.centers.size()/3);
    result.labels.resize(image.size());
    std::vector<km::Count> stats(4*k);
    auto start = km::Clock::now();
    for (int iteration = 0; iteration < o.max_iterations; ++iteration) {
        std::fill(stats.begin(), stats.end(), 0);
        for (int i = 0; i < image.size(); ++i) {
            int c = km::nearest(&image.rgb[3*i], result.centers.data(), k);
            km::accumulate(&image.rgb[3*i], c, stats.data());
        }
        ++result.iterations;
        if (km::update(result.centers, stats)) break;
    }
    // Reassign once so labels always correspond to the final centers, even at the iteration limit.
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
        km::present(image, segment(image, options), options, "Sequential");
        return 0;
    } catch (const std::exception& e) { return km::error(e); }
}
