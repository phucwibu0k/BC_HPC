#include "kmeans_common.hpp"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    try {
        auto o = km::parse(argc, argv);
        if (o.help) {
            if (rank == 0) km::usage();
            MPI_Finalize();
            return 0;
        }
        km::Image image;
        km::Result result;
        int metadata[2] = {};
        if (rank == 0) {
            image = km::load(o.input);
            metadata[0] = image.size();
            metadata[1] = std::min(o.k, image.size());
            result.centers = km::initialize(image, metadata[1]);
            std::cout << "processes=" << processes << '\n';
        }
        MPI_Bcast(metadata, 2, MPI_INT, 0, MPI_COMM_WORLD);
        int n = metadata[0], k = metadata[1];
        result.centers.resize(3*k);
        MPI_Bcast(result.centers.data(), 3*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<int> counts(processes), offsets(processes), bytes(processes), byte_offsets(processes);
        for (int p = 0; p < processes; ++p) {
            counts[p] = n/processes + (p < n%processes ? 1 : 0);
            offsets[p] = p == 0 ? 0 : offsets[p-1]+counts[p-1];
            bytes[p] = 3*counts[p];
            byte_offsets[p] = 3*offsets[p];
        }
        std::vector<unsigned char> pixels(bytes[rank]);
        std::vector<int> labels(counts[rank]);
        std::vector<km::Count> local(4*k), total(4*k);
        if (rank == 0) result.labels.resize(n);
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        MPI_Scatterv(image.rgb.data(), bytes.data(), byte_offsets.data(), MPI_UNSIGNED_CHAR,
                     pixels.data(), bytes[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        for (int iteration = 0; iteration < o.max_iterations; ++iteration) {
            std::fill(local.begin(), local.end(), 0);
            for (int i = 0; i < counts[rank]; ++i) {
                int c = km::nearest(&pixels[3*i], result.centers.data(), k);
                km::accumulate(&pixels[3*i], c, local.data());
            }
            MPI_Allreduce(local.data(), total.data(), 4*k, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            ++result.iterations;
            if (km::update(result.centers, total)) break;
        }
        for (int i = 0; i < counts[rank]; ++i)
            labels[i] = km::nearest(&pixels[3*i], result.centers.data(), k);
        MPI_Gatherv(labels.data(), counts[rank], MPI_INT, result.labels.data(),
                    counts.data(), offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);
        double duration = (MPI_Wtime()-start)*1000;
        MPI_Reduce(&duration, &result.milliseconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        // Release all workers before rank 0 waits for the user to close the image window.
        MPI_Finalize();
        if (rank == 0) km::present(image, result, o, "MPI");
        return 0;
    } catch (const std::exception& e) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (finalized) return rank == 0 ? km::error(e) : 1;
        // Abort every rank before displaying an error so workers cannot hang in a collective.
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
