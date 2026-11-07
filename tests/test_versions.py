"""Run with: python tests/test_versions.py linux/build (standard library only)."""
import argparse
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile


def ppm(path, pixels):
    path.write_bytes(f"P6\n{len(pixels)} 1\n255\n".encode() + bytes(v for p in pixels for v in p))


def read_ppm(path):
    data = path.read_bytes()
    # OpenCV writes plain P6 headers for the generated outputs.
    header = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\s", data)
    assert header, "Invalid PPM output"
    payload = data[header.end():]
    assert len(payload) == int(header[1]) * int(header[2]) * 3
    return [tuple(payload[i:i+3]) for i in range(0, len(payload), 3)]


def run(command, good=True):
    p = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=90)
    assert (p.returncode == 0) == good, (command, p.returncode, p.stdout, p.stderr)
    return p.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bin", type=Path)
    args = parser.parse_args()
    suffix = ".exe" if os.name == "nt" else ""
    binaries = [args.bin.resolve() / ("kmeans_" + version + suffix)
                for version in ("sequential", "openmp", "mpi", "cuda")]
    available = [b for b in binaries if b.exists()]
    assert binaries[0] in available, "Build the sequential target first"
    print("Testing:", ", ".join(b.name for b in available))
    for b in binaries:
        if b not in available:
            print("SKIP (not built):", b.name)
    rng = random.Random(17)
    cases = [
        ("two-colors", [(255, 0, 0)]*5 + [(0, 0, 255)]*6, 2, 100),
        ("empty-clusters", [(17, 88, 203)]*7, 6, 100),
        ("single-pixel", [(3, 7, 11)], 8, 100),
        ("mean", [(0, 0, 0), (255, 127, 63), (90, 20, 8)], 1, 100),
        ("random", [tuple(rng.randrange(256) for _ in range(3)) for _ in range(257)], 8, 100),
        ("iteration-limit", [tuple(rng.randrange(256) for _ in range(3)) for _ in range(127)], 4, 1),
    ]
    with tempfile.TemporaryDirectory(prefix="hpc-test-") as directory:
        root = Path(directory)
        for name, pixels, k, iterations in cases:
            source = root / (name + " ảnh có dấu và khoảng trắng.ppm")
            ppm(source, pixels)
            expected = None
            expected_metrics = None
            for b in available:
                output = root / (name + b.stem + ".ppm")
                command = [b, source, "--k", k, "--iterations", iterations,
                           "--no-gui", "--output", output]
                log = run(command)
                result = read_ppm(output)
                metrics = (re.search(r"iterations=(\d+)", log)[1],
                           float(re.search(r"sse=([0-9.eE+-]+)", log)[1]))
                assert all(math.isfinite(x) for x in metrics[1:])
                assert len(result) == len(pixels)
                if expected is None:
                    expected, expected_metrics = result, metrics
                assert result == expected, (name, b.name, "Different output pixels")
                assert metrics == expected_metrics, (name, b.name, "Different SSE/iterations")
                if name in ("two-colors", "empty-clusters", "single-pixel"):
                    assert result == pixels, (name, b.name, "Known colors not recovered")
                    assert metrics[1] == 0
                if name == "mean":
                    mean = tuple(math.floor(sum(p[d] for p in pixels)/len(pixels)+0.5) for d in range(3))
                    assert result == [mean]*len(pixels)
                if b.stem == "kmeans_openmp":
                    for threads in (1, 2):
                        other = root / f"{name}-omp-{threads}.ppm"
                        run([b, source, "--k", k, "--iterations", iterations, "--threads", threads,
                             "--no-gui", "--output", other])
                        assert read_ppm(other) == expected
                if b.stem == "kmeans_cuda" and name == "iteration-limit":
                    for block in (32, 128, 1024):
                        other = root / f"{name}-cuda-{block}.ppm"
                        run([b, source, "--k", k, "--iterations", iterations, "--block-size", block,
                             "--no-gui", "--output", other])
                        assert read_ppm(other) == expected
                # Refuse to overwrite the input or an existing output.
                original = source.read_bytes()
                run([b, source, "--no-gui", "--output", source], good=False)
                assert source.read_bytes() == original
                run(command, good=False)
            print("PASS:", name)
        source = root / "validation.ppm"
        ppm(source, [(0, 0, 0), (255, 255, 255)])
        broken = root / "broken.png"
        broken.write_text("not an image")
        for b in available:
            run([b, "--help"])
            for tail in ([], [source, "--k", 0], [source, "--iterations", -1],
                         [source, "--threads", 0], [source, "--block-size", 0], [source, "--k"], [broken], [root / "missing.png"]):
                run([b, *tail, "--no-gui"], good=False)
        mpi = args.bin.resolve() / ("kmeans_mpi" + suffix)
        launcher = shutil.which("mpiexec")
        if mpi.exists() and launcher:
            for processes in (1, 2, 4):
                output = root / f"mpi-{processes}.ppm"
                run([launcher, "-n", processes, mpi, source, "--k", 2, "--no-gui", "--output", output])
                assert read_ppm(output) == [(0, 0, 0), (255, 255, 255)]
            print("PASS: MPI 1/2/4 ranks, including zero-pixel ranks")
    print("PASS: validation, deterministic agreement, and input preservation")


if __name__ == "__main__":
    main()
