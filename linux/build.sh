#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$source_dir" -B "$source_dir/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$source_dir/build" --parallel
