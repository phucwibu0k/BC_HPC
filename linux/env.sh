# Optional: source linux/env.sh to use the tools prepared in this user's home.
hpc_tools="$HOME/.local/share/bc-hpc-tools"
hpc_compiler="$HOME/.local/share/bc-hpc-compiler/bin/x86_64-conda-linux-gnu-g++"
if [ -d "$hpc_tools/bin" ]; then
    export PATH="$hpc_tools/bin:$PATH"
fi
if [ -x "$hpc_tools/nvidia/cu13/bin/nvcc" ]; then
    export CUDACXX="$hpc_tools/nvidia/cu13/bin/nvcc"
    if [ -x "$hpc_compiler" ]; then export CUDAHOSTCXX="$hpc_compiler"; fi
fi
unset hpc_tools hpc_compiler
