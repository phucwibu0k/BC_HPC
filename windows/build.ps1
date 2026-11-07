param([Parameter(Mandatory=$true)][string]$OpenCVDir)
$ErrorActionPreference = 'Stop'
cmake -S $PSScriptRoot -B "$PSScriptRoot/build" -A x64 "-DOpenCV_DIR=$OpenCVDir"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build "$PSScriptRoot/build" --config Release --parallel
exit $LASTEXITCODE
