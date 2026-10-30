# compute-sanitizer, CUDA's checker of device memory, on one command: memcheck (reads and writes out of bounds,
# misaligned or into freed memory, and CUDA calls that fail) and initcheck (reads of GPU memory never written). Each
# tool runs the program once, slowed 10 to 100 times, so the commands are tiny renders (32x18, 4 samples) or GpuTests
# --quick.
# -Optix, for a command that runs OptiX (--optix, a mesh scene, the denoiser, an OptiX suite): initcheck gets
# --check-optix (without it initcheck doesn't follow OptiX's launches), and both tools get --check-optix-leaks (every
# module, program group, pipeline and denoiser still alive when the program destroys its OptiX context). Both options
# are switches without a value in CUDA 13.4's compute-sanitizer (its documentation's table of options), whose OptiX
# support covers OptiX's API 7.0 to 9.0. The programs are compiled at OptiX's debug level NONE as always (FULL would
# check another module): a report inside OptiX's own code shows its frames as "NVIDIA Internal", and one inside the
# denoiser's own kernels is listed, not chased.
# Both runs go through gpu.ps1, the queue. It passes when both print "ERROR SUMMARY: 0 errors" and the program exits 0.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\sanitize.ps1 -Step 89 -Optix -Label cornell -- `
#       out\build\x64-release\RayTracingGPU.exe cornell --optix -w 32 --height 18 -s 4 -o out\m5\runs\step89\s.pfm
# Exit code 0 when both are clean.
$ErrorActionPreference = "Stop"
$Step = -1
$Label = "sanitizer"
$Optix = $false
$Command = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    if ($a -eq "-Step") { $Step = [int]$args[++$i] }
    elseif ($a -eq "-Label") { $Label = [string]$args[++$i] }
    elseif ($a -eq "-Optix") { $Optix = $true }
    elseif ($a -eq "--") { $Command = @($args | Select-Object -Skip ($i + 1) | ForEach-Object { [string]$_ }); break }
    else { $Command = @($args | Select-Object -Skip $i | ForEach-Object { [string]$_ }); break }
}
if ($Step -lt 0 -or $Command.Count -eq 0) {
    throw "usage: sanitize.ps1 -Step NN [-Label L] [-Optix] -- <program> <arguments>"
}

$cuda = if ($env:CUDA_PATH) { $env:CUDA_PATH } else {
    (Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -Filter "v*" |
        Sort-Object Name | Select-Object -Last 1).FullName
}
$sanitizer = Join-Path $cuda "compute-sanitizer\compute-sanitizer.exe"
if (-not (Test-Path -LiteralPath $sanitizer)) { throw "sanitize.ps1: no $sanitizer" }
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"

# The OptiX options of each tool (none without -Optix)
$optixOptions = @{ memcheck = @(); initcheck = @() }
if ($Optix) {
    $optixOptions.memcheck = @("--check-optix-leaks")
    $optixOptions.initcheck = @("--check-optix", "--check-optix-leaks")
}

$failures = 0
foreach ($tool in @("memcheck", "initcheck")) {
    $all = @("-Step", "$Step", "-Label", "sanitizer $tool $Label", "--", $sanitizer, "--tool", $tool) +
        $optixOptions[$tool] + @("--error-exitcode", "99") + $Command
    $lines = @(& $gpuScript @all)
    $code = $LASTEXITCODE
    $summary = @($lines | Where-Object { $_ -match "ERROR SUMMARY" }) | Select-Object -Last 1
    if ($code -eq 0 -and $summary -match "ERROR SUMMARY: 0 errors") {
        Write-Output ("{0} {1}: {2}" -f $tool, $Label, $summary.Trim())
    }
    else {
        Write-Output ("{0} {1}: FAIL (exit code {2})" -f $tool, $Label, $code)
        $lines | Where-Object { $_ -match "=========|Error" } | Select-Object -First 30 | ForEach-Object { "  $_" }
        $failures++
    }
}
if ($failures -gt 0) { exit 1 }
exit 0
