# The references of the milestone's statistical checks, rendered once by Step 70's RayTracingGPU (out\m4\bin70), whose
# kernels the SASS gate proves the same as every later build's. At -w 480 --height 270, float, into out\m4\ref\:
#   <scene>-<strategy>-s1.pfm and -s2.pfm   cornell, light and final under material and mixture, 1024 samples, seeds
#                                           1 and 2 (compare.ps1's "ref" side);
#   <scene>-material256-s1.pfm              all 13 scenes under -p material, 256 samples, seed 1 (Step 73's check of
#                                           the quality kernel's base mode).
# A reference that exists is kept. Renders go through gpu.ps1.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\references.ps1 -Step 71
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string]$Reference = "out\m4\bin70"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$gpu = Join-Path $project "$Reference\RayTracingGPU.exe"
$refs = Join-Path $project "out\m4\ref"
New-Item -ItemType Directory -Force $refs | Out-Null

$renders = @()
foreach ($scene in @("cornell", "light", "final")) {
    foreach ($strategy in @("material", "mixture")) {
        foreach ($seed in @(1, 2)) {
            $renders += , @("$scene-$strategy-s$seed", @($scene, "-p", $strategy, "-s", "1024", "--seed", "$seed"))
        }
    }
}
foreach ($scene in @("materials", "weekend", "bouncing", "checker", "earth", "perlin", "quads", "light", "cornell",
        "smoke", "final", "aluminum", "glass")) {
    $renders += , @("$scene-material256-s1", @($scene, "-p", "material", "-s", "256", "--seed", "1"))
}

foreach ($r in $renders) {
    $path = Join-Path $refs ($r[0] + ".pfm")
    if (Test-Path -LiteralPath $path) { Write-Output "$($r[0]): kept"; continue }
    $all = @("-Step", "$Step", "-Label", "ref $($r[0])", "--", $gpu) + $r[1] +
        @("-w", "480", "--height", "270", "-o", $path)
    $lines = @(& $gpuScript @all)
    if ($LASTEXITCODE -ne 0) {
        $lines | ForEach-Object { Write-Output "  $_" }
        throw "references.ps1: $($r[0]) failed"
    }
    # The rays, kept for the comparisons of rays per sample (rays.txt beside the references: name, line)
    $rays = (@($lines | Where-Object { $_ -match '^Rays traced' }) | Select-Object -First 1) -replace
        ', [0-9.]+ million per second', ''
    Add-Content -LiteralPath (Join-Path $refs "rays.txt") -Value ("{0}: {1}" -f $r[0], $rays)
    Write-Output ("{0}: {1}" -f $r[0], $rays)
}
