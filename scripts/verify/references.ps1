# The references of a milestone's statistical checks, rendered once by an earlier step's RayTracingGPU, whose kernels
# the SASS gate proves the same machine code as every later build's. At -w 480 --height 270, float, as PFMs.
#   -List m5 (the default): Step 85's exes (out\m5\bin85), into State\ref\ (out\m5\ref\), every render with
#   --quality-kernel, so these images don't depend on the graphics driver as OptiX's do:
#     <scene>-<strategy>-s1.pfm and -s2.pfm   cornell, final, studio and veach under mis; bouncing and cornell under
#                                             material; smoke under nee: 1024 samples, seeds 1 and 2 (compare.ps1's
#                                             "STRATEGY@quality:ref" side; 14 renders);
#     <scene>-mis2048-s101.pfm and -s102.pfm  cornell, studio and final under mis, 2048 samples, seeds 101 and 102 (the
#                                             denoiser's references; 6 renders).
#   -List m4: milestone 4's list, by Step 70's exes (out\m4\bin70) into out\m4\ref\, kept as it was (history):
#     <scene>-<strategy>-s1.pfm and -s2.pfm   cornell, light and final under material and mixture, 1024 samples, seeds
#                                             1 and 2;
#     <scene>-material256-s1.pfm              all 13 scenes under -p material, 256 samples, seed 1.
# A reference that exists is kept. Each render's "Rays traced" and "Shadow rays" lines go into rays.txt beside the
# references (name: line), without the speed, for the comparisons of rays per sample. Renders go through gpu.ps1.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\references.ps1 -Step 86 [-List m5] [-State out\m5]
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string]$List = "m5",
    [string]$Reference = "",
    [string]$State = "out\m5"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"

$renders = @()
if ($List -eq "m5") {
    if (-not $Reference) { $Reference = "out\m5\bin85" }
    $stateDir = if ([IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $project $State }
    $refs = Join-Path $stateDir "ref"
    foreach ($r in @(@("cornell", "mis"), @("final", "mis"), @("studio", "mis"), @("veach", "mis"),
            @("bouncing", "material"), @("cornell", "material"), @("smoke", "nee"))) {
        foreach ($seed in @(1, 2)) {
            $renders += , @(("{0}-{1}-s{2}" -f $r[0], $r[1], $seed),
                @($r[0], "-p", $r[1], "--quality-kernel", "-s", "1024", "--seed", "$seed"))
        }
    }
    foreach ($scene in @("cornell", "studio", "final")) {
        foreach ($seed in @(101, 102)) {
            $renders += , @("$scene-mis2048-s$seed",
                @($scene, "-p", "mis", "--quality-kernel", "-s", "2048", "--seed", "$seed"))
        }
    }
}
elseif ($List -eq "m4") {
    if (-not $Reference) { $Reference = "out\m4\bin70" }
    $refs = Join-Path $project "out\m4\ref"
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
}
else { throw "references.ps1: -List is m5 or m4, not '$List'" }
$gpu = Join-Path $project "$Reference\RayTracingGPU.exe"
New-Item -ItemType Directory -Force $refs | Out-Null

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
    # The rays and the shadow rays (only under nee and mis), kept for the comparisons of rays per sample: the speed
    # after the counts is dropped (no figure of speed is kept)
    foreach ($kind in @("Rays traced", "Shadow rays")) {
        $line = @($lines | Where-Object { $_ -match "^$kind" }) | Select-Object -First 1
        if (-not $line) { continue }
        $line = $line -replace ', [0-9.]+ million per second', ''
        Add-Content -LiteralPath (Join-Path $refs "rays.txt") -Value ("{0}: {1}" -f $r[0], $line)
        Write-Output ("{0}: {1}" -f $r[0], $line)
    }
}
