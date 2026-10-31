# The denoiser's evaluation, its thresholds fixed before any run: does OptiX's AI denoiser remove most of the noise of
# a 16-sample image without moving its mean?
#   - One queued GpuTests process, the suite denoise-eval, renders through gpu::render with denoising on (the library
#     path of RayTracingGPU --denoise), at 480x270 under -p mis, and writes into State\runs\stepNN\:
#       <scene>-N<seed>.pfm, <scene>-N<seed>-denoised.pfm   16 samples, seeds 1 and 2, for cornell, studio, final and
#                                                           cornell-bust: the noisy image and its denoised twin, named
#                                                           as RayTracingGPU --denoise names them
#       cornell-bust-mis2048-s101.pfm, -s102.pfm             cornell-bust's references: 2048 samples with OptiX on this
#                                                           driver (cornell's, studio's and final's are State\ref\'s,
#                                                           rendered once by Step 85's quality kernel: references.ps1)
#       <scene>-1024spp-N1.pfm, -denoised.pfm                cornell and studio at 1024 samples, seed 1: information
#                                                           only (where denoising stops helping)
#       <scene>-N1-again.pfm, -again-denoised.pfm            a seed-1 image rendered and denoised a second time in the
#                                                           same process (at least one scene): the determinism check
#     One process, so OptiX's denoiser is created once rather than for every image.
#   - Then, on the CPU, PfmStats error for each scene and seed against R = the average of its two references (seeds 101
#     and 102), with the thresholds: relMSE'(denoised) <= relMSE'(noisy)/4, and the mean within 2% of R's in every
#     channel. The gain is relative: in MSE' a lamp's edges and the studio's softbox reflections, which the denoiser
#     softens, hold almost all of the denoised error (PfmStats prints that gain too, without a verdict). The
#     1024-sample pairs are printed without a verdict.
#   - Determinism: each -again-denoised.pfm must hold the bytes of the first denoised image of the same input (and an
#     -again.pfm those of the noisy one, which shows that the input was the same).
# A failure stops the step: it goes to the user, and no threshold is changed after a run.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\denoise.ps1 -Step 94 [-State out\m5]
#              [-Build out\build\x64-release] [-NoRender]
# -NoRender evaluates the files an earlier run wrote, without the GPU. Exit code 0 pass, 1 fail, 2 an error (a file
# missing, GpuTests or PfmStats that couldn't run).
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string]$State = "out\m5",
    [string]$Build = "out\build\x64-release",
    [switch]$NoRender
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$buildDir = if ([IO.Path]::IsPathRooted($Build)) { $Build } else { Join-Path $project $Build }
$tests = Join-Path $buildDir "GpuTests.exe"
$stats = Join-Path $buildDir "PfmStats.exe"
$stateDir = if ([IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $project $State }
$work = Join-Path $stateDir ("runs\step{0}" -f $Step)
$refs = Join-Path $stateDir "ref"
New-Item -ItemType Directory -Force $work | Out-Null

# The protocol and its thresholds, fixed before the first run and never retuned after one
$scenes = @("cornell", "studio", "final", "cornell-bust")
$seeds = @(1, 2)
$samples = 16
$thresholds = @("--min-gain", "4", "--max-drift", "0.02", "--tiles", "32")
$infoScenes = @("cornell", "studio")
$infoSamples = 1024

# A scene's two references: cornell-bust's are this step's renders with OptiX, the others Step 85's
function Get-References([string]$scene) {
    $folder = if ($scene -eq "cornell-bust") { $work } else { $refs }
    return @((Join-Path $folder "$scene-mis2048-s101.pfm"), (Join-Path $folder "$scene-mis2048-s102.pfm"))
}

# PfmStats error with its lines indented; its exit code (2: it couldn't run, an error)
function Invoke-Error([string[]]$arguments) {
    $out = @(& $stats error @arguments 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host "    $_" }
    return $code
}

function Test-SameBytes([string]$first, [string]$second) {
    $a = (Get-FileHash -LiteralPath $first -Algorithm SHA256).Hash
    $b = (Get-FileHash -LiteralPath $second -Algorithm SHA256).Hash
    return $a -eq $b
}

try {
    foreach ($exe in @($tests, $stats)) {
        if (-not (Test-Path -LiteralPath $exe)) { throw "no $exe (build it first)" }
    }
    $suiteFailed = $false
    if (-not $NoRender) {
        Write-Output ("denoise-eval: GpuTests renders and denoises into {0} (one process, one denoiser)" -f $work)
        $all = @("-Step", "$Step", "-Label", "denoise-eval", "--", $tests, "denoise-eval", "--require-optix",
            "--write", $work)
        $lines = @(& $gpuScript @all)
        $code = $LASTEXITCODE
        $lines | ForEach-Object { Write-Output "  $_" }
        if ($code -ne 0 -and $code -ne 1) { throw "GpuTests denoise-eval couldn't run (exit code $code)" }
        $suiteFailed = $code -eq 1
        Write-Output ("denoise-eval: exit code {0}" -f $code)
    }

    # Every file the evaluation reads, checked before any is read: a name the suite wrote differently is an error here,
    # never a check that silently didn't run
    $needed = @()
    foreach ($scene in $scenes) {
        $needed += Get-References $scene
        foreach ($seed in $seeds) {
            $needed += Join-Path $work ("{0}-N{1}.pfm" -f $scene, $seed)
            $needed += Join-Path $work ("{0}-N{1}-denoised.pfm" -f $scene, $seed)
        }
    }
    foreach ($scene in $infoScenes) {
        $needed += Join-Path $work ("{0}-{1}spp-N1.pfm" -f $scene, $infoSamples)
        $needed += Join-Path $work ("{0}-{1}spp-N1-denoised.pfm" -f $scene, $infoSamples)
    }
    $missing = @($needed | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -gt 0) {
        $missing | ForEach-Object { Write-Output "  missing: $_" }
        throw ("{0} files the evaluation reads are missing" -f $missing.Count)
    }

    # The verdicts: each scene and seed
    $pairs = 0
    $failedPairs = @()
    foreach ($scene in $scenes) {
        $references = Get-References $scene
        foreach ($seed in $seeds) {
            Write-Output ("{0}, seed {1}, {2} samples:" -f $scene, $seed, $samples)
            $noisy = Join-Path $work ("{0}-N{1}.pfm" -f $scene, $seed)
            $denoised = Join-Path $work ("{0}-N{1}-denoised.pfm" -f $scene, $seed)
            $code = Invoke-Error (@($noisy, $denoised) + $references + $thresholds + @("--spp", "$samples"))
            if ($code -ne 0 -and $code -ne 1) { throw "PfmStats error couldn't evaluate $denoised (exit code $code)" }
            $pairs++
            if ($code -eq 1) { $failedPairs += "$scene seed $seed" }
        }
    }

    # Where denoising stops helping: the same measures at 1024 samples, no verdict
    foreach ($scene in $infoScenes) {
        Write-Output ("INFO {0}, seed 1, {1} samples (no verdict):" -f $scene, $infoSamples)
        $noisy = Join-Path $work ("{0}-{1}spp-N1.pfm" -f $scene, $infoSamples)
        $denoised = Join-Path $work ("{0}-{1}spp-N1-denoised.pfm" -f $scene, $infoSamples)
        $code = Invoke-Error (@($noisy, $denoised) + (Get-References $scene) + $thresholds +
            @("--spp", "$infoSamples"))
        if ($code -ne 0 -and $code -ne 1) { throw "PfmStats error couldn't evaluate $denoised (exit code $code)" }
    }

    # Determinism: the same input denoised twice in one process, the same bytes
    $again = @(Get-ChildItem -LiteralPath $work -Filter "*-N1-again-denoised.pfm" | Sort-Object Name)
    if ($again.Count -eq 0) {
        throw ("no *-N1-again-denoised.pfm in {0}: the determinism check has nothing to read" -f $work)
    }
    $differ = @()
    foreach ($file in $again) {
        $first = Join-Path $work ($file.Name -replace '-again-denoised\.pfm$', '-denoised.pfm')
        if (-not (Test-Path -LiteralPath $first)) { throw ("no {0} to compare {1} with" -f $first, $file.Name) }
        # The noisy images too: both must be there, the same bytes, or the two denoises didn't get the same input
        $noisyFirst = Join-Path $work ($file.Name -replace '-again-denoised\.pfm$', '.pfm')
        $noisyAgain = Join-Path $work ($file.Name -replace '-again-denoised\.pfm$', '-again.pfm')
        foreach ($noisy in @($noisyFirst, $noisyAgain)) {
            if (-not (Test-Path -LiteralPath $noisy)) {
                throw ("no {0}: the determinism check can't tell whether {1} had the same input" -f $noisy, $file.Name)
            }
        }
        if (-not (Test-SameBytes $noisyFirst $noisyAgain)) {
            throw ("{0} isn't the bytes of {1}: the two denoises didn't get the same input" -f $noisyAgain,
                $noisyFirst)
        }
        $same = Test-SameBytes $first $file.FullName
        Write-Output ("determinism: {0} and {1}: {2}" -f (Split-Path -Leaf $first), $file.Name,
            $(if ($same) { "the same bytes" } else { "DIFFERENT" }))
        if (-not $same) { $differ += $file.Name }
    }

    $problems = @()
    if ($failedPairs.Count -gt 0) { $problems += ("the criteria failed on " + ($failedPairs -join ", ")) }
    if ($differ.Count -gt 0) { $problems += ("not deterministic: " + ($differ -join ", ")) }
    if ($suiteFailed) { $problems += "GpuTests denoise-eval failed a check of its own" }
    if ($problems.Count -eq 0) {
        Write-Output ("Evaluation: PASS ({0} of {0} images, {1} denoised twice with the same bytes)" -f $pairs,
            $again.Count)
        exit 0
    }
    # (The parentheses first: -f binds tighter than +)
    Write-Output (("Evaluation: FAIL ({0}). The thresholds were fixed before the run: report it to the user, never " +
        "retune them") -f ($problems -join "; "))
    exit 1
}
catch {
    Write-Output "Error: $($_.Exception.Message)"
    exit 2
}
