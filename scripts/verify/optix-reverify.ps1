# The re-verification of the OptiX path after a graphics driver update. OptiX compiles its programs inside the driver,
# so a new driver may change the bytes OptiX renders while every CUDA kernel stays as it was (their machine code is in
# the exes): the stamped byte gates (carry.ps1 -Set optix, hostgate's stamped chained sections) then print RE-VERIFY
# in a step that didn't change the programs. This script runs again the checks that stand behind those bytes, so that
# they can be recorded again:
#   1. optix-warm: one tiny render, since OptiX's cache of compiled programs is per driver and the first run compiles
#      them all (CPU work, many seconds, counted in the step's GPU time and reported apart by gpu.ps1 -Report). If it
#      fails, OptiX doesn't run with this driver and nothing else is tried;
#   2. GpuTests optix-runtime optix-tracer optix-render --require-optix: the runtime's probes, the OptiX tracer against
#      the BVH ray by ray, the same-seed survey of every GPU scene;
#   3. the white furnaces under --optix: furnace and env-furnace, -p material, 480x270, 64 samples, every value exactly
#      1.0 (PfmStats stats --equal 1 --share 1);
#   4. compare.ps1 on cornell, final and studio: mis@quality:ref (Step 85's references, which no driver changes)
#      against mis@optix:3,4. A protocol that comes out VOID (a control failed: the comparison can't be trusted on
#      that scene) runs once more, B's seeds +10 (A8.3); VOID again stops the script: nothing more is run, ask the user;
#   5. what later steps add (Claude fills these sections in when integrating Steps 91, 92 and 94: see below).
# Every run goes through gpu.ps1 into State\runs\stepNN\, each queued run's whole output in <label>.txt there.
# All pass: Claude stamps the stamped entries anew (carry.ps1 -Restamp for the entries whose bytes the step's check run
# found the ones recorded, -Record for the others; hostgate.ps1 -Accept), and the commit says "re-verified after a
# graphics driver update", naming no version. Any failure: stop and ask the user (a driver change or a latent bug; its
# diagnosis may pass the budget). The checks cost about 25 to 35 s of GPU plus the compile: past the step's 55 s, or
# when a job would take it past 60 s, gpu.ps1 asks for -Approved.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\optix-reverify.ps1 -Step 90
# Exit code 0 when every check passes, 1 when one fails, 2 when nothing is re-verified (a build without OptiX
# programs; a refusal of gpu.ps1, in this script's runs or compare.ps1's (its exit code 4); a protocol VOID twice, no
# check having failed before it; an error).
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string]$State = "out\m5",
    [string]$Build = "out\build\x64-release"
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$compareScript = Join-Path $PSScriptRoot "compare.ps1"
$buildDir = if ([IO.Path]::IsPathRooted($Build)) { $Build } else { Join-Path $project $Build }
$stateDir = if ([IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $project $State }
$gpu = Join-Path $buildDir "RayTracingGPU.exe"
$tests = Join-Path $buildDir "GpuTests.exe"
$stats = Join-Path $buildDir "PfmStats.exe"
$work = Join-Path $stateDir ("runs\step{0}" -f $Step)
$checks = 0
$failures = 0

# One run through the queue: its whole output kept as <label>.txt in the work folder. A refusal of the queue (the
# step's GPU budget, a fullscreen application in front) or a program it can't find stops the script: nothing was
# checked, so nothing passed or failed
function Invoke-Queued([string]$label, [string[]]$command) {
    $all = @("-Step", "$Step", "-Label", $label, "--") + $command
    $lines = @(& $gpuScript @all)
    $code = $LASTEXITCODE
    [IO.File]::WriteAllLines((Join-Path $work "$label.txt"), [string[]]$lines)
    $refused = @($lines | Where-Object {
            $_ -match '^gpu\.ps1: (a fullscreen|Step \d+ has used|no program|the GPU stayed)' })
    if ($code -ne 0 -and $refused.Count -gt 0) {
        $refused | ForEach-Object { Write-Host "  $_" }
        Write-Host "optix-reverify: gpu.ps1 didn't run '$label' (exit code $code): nothing re-verified"
        exit 2
    }
    return @{ code = $code; lines = $lines }
}

# A program outside the queue (PfmStats reads files only): its output and exit code
function Invoke-Native([string]$exe, [string[]]$arguments) {
    $ErrorActionPreference = "Continue" # a native program's stderr must not stop the script
    $out = @(& $exe @arguments 2>&1 | ForEach-Object { "$_" })
    return @{ code = $LASTEXITCODE; lines = $out }
}

function Write-Check([string]$what, [bool]$pass, [string]$detail) {
    $script:checks++
    if (-not $pass) { $script:failures++ }
    Write-Output ("{0}: {1} ({2})" -f $what, $(if ($pass) { "PASS" } else { "FAIL" }), $detail)
}

try {
    $sha = Join-Path $buildDir "optix\optix_programs.sha256"
    if (-not (Test-Path -LiteralPath $sha)) {
        Write-Output "optix-reverify: no $sha, so no OptiX programs in this build: nothing to re-verify"
        exit 2
    }
    New-Item -ItemType Directory -Force $work | Out-Null
    $programs = if ([IO.File]::ReadAllText($sha) -match '[0-9A-Fa-f]{64}') { $Matches[0].Substring(0, 8).ToLower() }
        else { "(no SHA-256 in $sha)" }
    Write-Output ("optix-reverify, Step {0}: the OptiX programs {1}, under the driver gpu.ps1 logs" -f $Step, $programs)

    # --- 1. the warm-up: the programs compiled for this driver -------------------------------------------------------
    $run = Invoke-Queued "optix-warm" @($gpu, "furnace", "--optix", "-p", "material", "-w", "8", "--height", "8",
        "-s", "1", "-o", (Join-Path $work "optix-warm.pfm"))
    Write-Check "optix-warm, OptiX's first compile under this driver" ($run.code -eq 0) "exit code $($run.code)"
    if ($run.code -ne 0) {
        $run.lines | Select-Object -Last 12 | ForEach-Object { "  $_" }
        Write-Output "Re-verification: FAIL (OptiX doesn't run with this driver): stop and ask the user, record nothing"
        exit 1
    }

    # --- 2. the suites: the runtime, the tracer ray by ray, the same-seed survey --------------------------------------
    $run = Invoke-Queued "reverify-suites" @($tests, "optix-runtime", "optix-tracer", "optix-render", "optix-triangles",
        "--require-optix")
    $run.lines | Where-Object { $_ -match '(: |, )(FAIL|SKIP)\b|^Error|^GpuTests: ' } | ForEach-Object { "  $_" }
    $detail = "exit code $($run.code)"
    Write-Check "GpuTests optix-runtime optix-tracer optix-render optix-triangles --require-optix" ($run.code -eq 0) `
        $detail

    # --- 3. the white furnaces under --optix: every value exactly 1.0 -------------------------------------------------
    foreach ($scene in @("furnace", "env-furnace", "mesh-furnace")) {
        $pfm = Join-Path $work "reverify-$scene.pfm"
        $run = Invoke-Queued "reverify-$scene" @($gpu, $scene, "--optix", "-p", "material", "-w", "480", "--height",
            "270", "-s", "64", "-o", $pfm)
        if ($run.code -ne 0) {
            Write-Check "$scene --optix -p material" $false "the render's exit code $($run.code)"
            continue
        }
        $result = Invoke-Native $stats @("stats", $pfm, "--equal", "1", "--share", "1")
        $result.lines | ForEach-Object { "  $_" }
        $detail = "PfmStats exit code $($result.code)"
        Write-Check "$scene --optix -p material, every value exactly 1.0" ($result.code -eq 0) $detail
    }

    # --- 4. the ImageCompare protocol against the quality kernel's references -----------------------------------------
    # compare.ps1's exit codes: 0 pass, 1 fail, 2 VOID (its "Protocol VOID" line) or an error, 4 refused by the queue
    foreach ($scene in @("cornell", "final", "studio")) {
        $b = "mis@optix:3,4"
        for ($attempt = 1; ; $attempt++) {
            $arguments = @{ Step = $Step; Scene = $scene; A = "mis@quality:ref"; B = $b; State = $State;
                Build = $Build }
            $lines = @(& $compareScript @arguments)
            $code = $LASTEXITCODE
            $lines | ForEach-Object { "  $_" }
            if ($code -eq 4) {
                Write-Output (("optix-reverify: gpu.ps1 refused a render of compare.ps1 {0} (its exit code 4): " +
                    "nothing re-verified") -f $scene)
                exit 2
            }
            $void = $code -eq 2 -and @($lines | Where-Object { $_ -match '^Protocol VOID' }).Count -gt 0
            if (-not $void -or $attempt -ge 2) { break }
            Write-Output "compare.ps1 ${scene}: VOID (a control failed): once more, B's seeds +10 (A8.3)"
            $b = "mis@optix:13,14"
        }
        if ($void) {
            Write-Output ""
            Write-Output (("Re-verification: VOID (compare.ps1 {0} void twice: its controls failed, so the " +
                "comparison can't be trusted on that scene): stop and ask the user; record nothing") -f $scene)
            if ($failures -gt 0) {
                Write-Output ("Re-verification: FAIL ({0} of {1} checks before it)" -f $failures, $checks)
                exit 1
            }
            exit 2
        }
        $detail = if ($code -eq 2) { "an error, exit code 2" } else { "exit code $code" }
        Write-Check "compare.ps1 $scene, mis@quality:ref against $b" ($code -eq 0) $detail
    }

    # --- 5. the checks later steps add (empty until Claude integrates that step) --------------------------------------
    # Step 91: the triangles (the suite optix-triangles in section 2, mesh-furnace in section 3), and its protocols:
    # cornell's quads under the quality kernel against cornell-triangles under OptiX (the triangles' hierarchies and the
    # origins moved off them under the new driver), and instances-baked against instances (the instance transforms)
    foreach ($pair in @(@("cornell", "mis@quality:ref", "cornell-triangles", "mis@optix:3,4"),
            @("instances-baked", "mis@optix:1,2", "instances", "mis@optix:3,4"))) {
        $arguments = @{ Step = $Step; Scene = $pair[0]; A = $pair[1]; SceneB = $pair[2]; B = $pair[3]; State = $State;
            Build = $Build }
        $lines = @(& $compareScript @arguments)
        $code = $LASTEXITCODE
        $lines | ForEach-Object { "  $_" }
        if ($code -eq 4) {
            Write-Output ("optix-reverify: gpu.ps1 refused a render of compare.ps1 {0}: nothing re-verified" -f $pair[2])
            exit 2
        }
        $detail = if ($code -eq 2) { "VOID or an error, exit code 2" } else { "exit code $code" }
        Write-Check ("compare.ps1 {0} {1} against {2} {3}" -f $pair[0], $pair[1], $pair[2], $pair[3]) ($code -eq 0) `
            $detail
    }

    # Step 92: pbr-test against pbr-reference, Step 92's protocol: compare.ps1 -Scene pbr-reference -A mis@optix:1,2
    #          -SceneB pbr-test -B mis@optix:3,4 (glTF's materials texel by texel against the same materials as the
    #          scene's own, both sides OptiX's).

    # Step 94: the denoiser's evaluation on cornell (denoise.ps1: the N16/D16 renders, PfmStats error, the thresholds),
    #          against its references State\ref\cornell-mis2048-s101.pfm and -s102.pfm: Step 85's renders, which a
    #          driver update doesn't change.

    Write-Output ""
    if ($failures -gt 0) {
        Write-Output (("Re-verification: FAIL ({0} of {1} checks): stop and ask the user (a driver change or a " +
            "latent bug); record nothing") -f $failures, $checks)
        exit 1
    }
    Write-Output ("Re-verification: PASS ({0} checks)" -f $checks)
    Write-Output (("Next: Claude stamps the stamped entries anew (carry.ps1 -Step {0} -Set optix -Restamp for the " +
        "entries whose bytes the step's check run found the ones recorded, -Record for the others; hostgate.ps1 " +
        "-Step {0} -Accept for the stamped chained sections), and the commit says ""re-verified after a graphics " +
        "driver update"", naming no version") -f $Step)
    exit 0
}
catch {
    Write-Output "Error: $($_.Exception.Message)"
    exit 2
}
