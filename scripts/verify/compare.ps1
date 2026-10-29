# The ImageCompare protocol: do two sampling strategies (or two kernels) converge to the same image? Both are rendered
# with two seeds each at -w 480 --height 270 (the milestone's size for statistics) and the same samples, and:
#   - controls: A1 against A2 and B1 against B2 must be consistent. A control compares a strategy with itself, so a
#     failing one says the comparison can't be trusted on this scene (it voids the protocol: exit code 2);
#   - tests: A1 against B1 and A2 against B2, pairs of independent renders (different seeds), must be consistent;
#   - the RMS rule (PfmStats rule): each test's trimmed RMS of d at most 1.10 times what the controls predict for two
#     strategies of unequal noise, √((R_A² + R_B²)/2): a pattern destroyed inside the tiles fails it;
#   - the noise of one render of each strategy, σ = RMS(d)/√2 of its control (PfmStats noise), is printed.
# The re-run rule, fixed before any result: one failing test (or the rule) renders both strategies again with seeds
# +10 and repeats all four comparisons and the rule, plus --tile 64 on the pair that failed; it passes if all of that
# passes. Two failures in the first run, or any in the re-run, fail the protocol.
#   compare.ps1 -Step 73 -Scene cornell -A material@books:ref -B material@quality:3,4 [-Spp 1024] [-ControlOnly]
# A side is STRATEGY[@KERNEL][:SEEDS]: KERNEL books (the default) or quality (--quality-kernel); SEEDS two numbers, or
# ref for the references rendered by Step 70's exes, out\m4\ref\<scene>-<strategy>-s1.pfm and -s2.pfm (seeds 1 and 2,
# 1024 samples). Renders go through gpu.ps1 into out\m4\runs\stepNN\. Exit code 0 pass, 1 fail, 2 void or an error.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][string]$Scene,
    [Parameter(Mandatory = $true)][string]$A,
    [Parameter(Mandatory = $true)][string]$B,
    [int]$Spp = 1024,
    [switch]$ControlOnly,
    [string]$Build = "out\build\x64-release"
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$gpu = Join-Path $project "$Build\RayTracingGPU.exe"
$compare = Join-Path $project "$Build\ImageCompare.exe"
$stats = Join-Path $project "$Build\PfmStats.exe"
$work = Join-Path $project ("out\m4\runs\step{0}" -f $Step)
$refs = Join-Path $project "out\m4\ref"
New-Item -ItemType Directory -Force $work | Out-Null
$size = @("-w", "480", "--height", "270")

function Read-Side([string]$text) {
    if ($text -cnotmatch '^([a-z]+)(@(books|quality))?(:(ref|\d+,\d+))?$') {
        throw "compare.ps1: a side is STRATEGY[@KERNEL][:SEEDS], not '$text'"
    }
    $side = @{ strategy = $Matches[1]; kernel = if ($Matches[3]) { $Matches[3] } else { "books" } }
    $seeds = if ($Matches[5]) { $Matches[5] } else { "ref" }
    $side.ref = $seeds -eq "ref"
    if ($side.ref -and ($side.kernel -ne "books" -or $Spp -ne 1024)) {
        throw "compare.ps1: the references are the books' kernels at 1024 samples, not '$text' at $Spp"
    }
    $side.seeds = if ($side.ref) { @(1, 2) } else { @($seeds.Split(",") | ForEach-Object { [int]$_ }) }
    $side.name = "{0}-{1}{2}" -f $Scene, $side.strategy, $(if ($side.kernel -eq "quality") { "-quality" } else { "" })
    return $side
}

# The PFM of one side and seed: a reference, or a render through the queue
function Get-Render($side, [int]$seed, [bool]$fresh) {
    if ($side.ref -and -not $fresh) {
        $path = Join-Path $refs ("{0}-{1}-s{2}.pfm" -f $Scene, $side.strategy, $seed)
        if (-not (Test-Path -LiteralPath $path)) { throw "compare.ps1: no reference $path" }
        return $path
    }
    $path = Join-Path $work ("{0}-{1}spp-s{2}.pfm" -f $side.name, $Spp, $seed)
    $arguments = @($Scene) + $size + @("-s", "$Spp", "--seed", "$seed", "-p", $side.strategy, "-o", $path)
    if ($side.kernel -eq "quality") { $arguments += "--quality-kernel" }
    $all = @("-Step", "$Step", "-Label", ("{0} s{1}" -f $side.name, $seed), "--", $gpu) + $arguments
    $lines = @(& $gpuScript @all)
    if ($LASTEXITCODE -ne 0) {
        $lines | ForEach-Object { Write-Host "  $_" }
        throw "compare.ps1: the render of $($side.name), seed $seed, failed (exit code $LASTEXITCODE)"
    }
    $lines | Where-Object { $_ -match '^(Kernel|Rays traced|Shadow rays|Mean radiance)' } |
        ForEach-Object { Write-Host ("  {0} s{1}: {2}" -f $side.name, $seed, $_) }
    return $path
}

# ImageCompare's verdict line and exit code
function Invoke-Compare([string]$what, [string]$first, [string]$second, [string[]]$extra = @()) {
    $out = @(& $compare $first $second @extra 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    $verdict = @($out | Where-Object { $_ -match '^Verdict' }) | Select-Object -Last 1
    $note = if (@($out | Where-Object { $_ -match 'different shapes' }).Count -gt 0) { " (the noise-shape note)" }
    else { "" }
    Write-Host ("  {0}: exit {1}{2}: {3}" -f $what, $code, $note, $verdict)
    return $code
}

function Invoke-Stats([string[]]$arguments) {
    $out = @(& $stats @arguments 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host "    $_" }
    return $code
}

# One round: the controls, the tests and the rule; the number of tests (and the rule) that failed, or -1 if void
function Invoke-Round($sideA, $sideB, [int]$offset, [bool]$fresh) {
    $a1 = Get-Render $sideA ($sideA.seeds[0] + $offset) $fresh
    $a2 = Get-Render $sideA ($sideA.seeds[1] + $offset) $fresh
    $b1 = Get-Render $sideB ($sideB.seeds[0] + $offset) $fresh
    $b2 = Get-Render $sideB ($sideB.seeds[1] + $offset) $fresh
    Write-Host ("Seeds {0}: A = {1}, B = {2}" -f $(if ($offset) { "+$offset" } else { "as given" }), $sideA.name,
        $sideB.name)
    $controls = 0
    $controls += [int]((Invoke-Compare "control A1-A2" $a1 $a2) -ne 0)
    if ($a1 -ne $b1) { $controls += [int]((Invoke-Compare "control B1-B2" $b1 $b2) -ne 0) }
    Write-Host "  noise of A:"
    Invoke-Stats @("noise", $a1, $a2) | Out-Null
    if ($a1 -ne $b1) {
        Write-Host "  noise of B:"
        Invoke-Stats @("noise", $b1, $b2) | Out-Null
    }
    if ($controls -gt 0) { return @{ failed = -1 } }
    if ($ControlOnly) { return @{ failed = 0 } }
    $t1 = Invoke-Compare "test A1-B1" $a1 $b1
    $t2 = Invoke-Compare "test A2-B2" $a2 $b2
    Write-Host "  the RMS rule:"
    $rule = Invoke-Stats @("rule", $a1, $b1, $a2, $b2, $a1, $a2, $b1, $b2)
    return @{ failed = [int]($t1 -ne 0) + [int]($t2 -ne 0) + [int]($rule -ne 0); t1 = $t1; t2 = $t2;
        pairs = @(@($a1, $b1), @($a2, $b2)) }
}

# An error (a bad side, a missing reference, a render that fails) is exit code 2, never a verdict
try {
    $sideA = Read-Side $A
    $sideB = Read-Side $B
    Write-Output ("compare {0}: A {1}, B {2}, {3} samples, 480x270" -f $Scene, $A, $B, $Spp)
    $first = Invoke-Round $sideA $sideB 0 $false
    if ($first.failed -lt 0) { Write-Output "Protocol VOID: a control failed"; exit 2 }
    if ($first.failed -eq 0) { Write-Output "Protocol: PASS"; exit 0 }
    if ($first.failed -ge 2) { Write-Output "Protocol: FAIL ($($first.failed) failures in the first run)"; exit 1 }
    Write-Output "One failure: the re-run with seeds +10"
    $second = Invoke-Round $sideA $sideB 10 $true
    if ($second.failed -lt 0) { Write-Output "Protocol VOID: a control failed in the re-run"; exit 2 }
    $tiled = 0
    $failedPair = if ($first.t1 -ne 0) { $first.pairs[0] } elseif ($first.t2 -ne 0) { $first.pairs[1] } else { $null }
    if ($failedPair) {
        $code = Invoke-Compare "the failed pair, --tile 64" $failedPair[0] $failedPair[1] @("--tile", "64")
        $tiled = [int]($code -ne 0)
    }
    if ($second.failed -eq 0 -and $tiled -eq 0) { Write-Output "Protocol: PASS (after the re-run)"; exit 0 }
    Write-Output "Protocol: FAIL (the re-run failed too)"
    exit 1
}
catch {
    Write-Output "Error: $($_.Exception.Message)"
    exit 2
}
