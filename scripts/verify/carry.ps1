# The carry check: configurations a step verified (statistically, against the references) still render the same bytes
# in later steps, so their verification carries forward without rendering the comparisons again. Each configuration is
# a RayTracingGPU command, rendered at -w 480 --height 270 -s 64 --seed 3 into a PFM, whose SHA-256 is kept with the
# step that verified it in out\m4\carry\set.txt (lines name|step|sha256|arguments). A difference names the
# configuration whose protocol the step must run again (or re-record, when the step changes it by design).
#   carry.ps1 -Step 74                                   check every configuration of the set (none: exit code 2)
#   carry.ps1 -Step 73 -Record cornell-base -Arguments "cornell -p material --quality-kernel"
#                                                        add a configuration (or record it again) as verified by Step 73
# Renders go through gpu.ps1. Exit code 0 when every hash is the same.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string]$Record = "",
    [string]$Arguments = "",
    [string]$Build = "out\build\x64-release"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$gpu = Join-Path $project "$Build\RayTracingGPU.exe"
$setFile = Join-Path $project "out\m4\carry\set.txt"
$work = Join-Path $project ("out\m4\runs\step{0}\carry" -f $Step)
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $setFile) | Out-Null

function Read-Set {
    if (-not (Test-Path -LiteralPath $setFile)) { return @() }
    return @(Get-Content -LiteralPath $setFile | Where-Object { $_.Trim() } | ForEach-Object {
            $p = $_.Split("|", 4)
            [pscustomobject]@{ name = $p[0]; step = [int]$p[1]; sha = $p[2]; arguments = $p[3] }
        })
}

function Get-CarryHash([string]$name, [string]$argumentText) {
    $path = Join-Path $work "$name.pfm"
    $list = @($argumentText -split '\s+' | Where-Object { $_ }) +
        @("-w", "480", "--height", "270", "-s", "64", "--seed", "3", "-o", $path)
    $all = @("-Step", "$Step", "-Label", "carry $name", "--", $gpu) + $list
    $lines = @(& $gpuScript @all)
    if ($LASTEXITCODE -ne 0) {
        $lines | ForEach-Object { Write-Host "  $_" }
        throw "carry.ps1: $name failed (exit code $LASTEXITCODE)"
    }
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}

$set = Read-Set
if ($Record) {
    if (-not $Arguments) {
        $known = $set | Where-Object { $_.name -eq $Record } | Select-Object -First 1
        if (-not $known) { throw "carry.ps1: -Record $Record needs -Arguments (a new configuration)" }
        $Arguments = $known.arguments
    }
    $sha = Get-CarryHash $Record $Arguments
    $set = @($set | Where-Object { $_.name -ne $Record }) +
        @([pscustomobject]@{ name = $Record; step = $Step; sha = $sha; arguments = $Arguments })
    [IO.File]::WriteAllLines($setFile, [string[]]@($set | ForEach-Object {
                "{0}|{1}|{2}|{3}" -f $_.name, $_.step, $_.sha, $_.arguments }))
    Write-Output ("carry: {0} recorded as Step {1}'s ({2})" -f $Record, $Step, $Arguments)
    exit 0
}

# A set with nothing in it proves nothing: an error, not a pass
if ($set.Count -eq 0) {
    Write-Output "carry: no configurations in $setFile (record them first)"
    exit 2
}
$different = 0
foreach ($entry in $set) {
    $sha = Get-CarryHash $entry.name $entry.arguments
    if ($sha -eq $entry.sha) {
        Write-Output ("carry {0} (Step {1}): identical" -f $entry.name, $entry.step)
    }
    else {
        Write-Output ("carry {0} (Step {1}): DIFFERENT: run its protocol again ({2})" -f $entry.name, $entry.step,
            $entry.arguments)
        $different++
    }
}
Write-Output ("carry: {0} configurations, {1} different" -f $set.Count, $different)
if ($different -gt 0) { exit 1 }
exit 0
