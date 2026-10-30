# The carry check: configurations a step verified (statistically, against the references) still render the same bytes
# in later steps, so their verification carries forward without rendering the comparisons again. Each configuration is
# a RayTracingGPU command, rendered at -w 480 --height 270 -s 64 --seed 3 into a PFM, whose SHA-256 is kept with the
# step that verified it; with --tonemap the PNG's too, and with --denoise the -denoised PFM's and PNG's, the hashes
# joined by colons. Two sets, kept in -State (out\m5, never committed):
#   - cuda (the default): carry\set.txt, lines name|step|sha256|arguments. The CUDA kernels' machine code is in the
#     exes, so these bytes must stay the same at every step. When the file doesn't exist yet, it is seeded once from
#     milestone 4's set (out\m4\carry\set.txt: its configurations keep the steps that verified them).
#   - optix: carry\optix.txt, lines name|step|sha256|arguments|driver|programs. OptiX's programs are compiled by the
#     graphics driver, so each entry is stamped with the driver's version (nvidia-smi) and the programs' SHA-256 (the
#     first word of <Build>\optix\optix_programs.sha256, 64 hex digits, or none). With the same stamp a different hash
#     is DIFFERENT, as in the CUDA set. With another driver or other programs the entry says RE-VERIFY and prints this
#     run's hash, neither a pass nor a failure: its protocol is run again and -Record stamps it anew, or, when this
#     step's check run rendered the bytes recorded, -Restamp stamps it anew without rendering. A stamp that can't be
#     read is an error (exit code 2), never a stamp of its own, which would turn a real difference under the same
#     driver and programs into RE-VERIFY; nothing is rendered or recorded then: nvidia-smi failing or printing no
#     version, an entry stamped with OptiX programs while the build has no optix_programs.sha256, an entry whose stamp
#     isn't <driver version>|<64 hex digits or none>.
# Every check run keeps what it rendered in State\runs\stepNN\carry\hashes-<set>.txt: RayTracingGPU.exe's SHA-256, the
# stamp (OptiX set), and each configuration's hash; -Restamp reads it.
# A difference names the configuration whose protocol the step must run again (or re-record, when the step changes it
# by design).
#   carry.ps1 -Step 86 [-Set optix]                      check every configuration of the set (none: exit code 2)
#   carry.ps1 -Step 73 -Record cornell-base -Arguments "cornell -p material --quality-kernel"
#                                                        add a configuration (or record it again) as verified by Step 73
#   carry.ps1 -Step 89 -Set optix -Record cornell-mis -Arguments "cornell --optix -p mis"
#                                                        the same in the OptiX set, with the stamp of now
#   carry.ps1 -Step 91 -Set optix -Restamp cornell-mis   (or -Restamp all) the stamp of now on entries whose hash in
#                                                        Step 91's check run is the one recorded, rendering nothing; the
#                                                        verifying step stays. Refused without that run (exit code 2),
#                                                        when RayTracingGPU.exe or the stamp changed since it (2), and
#                                                        for an entry it rendered other bytes of, or didn't render (1)
# Renders go through gpu.ps1. Exit code 0 when every hash is the same, 1 when one differs, 2 for an empty set or an
# error (a stamp that can't be read, a render that fails, a refusal of the queue), 3 when none differs but an OptiX
# entry is to be re-verified.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [ValidateSet("cuda", "optix")][string]$Set = "cuda",
    [string]$Record = "",
    [string]$Arguments = "",
    [string]$Restamp = "",
    [string]$Build = "out\build\x64-release",
    [string]$State = "out\m5"
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$buildDir = if ([IO.Path]::IsPathRooted($Build)) { $Build } else { Join-Path $project $Build }
$stateDir = if ([IO.Path]::IsPathRooted($State)) { $State } else { Join-Path $project $State }
$gpu = Join-Path $buildDir "RayTracingGPU.exe"
$optix = $Set -eq "optix"
$carryDir = Join-Path $stateDir "carry"
$setFile = Join-Path $carryDir $(if ($optix) { "optix.txt" } else { "set.txt" })
$work = Join-Path $stateDir ("runs\step{0}\carry" -f $Step)
$hashesFile = Join-Path $work "hashes-$Set.txt"

# The entries of the set. An OptiX line is split from both ends, so arguments holding a | stay whole; its stamp must be
# one that was read (never "unknown")
function Read-Entries {
    if (-not (Test-Path -LiteralPath $setFile)) { return @() }
    return @(Get-Content -LiteralPath $setFile | Where-Object { $_.Trim() } | ForEach-Object {
            if ($optix) {
                $p = $_.Split("|")
                $n = $p.Count
                if ($n -lt 6) { throw "carry.ps1: '$_' in $setFile isn't name|step|sha256|arguments|driver|programs" }
                if ($p[$n - 2] -cnotmatch '^\d+(\.\d+)+$' -or $p[$n - 1] -cnotmatch '^([0-9a-f]{64}|none)$') {
                    throw "carry.ps1: '$_' in $setFile doesn't end with a stamp <driver version>|<programs or none>"
                }
                [pscustomobject]@{ name = $p[0]; step = [int]$p[1]; sha = $p[2]; arguments = $p[3..($n - 3)] -join "|";
                    driver = $p[$n - 2]; programs = $p[$n - 1] }
            }
            else {
                $p = $_.Split("|", 4)
                [pscustomobject]@{ name = $p[0]; step = [int]$p[1]; sha = $p[2]; arguments = $p[3] }
            }
        })
}

# The stamp of OptiX's output now: the graphics driver's version and the OptiX programs' SHA-256 in this build ("none"
# before OptiX is in the build). nvidia-smi's whole output is taken before its exit code is read: piped into
# Select-Object -First, the program is stopped before PowerShell stores its exit code, which leaves the previous
# command's. A driver that can't be read is an error, never a stamp of its own
function Get-Stamp {
    $smi = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $smi) { throw "carry.ps1: no nvidia-smi on the PATH: the driver's version, the stamp's half, is unread" }
    $ErrorActionPreference = "Continue" # nvidia-smi's standard error must not stop the script
    $out = @(& $smi.Path --query-gpu=driver_version --format=csv,noheader 2>$null)
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $driver = if ($out.Count -gt 0) { ([string]$out[0]).Trim() } else { "" }
    if ($code -ne 0 -or $driver -notmatch '^\d+(\.\d+)+$') {
        throw (("carry.ps1: nvidia-smi printed no driver version (exit code {0}): a stamp that can't be read is an " +
            "error, never a stamp of its own") -f $code)
    }
    $programs = "none"
    $sums = Join-Path $buildDir "optix\optix_programs.sha256"
    if (Test-Path -LiteralPath $sums) {
        $first = @(([IO.File]::ReadAllText($sums)).Trim() -split '\s+')[0]
        if ($first -notmatch '^[0-9A-Fa-f]{64}$') { throw "carry.ps1: $sums doesn't start with a SHA-256" }
        $programs = $first.ToLowerInvariant()
    }
    return [pscustomobject]@{ driver = $driver; programs = $programs }
}

# A configuration with --tonemap also saves its PNG, the display's bytes (Step 74), and one with --denoise each output's
# -denoised twin: its hash is the PFM's, the PNG's, the denoised PFM's and the denoised PNG's (those it has), joined by
# colons. The files are deleted first, so a file the render didn't write can't be hashed from an earlier run.
function Get-CarryHash([string]$name, [string]$argumentText) {
    $path = Join-Path $work "$name.pfm"
    $png = Join-Path $work "$name.png"
    $words = @($argumentText -split '\s+' | Where-Object { $_ })
    $toned = $words -contains "--tonemap"
    $list = $words + @("-w", "480", "--height", "270", "-s", "64", "--seed", "3", "-o", $path)
    if ($toned) { $list += @("-o", $png) }
    $files = @($path)
    if ($toned) { $files += $png }
    if ($words -contains "--denoise") {
        $files += Join-Path $work "$name-denoised.pfm"
        if ($toned) { $files += Join-Path $work "$name-denoised.png" }
    }
    foreach ($file in $files) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
    }
    $label = if ($optix) { "carry-optix $name" } else { "carry $name" }
    $all = @("-Step", "$Step", "-Label", $label, "--", $gpu) + $list
    $lines = @(& $gpuScript @all)
    if ($LASTEXITCODE -ne 0) {
        $lines | ForEach-Object { Write-Host "  $_" }
        throw "carry.ps1: $name failed (exit code $LASTEXITCODE)"
    }
    return (@($files | ForEach-Object { (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }) -join ":")
}

function Format-Entry($entry) {
    if ($optix) {
        return "{0}|{1}|{2}|{3}|{4}|{5}" -f $entry.name, $entry.step, $entry.sha, $entry.arguments, $entry.driver,
            $entry.programs
    }
    return "{0}|{1}|{2}|{3}" -f $entry.name, $entry.step, $entry.sha, $entry.arguments
}

function Write-Entries($entries) {
    [IO.File]::WriteAllLines($setFile, [string[]]@($entries | ForEach-Object { Format-Entry $_ }))
}

# This step's check run of the set, as hashes-<set>.txt keeps it: @{ exe; stamp; hashes = name -> @{ sha; arguments } }
function Read-CheckRun {
    $result = @{ exe = ""; stamp = ""; hashes = @{} }
    foreach ($line in [IO.File]::ReadAllLines($hashesFile)) {
        if ($line -cmatch '^exe ([0-9A-F]{64})$') { $result.exe = $Matches[1] }
        elseif ($line -cmatch '^stamp (\S+)$') { $result.stamp = $Matches[1] }
        elseif ($line.Trim()) {
            $q = $line.Split("|", 3)
            if ($q.Count -ne 3) { throw "carry.ps1: '$line' in $hashesFile isn't name|sha256|arguments" }
            $result.hashes[$q[0]] = @{ sha = $q[1]; arguments = $q[2] }
        }
    }
    if (-not $result.exe) { throw "carry.ps1: $hashesFile names no exe" }
    return $result
}

# Any error, a stamp that can't be read first among them, is exit code 2: nothing is recorded
try {
    New-Item -ItemType Directory -Force $work | Out-Null
    New-Item -ItemType Directory -Force $carryDir | Out-Null

    # The CUDA set starts as milestone 4's, once
    $cudaSet = Join-Path $carryDir "set.txt"
    $m4Set = Join-Path $project "out\m4\carry\set.txt"
    if (-not (Test-Path -LiteralPath $cudaSet) -and (Test-Path -LiteralPath $m4Set)) {
        Copy-Item -LiteralPath $m4Set $cudaSet
        $seeded = @(Get-Content -LiteralPath $cudaSet | Where-Object { $_.Trim() }).Count
        Write-Output ("carry: the CUDA set seeded from out\m4\carry\set.txt ({0} configurations, their steps kept)" -f
            $seeded)
    }

    # @(): a set of one entry is still an array (a lone PSCustomObject has no Count in Windows PowerShell 5.1)
    $entries = @(Read-Entries)
    $stamp = $null
    if ($optix) {
        $stamp = Get-Stamp
        $kept = @($entries | Where-Object { $_.programs -ne "none" })
        if ($stamp.programs -eq "none" -and $kept.Count -gt 0) {
            throw (("carry.ps1: {0} entries are stamped with OptiX programs ({1} first), but this build has no " +
                "optix\optix_programs.sha256: the programs' half of the stamp can't be read") -f $kept.Count,
                $kept[0].name)
        }
    }
    if ($Record -and $Restamp) { throw "carry.ps1: -Record or -Restamp, not both" }
    if ($Record -eq "all") { throw "carry.ps1: 'all' is -Restamp's word for every entry, not a configuration's name" }

    if ($Record) {
        if (-not $Arguments) {
            $known = $entries | Where-Object { $_.name -eq $Record } | Select-Object -First 1
            if (-not $known) { throw "carry.ps1: -Record $Record needs -Arguments (a new configuration)" }
            $Arguments = $known.arguments
        }
        $sha = Get-CarryHash $Record $Arguments
        $entry = [pscustomobject]@{ name = $Record; step = $Step; sha = $sha; arguments = $Arguments }
        if ($optix) {
            $entry | Add-Member -NotePropertyName driver -NotePropertyValue $stamp.driver
            $entry | Add-Member -NotePropertyName programs -NotePropertyValue $stamp.programs
        }
        $entries = @($entries | Where-Object { $_.name -ne $Record }) + @($entry)
        Write-Entries $entries
        Write-Output ("carry: {0} recorded as Step {1}'s ({2}){3}" -f $Record, $Step, $Arguments,
            $(if ($optix) { ", stamped with this driver and these OptiX programs" } else { "" }))
        exit 0
    }

    if ($Restamp) {
        # Only bytes this step's check run rendered, with this exe under this stamp, are re-stamped
        if (-not $optix) { throw "carry.ps1: -Restamp is for -Set optix: the CUDA entries carry no stamp" }
        if (-not (Test-Path -LiteralPath $hashesFile)) {
            Write-Output (("carry: -Restamp refused: no check run of the OptiX set in Step {0} ({1}): run carry.ps1 " +
                "-Step {0} -Set optix first") -f $Step, $hashesFile)
            exit 2
        }
        $checkRun = Read-CheckRun
        if ($checkRun.exe -cne (Get-FileHash -LiteralPath $gpu -Algorithm SHA256).Hash) {
            Write-Output (("carry: -Restamp refused: RayTracingGPU.exe changed since Step {0}'s check run: check " +
                "the set again") -f $Step)
            exit 2
        }
        if ($checkRun.stamp -cne ("{0}|{1}" -f $stamp.driver, $stamp.programs)) {
            Write-Output (("carry: -Restamp refused: the driver or the OptiX programs changed since Step {0}'s " +
                "check run: check the set again") -f $Step)
            exit 2
        }
        if ($Restamp -ne "all" -and @($entries | Where-Object { $_.name -eq $Restamp }).Count -eq 0) {
            throw "carry.ps1: no entry $Restamp in $setFile"
        }
        $restamped = 0
        $refused = 0
        foreach ($entry in $entries) {
            if ($Restamp -ne "all" -and $entry.name -ne $Restamp) { continue }
            $seen = $checkRun.hashes[$entry.name]
            if (-not $seen) {
                Write-Output ("carry {0}: -Restamp refused: Step {1}'s check run didn't render it" -f $entry.name,
                    $Step)
                $refused++
            }
            elseif ($seen.arguments -cne $entry.arguments) {
                Write-Output ("carry {0}: -Restamp refused: Step {1}'s check run rendered other arguments ({2})" -f
                    $entry.name, $Step, $seen.arguments)
                $refused++
            }
            elseif ($seen.sha -ne $entry.sha) {
                Write-Output (("carry {0}: -Restamp refused: Step {1}'s check run rendered other bytes: run its " +
                    "protocol again, then -Record") -f $entry.name, $Step)
                $refused++
            }
            elseif ($entry.driver -ceq $stamp.driver -and $entry.programs -ceq $stamp.programs) {
                Write-Output ("carry {0} (Step {1}): already stamped with this driver and these programs" -f
                    $entry.name, $entry.step)
            }
            else {
                $entry.driver = $stamp.driver
                $entry.programs = $stamp.programs
                Write-Output (("carry {0} (Step {1}): re-stamped with this driver and these OptiX programs, its " +
                    "bytes those Step {2}'s check run rendered") -f $entry.name, $entry.step, $Step)
                $restamped++
            }
        }
        if ($restamped -gt 0) { Write-Entries $entries }
        Write-Output ("carry: {0} OptiX configurations re-stamped, {1} refused" -f $restamped, $refused)
        if ($refused -gt 0) { exit 1 }
        exit 0
    }

    # A set with nothing in it proves nothing: an error, not a pass
    if ($entries.Count -eq 0) {
        Write-Output "carry: no configurations in $setFile (record them first)"
        exit 2
    }
    # What this run renders is kept for -Restamp, whole runs only: an earlier run's file is gone before the first render
    if (Test-Path -LiteralPath $hashesFile) { Remove-Item -LiteralPath $hashesFile -Force }
    $seenLines = @("exe " + (Get-FileHash -LiteralPath $gpu -Algorithm SHA256).Hash)
    if ($optix) { $seenLines += ("stamp {0}|{1}" -f $stamp.driver, $stamp.programs) }
    $different = 0
    $reverify = 0
    foreach ($entry in $entries) {
        $sha = Get-CarryHash $entry.name $entry.arguments
        $seenLines += ("{0}|{1}|{2}" -f $entry.name, $sha, $entry.arguments)
        $changed = @()
        if ($optix) {
            if ($entry.driver -cne $stamp.driver) { $changed += "the driver" }
            if ($entry.programs -cne $stamp.programs) { $changed += "the OptiX programs" }
        }
        if ($changed.Count -gt 0) {
            # Another stamp: the driver compiles the programs, so the bytes may differ without a fault. Neither a pass
            # nor a failure; the hash is printed, so the step can tell whether the bytes changed.
            Write-Output ("carry {0} (Step {1}): RE-VERIFY {0} ({2} changed): this run's hash {3} ({4})" -f
                $entry.name, $entry.step, ($changed -join " and "), $sha,
                $(if ($sha -eq $entry.sha) { "the bytes recorded: -Restamp" } else { "not the bytes recorded" }))
            $reverify++
        }
        elseif ($sha -eq $entry.sha) {
            Write-Output ("carry {0} (Step {1}): identical" -f $entry.name, $entry.step)
        }
        else {
            Write-Output ("carry {0} (Step {1}): DIFFERENT: run its protocol again ({2})" -f $entry.name, $entry.step,
                $entry.arguments)
            $different++
        }
    }
    [IO.File]::WriteAllLines($hashesFile, [string[]]$seenLines)
    if ($optix) {
        Write-Output ("carry: {0} OptiX configurations, {1} different, {2} to re-verify" -f $entries.Count, $different,
            $reverify)
    }
    else { Write-Output ("carry: {0} configurations, {1} different" -f $entries.Count, $different) }
    if ($different -gt 0) { exit 1 }
    if ($reverify -gt 0) { exit 3 }
    exit 0
}
catch {
    Write-Output "carry: error: $($_.Exception.Message)"
    exit 2
}
