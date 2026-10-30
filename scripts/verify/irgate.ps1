# The IR gate: the OptiX programs are PTX text compiled into the exes, which the graphics driver compiles again when a
# program starts OptiX. The SASS gate can't see them (the exes hold no machine code of theirs), so this gate follows
# the PTX itself; nothing runs on the GPU. Its inputs, from Step 87 on: Build\optix\optix_programs.ptx (the render
# module, embedded in RayTracingGPU) and Build\optix\optix_check_ptx.ptx (GpuTests' own module: the probes and the
# tracer check), each with the .sha256 beside it that cmake\embed.cmake writes, and the folder of OptiX's headers the
# build used (RAYTRACING_OPTIX_INCLUDE_DIR in Build\CMakeCache.txt). It checks:
#   1. each .sha256 is its PTX's SHA-256, and the same 64 hex digits are in every exe that embeds it: optix_programs in
#      GpuTests.exe from Step 87, RayTracingGPU.exe from Step 89 (--optix) and RayTracingViewer.exe from Step 95 (the
#      window's OptiX), optix_check_ptx in GpuTests.exe from Step 87. So the PTX checked here is the one every program
#      carries, also after a build of one target only: a viewer left with older PTX fails, and so does an exe left from
#      before it linked the module (no module's hash at all, from its step on). Before that step an exe that holds no
#      module's hash is a note: a static library's object is linked into an exe only when the exe calls it. A build
#      without RayTracingViewer.exe (no SDL3) is a note; a missing RayTracingGPU.exe or GpuTests.exe is an error;
#   2. against the newest earlier record, State\ir\stepK.txt (K < NN): -Expect same fails on any change; -Expect
#      changed writes the cleaned diff of what changed to State\ir\stepNN.diff for review (comment lines dropped: nvcc
#      writes its version there; the two hashes of the source file's path in an unnamed namespace's mangled name,
#      _GLOBAL__N__<8 hex>_..._cu_<8 hex>, replaced by x's as sassgate.sh does, since a worktree's path differs; so is
#      the one hash of an internal-linkage variable's name, _INTERNAL_<8 hex>_..._cu_, which names the same path:
#      programs.cu's copy of ggx_energy_table is _INTERNAL_c8a20deb_11_programs_cu_ here and _INTERNAL_48507533_... in
#      a worktree, so the PTX's own SHA-256 differs from checkout to checkout while its cleaned text doesn't) and
#      saves both PTX files in State\ir\stepNN\, which the next diff starts from. Either way State\ir\stepNN.txt then
#      records the two hashes (a failing run records nothing; -Expect same that fails still writes the diff);
#   3. no file under src\ or tests\ has the name of a file in OptiX's include folder (without case, as Windows opens
#      files): the project's include folders come before OptiX's, so a src\gpu\optix.h would replace NVIDIA's optix.h
#      wherever it is included, inside OptiX's own headers too;
#   4. neither PTX has a .file or .loc line (no -lineinfo, no -G), a path with a drive letter or a user's folder (in
#      its text or in a string constant's bytes): the PTX names no file of the machine that built it, so Debug's can
#      equal Release's;
#   5. with -Debug DIR (a Debug build folder, Step 96): DIR\optix\*.ptx are Release's byte for byte, which proves every
#      OptiX program of the Debug build Release's without running one.
# A build without OptiX programs (Steps 86 and before, or one configured with -DRAYTRACING_OPTIX=OFF) prints "no OptiX
# programs in this build" and passes; when an earlier step recorded them and the build wasn't configured without
# OptiX, they went missing: a failure.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\irgate.ps1 -Step 88 -Expect changed
#   powershell -ExecutionPolicy Bypass -File scripts\verify\irgate.ps1 -Step 96 -Expect same -Debug out\build\x64-debug
# -Build is out\build\x64-release and -State out\m5 (the milestone's machine state) unless given. The last line, for the
# commit: "OptiX programs: PTX 3f2a9c1e (changed), check module 0b1c2d3e (unchanged)".
# Exit code 0 pass, 1 fail, 2 an error (a missing input, a bad option).
# (No [Parameter()] attributes: they would add PowerShell's common parameters, whose -Debug would clash with this one.)
param(
    [int]$Step = -1,
    [string]$Expect = "",
    [string]$Build = "out\build\x64-release",
    [string]$State = "out\m5",
    [string]$Debug = ""
)
$ErrorActionPreference = "Stop"
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
function Resolve-Folder([string]$path) {
    if ([IO.Path]::IsPathRooted($path)) { return $path }
    return Join-Path $project $path
}

# The two modules: the PTX's name in Build\optix, the exes that embed it, and the name the last line gives it
$modules = @(
    @{ name = "optix_programs"; exes = @("RayTracingGPU.exe", "RayTracingViewer.exe", "GpuTests.exe"); what = "PTX" },
    @{ name = "optix_check_ptx"; exes = @("GpuTests.exe"); what = "check module" })
# The exe a build may lack: the viewer is built only with SDL3
$optionalExes = @("RayTracingViewer.exe")
# The step from which each exe must embed its modules: GpuTests calls OptiX from Step 87 (the probes), RayTracingGPU
# from Step 89 (--optix), the viewer from Step 95 (OptiX in the window)
$embedsFrom = @{ "GpuTests.exe" = 87; "RayTracingGPU.exe" = 89; "RayTracingViewer.exe" = 95 }
# sassgate.sh's normalization of an unnamed namespace: eight x's for eight hex digits
$globalN = '_GLOBAL__N__[0-9a-f]{8}_([0-9]+_[A-Za-z0-9_]+_cu_)[0-9a-f]{8}'
$globalX = '_GLOBAL__N__xxxxxxxx_${1}xxxxxxxx'
$internalN = '_INTERNAL_[0-9a-f]{8}_([0-9]+_[A-Za-z0-9_]+_cu_)'
$internalX = '_INTERNAL_xxxxxxxx_${1}'
$failures = 0
$temp = $null

function Fail([string]$text) {
    Write-Output "FAIL: $text"
    $script:failures++
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# The newest record of an earlier step, State\ir\stepK.txt with K < Step (lines "name sha256"), or $null
function Get-EarlierRecord {
    if (-not (Test-Path -LiteralPath $ir)) { return $null }
    $best = $null
    foreach ($file in @(Get-ChildItem -LiteralPath $ir -File)) {
        if ($file.Name -notmatch '^step(\d+)\.txt$') { continue }
        $k = [int]$Matches[1]
        if ($k -lt $Step -and (-not $best -or $k -gt $best.step)) { $best = @{ step = $k; path = $file.FullName } }
    }
    if (-not $best) { return $null }
    $best.hashes = @{}
    foreach ($line in [IO.File]::ReadAllLines($best.path)) {
        if ($line -match '^(\S+) ([0-9a-f]{64})$') { $best.hashes[$Matches[1]] = $Matches[2] }
    }
    return $best
}

# The saved copy of a module's PTX with the given SHA-256: the newest State\ir\stepJ\NAME.ptx (J < Step) that has it
# (a step that expected no change saved none, its PTX being the one saved before)
function Find-SavedPtx([string]$name, [string]$sha) {
    if (-not (Test-Path -LiteralPath $ir)) { return $null }
    $folders = @(Get-ChildItem -LiteralPath $ir -Directory | Where-Object {
            $_.Name -match '^step(\d+)$' -and [int]$Matches[1] -lt $Step } |
        Sort-Object { [int]$_.Name.Substring(4) } -Descending)
    foreach ($folder in $folders) {
        $path = Join-Path $folder.FullName "$name.ptx"
        if ((Test-Path -LiteralPath $path) -and (Get-Sha256 $path) -eq $sha) { return $path }
    }
    return $null
}

# A PTX without its comment lines and with the unnamed namespaces' hashes replaced (no source: an empty file)
function Write-CleanPtx([string]$source, [string]$target) {
    $kept = New-Object 'System.Collections.Generic.List[string]'
    if ($source) {
        foreach ($line in [IO.File]::ReadAllLines($source)) {
            if ($line -match '^\s*//') { continue }
            $kept.Add(($line -creplace $globalN, $globalX -creplace $internalN, $internalX))
        }
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $target) | Out-Null
    [IO.File]::WriteAllLines($target, $kept)
}

# git's unified diff of the two cleaned PTX files, named by their steps
function Get-CleanDiff([string]$name, [string]$before, [string]$beforeLabel, [string]$after) {
    $a = "$beforeLabel/$name.ptx"
    $b = "step$Step/$name.ptx"
    Write-CleanPtx $before (Join-Path $temp $a)
    Write-CleanPtx $after (Join-Path $temp $b)
    $ErrorActionPreference = "Continue" # git's stderr must not stop the script
    $lines = @(& git -C $temp diff --no-index --no-color --no-ext-diff --text -U3 -- $a $b)
    if ($LASTEXITCODE -gt 1) { throw "irgate: git diff failed on $name (exit code $LASTEXITCODE)" }
    return $lines
}

try {
    if ($Step -lt 0 -or ($Expect -ne "same" -and $Expect -ne "changed")) {
        throw "usage: irgate.ps1 -Step NN -Expect same|changed [-Build DIR] [-State DIR] [-Debug DIR]"
    }
    $buildDir = Resolve-Folder $Build
    $optixDir = Join-Path $buildDir "optix"
    $cache = Join-Path $buildDir "CMakeCache.txt"
    $ir = Join-Path (Resolve-Folder $State) "ir"
    $earlier = Get-EarlierRecord

    # The inputs: both PTX files and both hashes, or none of them
    $inputs = @()
    foreach ($m in $modules) {
        $inputs += @((Join-Path $optixDir "$($m.name).ptx"), (Join-Path $optixDir "$($m.name).sha256"))
    }
    $present = @($inputs | Where-Object { Test-Path -LiteralPath $_ })
    if ($present.Count -eq 0) {
        $off = $false
        if (Test-Path -LiteralPath $cache) {
            $off = @(Get-Content -LiteralPath $cache |
                Where-Object { $_ -match '^RAYTRACING_OPTIX:BOOL=(OFF|FALSE|NO|N|0)$' }).Count -gt 0
        }
        if ($earlier -and -not $off) {
            Write-Output "no OptiX programs in this build, though Step $($earlier.step) recorded them"
            Write-Output "IR gate: FAIL (1)"
            exit 1
        }
        Write-Output "no OptiX programs in this build"
        exit 0
    }
    if ($present.Count -ne $inputs.Count) {
        $missing = @($inputs | Where-Object { $present -notcontains $_ })
        throw "irgate: Build\optix is incomplete: no $($missing -join ', ')"
    }

    # --- 1. the hashes, and the hash strings in the exes --------------------------------------------------------------
    $current = @{}
    $written = @{}
    foreach ($m in $modules) {
        $text = [IO.File]::ReadAllText((Join-Path $optixDir "$($m.name).sha256"))
        if ($text -notmatch '[0-9A-Fa-f]{64}') { throw "irgate: $($m.name).sha256 holds no SHA-256" }
        $written[$m.name] = $Matches[0].ToLowerInvariant()
        $actual = Get-Sha256 (Join-Path $optixDir "$($m.name).ptx")
        $current[$m.name] = $actual
        if ($written[$m.name] -ne $actual) {
            Fail "$($m.name).sha256 says $($written[$m.name]), the PTX's SHA-256 is $actual"
        }
        else { Write-Output "$($m.name): SHA-256 $actual, in its .sha256" }
    }
    # Each exe read once, for the hashes of every module it embeds
    foreach ($exeName in @($modules | ForEach-Object { $_.exes } | Select-Object -Unique)) {
        $exe = Join-Path $buildDir $exeName
        if (-not (Test-Path -LiteralPath $exe)) {
            if ($optionalExes -notcontains $exeName) { throw "irgate: no $exe" }
            Write-Output "note: no $exeName in this build (built without SDL3): its embedded PTX not checked"
            continue
        }
        # The exe's bytes as Latin-1 text, one character per byte: the string is ASCII wherever the linker put it
        $bytes = [Text.Encoding]::GetEncoding(28591).GetString([IO.File]::ReadAllBytes($exe))
        # Every embedded module's hash is a 64-hex string ending in a NUL (embed.cmake's NAME_sha256[]). A static
        # library's object is linked into an exe only when the exe uses it, so an exe that calls no OptiX yet (before
        # its step in $embedsFrom) holds none: a note. From its step on, an exe that holds none is one left from an
        # earlier step, and another module's hash where this one's should be is a stale exe: both are failures (a
        # partial build).
        $held = @([regex]::Matches($bytes, '[0-9a-f]{64}\x00') | ForEach-Object { $_.Value.Substring(0, 64) } |
            Select-Object -Unique)
        $from = $embedsFrom[$exeName]
        foreach ($m in @($modules | Where-Object { $_.exes -contains $exeName })) {
            $hash = $written[$m.name]
            if ($bytes.IndexOf($hash, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                Write-Output "$($m.name): its SHA-256 in $exeName"
            }
            elseif ($held.Count -eq 0 -and $Step -lt $from) {
                Write-Output ("note: $exeName embeds no OptiX module (it calls none before Step ${from}): " +
                    "$($m.name) not linked in")
            }
            elseif ($held.Count -eq 0) {
                Fail ("$exeName holds no OptiX module, though it embeds $($m.name) from Step $from on: an exe left " +
                    "from an earlier step (a partial build?)")
            }
            else {
                Fail ("$exeName doesn't hold $($m.name)'s SHA-256 ${hash}, but {0}: it embeds other PTX (a partial " +
                    "build?)" -f ($held -join ", "))
            }
        }
        $bytes = $null
    }

    # --- 2. against the earlier record --------------------------------------------------------------------------------
    if ($Expect -eq "same" -and -not $earlier) {
        throw "irgate: no earlier record in $ir to compare with (the first step with OptiX programs is -Expect changed)"
    }
    $changed = @{}
    foreach ($m in $modules) {
        $before = if ($earlier) { $earlier.hashes[$m.name] } else { $null }
        $changed[$m.name] = $before -ne $current[$m.name]
    }
    $anyChanged = @($modules | Where-Object { $changed[$_.name] }).Count -gt 0
    $since = if ($earlier) { "since Step $($earlier.step)" } else { "(no earlier record)" }
    if ($anyChanged -or $Expect -eq "changed") {
        $temp = Join-Path ([IO.Path]::GetTempPath()) ("irgate-" + [guid]::NewGuid().ToString("N"))
        $diff = @()
        foreach ($m in $modules) {
            $name = $m.name
            if (-not $changed[$name]) {
                $diff += "$name.ptx: unchanged $since ($($current[$name]))"
                Write-Output "$name.ptx: unchanged $since"
                continue
            }
            $before = $null
            $label = "none"
            if ($earlier) {
                $before = Find-SavedPtx $name $earlier.hashes[$name]
                $label = "step$($earlier.step)"
                if (-not $before) {
                    $diff += "$name.ptx: changed $since, but no copy of Step $($earlier.step)'s PTX is saved: no diff"
                    Write-Output "note: $name.ptx changed $since, but no copy of the earlier PTX is in ${ir}: no diff"
                    continue
                }
            }
            $lines = @(Get-CleanDiff $name $before $label (Join-Path $optixDir "$name.ptx"))
            $count = @($lines | Where-Object { $_ -match '^[-+]' -and $_ -notmatch '^(\+\+\+|---) ' }).Count
            $diff += "$name.ptx: changed $since ($($current[$name])), $count lines of the cleaned PTX differ"
            $diff += $lines
            if ($count -eq 0) {
                Write-Output "$name.ptx: changed $since, but not its cleaned text (comment lines or namespace hashes)"
            }
            else { Write-Output "$name.ptx: changed $since, $count lines of the cleaned PTX differ" }
        }
        New-Item -ItemType Directory -Force $ir | Out-Null
        $diffPath = Join-Path $ir "step$Step.diff"
        [IO.File]::WriteAllLines($diffPath, [string[]]$diff)
        Write-Output "the cleaned diff: $diffPath"
    }
    if ($Expect -eq "same" -and $anyChanged) { Fail "-Expect same, but the OptiX programs changed $since" }
    if ($Expect -eq "changed" -and -not $anyChanged) {
        Write-Output "note: -Expect changed, but neither PTX changed $since"
    }

    # --- 3. no project file named like one of OptiX's headers ---------------------------------------------------------
    if (-not (Test-Path -LiteralPath $cache)) { throw "irgate: no $cache" }
    $entry = @(Get-Content -LiteralPath $cache |
        Where-Object { $_ -match '^RAYTRACING_OPTIX_INCLUDE_DIR(:[A-Za-z]+)?=' }) | Select-Object -First 1
    if (-not $entry) { throw "irgate: $cache has no RAYTRACING_OPTIX_INCLUDE_DIR" }
    $include = ($entry -replace '^[^=]*=', '').Trim().Replace("/", "\")
    if (-not $include -or -not (Test-Path -LiteralPath $include -PathType Container)) {
        throw "irgate: OptiX's include folder '$include' doesn't exist"
    }
    $optixNames = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($file in @(Get-ChildItem -LiteralPath $include -Recurse -File)) { [void]$optixNames.Add($file.Name) }
    $ours = @(Get-ChildItem -LiteralPath (Join-Path $project "src"), (Join-Path $project "tests") -Recurse -File)
    $clashes = @($ours | Where-Object { $optixNames.Contains($_.Name) })
    foreach ($file in $clashes) {
        Fail ("{0} has the name of one of OptiX's headers" -f $file.FullName.Substring($project.Length + 1))
    }
    if ($clashes.Count -eq 0) {
        Write-Output ("names: none of the {0} files under src\ and tests\ is named like one of OptiX's {1}" -f
            $ours.Count, $optixNames.Count)
    }

    # --- 4. no file names in the PTX ----------------------------------------------------------------------------------
    # Each line as written, and a string constant's bytes as text: nvcc writes a string (an assert's __FILE__, a
    # printf's format) as a byte array, ".global .align 1 .b8 $str[12] = {67, 58, 92, ..., 0}". Only arrays that read
    # as a C string (printable bytes, then the 0) are decoded: nvcc writes tables of numbers as byte arrays too, whose
    # bytes, if random, would spell a drive letter's "X:\" or "X:/" about once in 200000
    $userFolder = if ($env:USERPROFILE) { $env:USERPROFILE } else { "" }
    foreach ($m in $modules) {
        $lines = [IO.File]::ReadAllLines((Join-Path $optixDir "$($m.name).ptx"))
        $found = 0
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            $why = $null
            if ($line -match '^\s*\.(file|loc)\b') { $why = "a .$($Matches[1]) line" }
            $texts = @($line)
            if ($line -match '\.b8\s+[^\[=]+\[\d+\]\s*=\s*\{\s*(\d+(\s*,\s*\d+)*)\s*\}') {
                $codes = @($Matches[1] -split '\s*,\s*' | ForEach-Object { [int]$_ })
                if ($codes.Count -gt 1 -and $codes[-1] -eq 0) {
                    $string = -join @($codes[0..($codes.Count - 2)] | ForEach-Object { [char]$_ })
                    if ($string -cmatch '^[\t\n\r\x20-\x7E]+$') { $texts += $string }
                }
            }
            foreach ($t in $texts) {
                if ($why) { break }
                if ($t -match '(?<![A-Za-z0-9_])[A-Za-z]:[\\/]') { $why = "a path with a drive letter" }
                elseif ($t -match '(?i)[\\/](Users|home)[\\/]') { $why = "a user's folder" }
                elseif ($userFolder -and ($t.IndexOf($userFolder, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                        $t.IndexOf($userFolder.Replace("\", "/"), [StringComparison]::OrdinalIgnoreCase) -ge 0)) {
                    $why = "a user's folder"
                }
            }
            if (-not $why) { continue }
            $found++
            if ($found -le 5) { Fail ("{0}.ptx:{1}: {2}" -f $m.name, ($i + 1), $why) }
            else { $failures++ }
        }
        if ($found -gt 5) { Write-Output ("  ... and {0} more lines of {1}.ptx" -f ($found - 5), $m.name) }
        if ($found -eq 0) { Write-Output "$($m.name).ptx: no .file or .loc line, no path" }
    }

    # --- 5. Debug = Release -------------------------------------------------------------------------------------------
    if ($Debug) {
        $debugOptix = Join-Path (Resolve-Folder $Debug) "optix"
        if (-not (Test-Path -LiteralPath $debugOptix -PathType Container)) { throw "irgate: no $debugOptix" }
        $releasePtx = @(Get-ChildItem -LiteralPath $optixDir -File | Where-Object { $_.Extension -eq ".ptx" })
        $debugPtx = @(Get-ChildItem -LiteralPath $debugOptix -File | Where-Object { $_.Extension -eq ".ptx" })
        foreach ($file in $releasePtx) {
            $other = Join-Path $debugOptix $file.Name
            if (-not (Test-Path -LiteralPath $other)) { Fail "the Debug build has no optix\$($file.Name)" }
            elseif ((Get-Item -LiteralPath $other).Length -ne $file.Length -or
                    (Get-Sha256 $other) -ne (Get-Sha256 $file.FullName)) {
                Fail "the Debug build's optix\$($file.Name) isn't Release's"
            }
            else { Write-Output "Debug = Release: optix\$($file.Name), byte for byte" }
        }
        foreach ($file in $debugPtx) {
            if (-not (Test-Path -LiteralPath (Join-Path $optixDir $file.Name))) {
                Fail "the Debug build has optix\$($file.Name), the Release build hasn't"
            }
        }
    }

    # The record, and the copies the next diff starts from: only when every check passed
    if ($failures -eq 0) {
        New-Item -ItemType Directory -Force $ir | Out-Null
        [IO.File]::WriteAllLines((Join-Path $ir "step$Step.txt"),
            [string[]]@($modules | ForEach-Object { "{0} {1}" -f $_.name, $current[$_.name] }))
        if ($Expect -eq "changed") {
            $saved = Join-Path $ir "step$Step"
            New-Item -ItemType Directory -Force $saved | Out-Null
            foreach ($m in $modules) {
                Copy-Item -LiteralPath (Join-Path $optixDir "$($m.name).ptx") -Destination $saved -Force
            }
        }
    }
    $parts = @($modules | ForEach-Object {
            $status = if (-not $earlier) { "new" } elseif ($changed[$_.name]) { "changed" } else { "unchanged" }
            "{0} {1} ({2})" -f $_.what, $current[$_.name].Substring(0, 8), $status })
    Write-Output ("OptiX programs: " + ($parts -join ", "))
    if ($failures -gt 0) {
        Write-Output "IR gate: FAIL ($failures)"
        exit 1
    }
    Write-Output "IR gate: PASS"
    exit 0
}
catch {
    Write-Output "Error: $($_.Exception.Message)"
    exit 2
}
finally {
    if ($temp -and (Test-Path -LiteralPath $temp)) { Remove-Item -LiteralPath $temp -Recurse -Force }
}
