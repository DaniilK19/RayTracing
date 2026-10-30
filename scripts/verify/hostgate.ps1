# The host gate: the GPU programs' host code (options, scene translation, the window's session and scripts, saving)
# behaves as before. The SASS gate proves the kernels unchanged; this proves what they are given and what is done with
# their results. Its machine state is kept in -State (out\m5, never committed), and -Reference names the exes whose
# output makes each baseline, once, in State\baseline\ (Step 85's, out\m5\bin85, the SASS gate's base too). Its parts,
# each named as -Only names it:
#   1. cli, text: the command lines of cli-cases.txt, each ending before it renders: their output and exit code,
#      against the newest earlier step's accepted texts (State\cli\stepNN\, or the reference exes' in
#      State\baseline\cli\), apart from the differences the step allows (State\cli-allowed\NN.diff, a git diff of the
#      texts);
#   2. regress, the window: tests\regress-viewer.txt rendered headless, every frame's bytes (the display kernel's) and
#      what the window prints (each scene's strategy and precision, each image's samples);
#   3. smoke, RayTracingGPU: five small renders (float64; a strategy; rounds of strata with short launches; the lights
#      list; the kernel for all three books on a first-book scene), the PFM's and the PNG's bytes, and the Scene, View,
#      Kernel, mean radiance and rays-traced lines;
#   4. tour, with -Tour: docs\viewer-tour.txt, the README GIF's script, rendered headless: every frame and the --record
#      file;
#   5. strategies, with -Strategies: tests\window-strategies.input, the window's M, Tab and F2 through the strategies
#      and the scenes beyond the books. A live window splits its samples over frames by the clock, so two live runs
#      needn't agree frame for frame: the reference viewer ran the input once in a WINDOW, as its first lines say
#      (RayTracingViewer cornell -w 320 --height 180 -s 32 --input FILE --record R), and its recording, kept in
#      State\baseline\, is replayed without a window (--headless --script R --record R2 --frames F, in a folder of its
#      own, where P saves its views): what the replay prints (times and the folder masked), R2's SHA-256 and every
#      frame's, against the reference viewer's replay. A replay sets the recorded settings directly and presses no
#      key, so it proves how those states render, not where M, Tab and F2 lead. The keys are checked by a recording's
#      key trace, its lines other than its frame lines: the settings each key led to, in turn, which the clock doesn't
#      change. With -StrategiesLive (Step 95) the viewer under test runs the input live once more, in a window, and its
#      recording's key trace must equal the reference's (State\baseline\window-strategies-keys.txt, taken from the
#      reference's recording, no run of its own); what its console printed, progress lines left out, is shown for
#      review. Without -StrategiesLive only the replay runs. A live run comes first: while a fullscreen application is
#      in front, the queue opens no window, and the gate stops at once, postponed (exit code 4), having run nothing
#      else;
#   6. chained, the chained sections, one a line of chained.txt (whose first lines give the format): a viewer script
#      or a list of RayTracingGPU commands, compared with the newest earlier step's accepted lines (State\chained\
#      stepNN\<name>.txt). Before any step has accepted lines, a section is compared with the reference exes' (made
#      once into State\baseline\chained-<name>.txt), or, for options they don't have, with nothing: its first run
#      waits for review. When an earlier step accepted lines of a section that chained.txt no longer lists, the gate
#      fails before anything runs, unless chained.txt retires it (a line "retired|<name>"): a line lost in a merge
#      would drop its gate silently.
# Parts 2 to 5 compare with the reference exes' baselines. Every render goes through gpu.ps1 (the queue and the GPU
# log); results go to State\runs\stepNN\host\. Without -Only, parts 1, 2, 3 and 6 run, and 4 and 5 with -Tour and
# -Strategies (or -StrategiesLive). -Only LIST (comma-separated part names) runs only those, -Tour and -Strategies
# adding theirs, and the last line names them: a step that met a difference renders again only the parts it accepts.
#
# A stamped chained section is OptiX's output, whose programs the graphics driver compiles. Its kept lines start with
# "stamp: <driver>|<programs>": the driver's version (nvidia-smi), and the first word of the build's
# optix\optix_programs.sha256 (64 hex digits), or none before OptiX is in the build. Under the stamp of the accepted
# lines it compares as any other section, so a difference fails. Under another stamp it says RE-VERIFY (the driver or
# the programs changed): its lines wait for review, neither a pass nor a failure, and once OptiX's output has been
# verified again, -Accept takes them with the new stamp. A stamp that can't be read is an error, never a stamp of its
# own (which would turn a real difference under the same driver and programs into RE-VERIFY): the gate stops before
# any render (exit code 2) when nvidia-smi fails or prints no version, when a stamped section's kept lines don't start
# with a stamp line, and when they were kept under OptiX programs and the build has no optix_programs.sha256.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\hostgate.ps1 -Step 86 [-Tour] [-Strategies]
#              [-StrategiesLive] [-Only cli,regress,smoke,tour,strategies,chained] [-Accept]
# Texts and chained lines that differ from the allowed or the accepted ones fail the gate and are left for review, in
# State\cli-allowed\NN.pending.diff and State\chained\NN.<name>.pending.txt. -Accept then accepts exactly those (a run
# whose differences changed since is refused), and keeps the texts and lines as step NN's, which the next step compares
# with; with -Only, only the parts it names (-Accept -Only chained: the chained sections' lines alone; the texts only
# with cli in the list). Run it once at every step's commit, also when nothing differs.
# Exit code 0 when everything matches ("Host gate: PASS"), 1 when something differs ("Host gate: FAIL (n)"), 2 for an
# error ("Host gate: ERROR": a stamp that can't be read, a render that fails, a refusal of the queue other than a live
# window's postponement, a bad option or line), 3 when nothing differs but a stamped section is to be re-verified
# ("Host gate: RE-VERIFY (n)"), 4 when a live window run of part 5 is postponed.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [switch]$Tour,
    [switch]$Strategies,
    [switch]$StrategiesLive,
    [string[]]$Only = @(),
    [switch]$Accept,
    [string]$Build = "out\build\x64-release",
    [string]$Reference = "out\m5\bin85",
    [string]$State = "out\m5"
)
$ErrorActionPreference = "Stop"
# An error anywhere is exit code 2, never a verdict
trap {
    Write-Output "Error: $($_.Exception.Message)"
    Write-Output "Host gate: ERROR"
    exit 2
}
[Console]::OutputEncoding = [Text.Encoding]::UTF8 # the programs write UTF-8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"

function Get-FullPath([string]$path) {
    if ([IO.Path]::IsPathRooted($path)) { return $path }
    return Join-Path $project $path
}
$buildDir = Get-FullPath $Build
$referenceDir = Get-FullPath $Reference
$stateDir = Get-FullPath $State
$run = Join-Path $stateDir ("runs\step{0}\host" -f $Step)
$baseline = Join-Path $stateDir "baseline"
$chainedDir = Join-Path $stateDir "chained"
$failures = 0
$reverify = 0

# The parts this run renders. -Only takes "cli,chained" as one argument (from -File) or as an array (from &)
$allParts = @("cli", "regress", "smoke", "tour", "strategies", "chained")
$named = @(($Only -join ",") -split '[,\s]+' | Where-Object { $_ })
foreach ($part in $named) {
    if ($allParts -notcontains $part) { throw "hostgate: -Only takes $($allParts -join ', '), not '$part'" }
}
$runParts = @()
if ($named.Count -gt 0) { $runParts += $named } else { $runParts += @("cli", "regress", "smoke", "chained") }
if ($Tour) { $runParts += "tour" }
if ($Strategies -or $StrategiesLive) { $runParts += "strategies" }
$runParts = @($allParts | Where-Object { $runParts -contains $_ })
$onlyNote = if ($named.Count -gt 0) { " (only " + ($runParts -join ", ") + ")" } else { "" }

function Resolve-Exe([string]$folder, [string]$name) {
    $path = Join-Path $folder $name
    if (-not (Test-Path -LiteralPath $path)) { throw "hostgate: no $path" }
    return $path
}
$newGpu = Resolve-Exe $buildDir "RayTracingGPU.exe"
$newViewer = Resolve-Exe $buildDir "RayTracingViewer.exe"
$refGpu = Resolve-Exe $referenceDir "RayTracingGPU.exe"
$refViewer = Resolve-Exe $referenceDir "RayTracingViewer.exe"

# The chained sections, read before anything runs: a bad line stops the gate before any GPU work
$sections = @()
$retired = @()
foreach ($line in @(Get-Content -LiteralPath (Join-Path $PSScriptRoot "chained.txt") -Encoding UTF8 |
            Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") })) {
    $p = @($line.Split("|") | ForEach-Object { $_.Trim() })
    if ($p.Count -eq 2 -and $p[0] -ceq "retired") {
        if ($p[1] -notmatch '^[A-Za-z0-9._-]+$') { throw "hostgate: chained.txt: '$line' isn't retired|name" }
        $retired += $p[1]
        continue
    }
    if ($p.Count -ne 5 -or $p[0] -notmatch '^[A-Za-z0-9._-]+$' -or $p[1] -cnotmatch '^[VG]$' -or
        $p[3] -cnotmatch '^(yes|no)$' -or $p[4] -cnotmatch '^(reference|none)$') {
        throw "hostgate: chained.txt: '$line' isn't name|V or G|file|yes or no|reference or none"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $project $p[2]) -PathType Leaf)) {
        throw "hostgate: chained.txt: no file $($p[2])"
    }
    if (@($sections | Where-Object { $_.name -eq $p[0] }).Count -gt 0) {
        throw "hostgate: chained.txt: two sections named $($p[0])"
    }
    $sections += [pscustomobject]@{ name = $p[0]; kind = $p[1]; file = $p[2]; stamped = $p[3] -eq "yes";
        baseline = $p[4] }
}
foreach ($name in $retired) {
    if (@($sections | Where-Object { $_.name -eq $name }).Count -gt 0) {
        throw "hostgate: chained.txt: $name is a section and retired"
    }
}

function New-Folder([string]$path) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    New-Item -ItemType Directory -Force $path | Out-Null
}

# A file's text with its line ends as \n and nothing after its last line
function Read-Text([string]$path) {
    return ([IO.File]::ReadAllText($path) -replace "`r`n", "`n").TrimEnd()
}

# A render through the queue; its output lines, and $LASTEXITCODE set
function Invoke-Queued([string]$label, [string]$exe, [string[]]$arguments) {
    $all = @("-Step", "$Step", "-Label", $label, "--", $exe) + $arguments
    $lines = @(& $gpuScript @all)
    $script:code = $LASTEXITCODE
    if ($script:code -eq 3 -or $script:code -eq 4) {
        $lines | ForEach-Object { Write-Host $_ }
        throw "hostgate: the queue refused '$label' (exit code $($script:code))"
    }
    return , $lines
}

function Get-Hashes([string]$folder, [string]$filter) {
    return @(Get-ChildItem -LiteralPath $folder -Filter $filter -File | Sort-Object Name |
        ForEach-Object { "{0} {1}" -f $_.Name, (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash })
}

# A line a program printed with the paths of the run's folder and of the repository replaced by names, so that runs
# in different folders (the reference exes', a later step's) print the same
function Hide-Paths([string]$text, [string]$folder) {
    foreach ($pair in @(@($folder, "<out>"), @($project, "<root>"))) {
        $text = $text.Replace($pair[0], $pair[1]).Replace(($pair[0] -replace '\\', '/'), $pair[1])
    }
    return $text
}

# Compares a list of lines with the baseline file, made by $make with the reference exes when it doesn't exist yet
function Compare-WithBaseline([string]$what, [string[]]$lines, [string]$file, [scriptblock]$make) {
    $path = Join-Path $baseline $file
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $path) | Out-Null
        $made = & $make
        [IO.File]::WriteAllLines($path, [string[]]$made)
        Write-Host "  (baseline made with $Reference)"
    }
    $expected = @([IO.File]::ReadAllLines($path))
    $diff = @(Compare-Object -ReferenceObject $expected -DifferenceObject $lines -SyncWindow 0 -CaseSensitive)
    if ($diff.Count -eq 0 -and $expected.Count -eq $lines.Count) {
        Write-Host ("{0}: identical ({1} lines)" -f $what, $lines.Count)
        return 0
    }
    Write-Host ("{0}: DIFFERENT" -f $what)
    $diff | Select-Object -First 20 | ForEach-Object { Write-Host ("  {0} {1}" -f $_.SideIndicator, $_.InputObject) }
    return 1
}

# --- The stamp and the chained sections' kept lines, read before anything runs --------------------------------------

# A stamped section's stamp, "<driver>|<programs>": the graphics driver's version and the OptiX programs' SHA-256 (the
# first word of the build's optix\optix_programs.sha256, or none before OptiX is in the build). nvidia-smi's whole
# output is taken before its exit code is read: piped into Select-Object -First, the program is stopped before
# PowerShell stores its exit code, which leaves the previous command's. A driver that can't be read is an error, never
# a stamp of its own
function Get-Stamp {
    $smi = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $smi) { throw "hostgate: no nvidia-smi on the PATH: the driver's version, the stamp's half, is unread" }
    $ErrorActionPreference = "Continue" # nvidia-smi's standard error must not stop the script
    $out = @(& $smi.Path --query-gpu=driver_version --format=csv,noheader 2>$null)
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $driver = if ($out.Count -gt 0) { ([string]$out[0]).Trim() } else { "" }
    if ($code -ne 0 -or $driver -notmatch '^\d+(\.\d+)+$') {
        throw (("hostgate: nvidia-smi printed no driver version (exit code {0}): a stamp that can't be read stops " +
            "the gate, it is never a stamp of its own") -f $code)
    }
    $programs = "none"
    $sums = Join-Path $buildDir "optix\optix_programs.sha256"
    if (Test-Path -LiteralPath $sums) {
        $first = @(([IO.File]::ReadAllText($sums)).Trim() -split '\s+')[0]
        if ($first -notmatch '^[0-9A-Fa-f]{64}$') { throw "hostgate: $sums doesn't start with a SHA-256" }
        $programs = $first.ToLowerInvariant()
    }
    return "$driver|$programs"
}

# The newest earlier step's accepted lines of a section, State\chained\stepK\<name>.txt with K < NN: @{ path; step },
# or $null
function Get-EarlierKept([string]$name) {
    $earlier = @(Get-ChildItem -LiteralPath $chainedDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step(\d+)$' -and [int]$Matches[1] -lt $Step -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "$name.txt")) } |
        Sort-Object { [int]($_.Name.Substring(4)) } | Select-Object -Last 1)
    if ($earlier.Count -eq 0) { return $null }
    return @{ path = (Join-Path $earlier[0].FullName "$name.txt"); step = [int]$earlier[0].Name.Substring(4) }
}

# A stamped section's kept lines: their stamp and the lines after it. Kept lines that don't start with "stamp:
# <driver version>|<programs>" stop the gate (compared under a stamp of "", they would say RE-VERIFY whatever differs),
# and so do lines kept under OptiX programs when this build has none ($stamp, read first)
function Read-StampedKept([string]$path) {
    $text = Read-Text $path
    $end = $text.IndexOf("`n")
    $first = if ($end -lt 0) { $text } else { $text.Substring(0, $end) }
    if ($first -cmatch '^stamp: (\d+(?:\.\d+)+)\|([0-9a-f]{64}|none)$') {
        $keptDriver = $Matches[1]
        $keptPrograms = $Matches[2]
    }
    else { throw "hostgate: $path, a stamped section's lines, doesn't start with 'stamp: <driver>|<programs>'" }
    if ($keptPrograms -ne "none" -and $stamp.EndsWith("|none")) {
        throw (("hostgate: {0} was kept under OptiX programs {1}, but this build has no optix\optix_programs.sha256: " +
            "the programs' half of the stamp can't be read") -f $path, $keptPrograms.Substring(0, 8))
    }
    $body = if ($end -lt 0) { "" } else { $text.Substring($end + 1) }
    return @("$keptDriver|$keptPrograms", $body)
}

$stamp = ""
if ($runParts -contains "chained") {
    # A section that an earlier step accepted lines of and chained.txt no longer lists, nor retires: its line was lost
    $listed = @($sections | ForEach-Object { $_.name })
    $lost = @(Get-ChildItem -LiteralPath $chainedDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step(\d+)$' -and [int]$Matches[1] -lt $Step } |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -File -Filter "*.txt" } |
        ForEach-Object { $_.BaseName } | Sort-Object -Unique |
        Where-Object { $listed -notcontains $_ -and $retired -notcontains $_ })
    if ($lost.Count -gt 0) {
        foreach ($name in $lost) {
            Write-Output (("chained {0}: an earlier step accepted its lines, but chained.txt no longer lists it: put " +
                "its line back, or retire it with a line ""retired|{0}""") -f $name)
        }
        Write-Output ("Host gate: FAIL ({0}){1}" -f $lost.Count, $onlyNote)
        exit 1
    }
    if (@($sections | Where-Object { $_.stamped }).Count -gt 0) {
        $stamp = Get-Stamp
        foreach ($section in @($sections | Where-Object { $_.stamped })) {
            $earlier = Get-EarlierKept $section.name
            if ($earlier) { Read-StampedKept $earlier.path | Out-Null }
        }
    }
}

# --- 5. The window's strategies and scenes (-Strategies), first ------------------------------------------------------
# The reference viewer's one live run, under scripted input, makes the recording (State\baseline\window-strategies-
# record.txt); the queue refuses that window while a fullscreen application is in front (as it refuses all GPU work
# then), and postponed, the gate stops before parts 1 to 4 and 6, which keep what they had. The rendering checks are
# replays of that recording without a window: deterministic, frame for frame. With -StrategiesLive the viewer under
# test runs the input live too, right after, for the keys: its recording's key trace against the reference's.

function Get-StrategiesRecording {
    $record = Join-Path $baseline "window-strategies-record.txt"
    if (Test-Path -LiteralPath $record) { return $record }
    $folder = Join-Path $run "strategies-live"
    New-Folder $folder
    $made = Join-Path $folder "record.txt"
    $all = @("-Step", "$Step", "-Label", "window-strategies (reference, live)", "-Window", "--", $refViewer, "cornell",
        "-w", "320", "--height", "180", "-s", "32", "--input", (Join-Path $project "tests\window-strategies.input"),
        "--record", $made)
    # P saves the view on screen in the current folder (and prints its path): the run's folder, not the repository
    Push-Location -LiteralPath $folder
    try { $out = @(& $gpuScript @all); $exit = $LASTEXITCODE }
    finally { Pop-Location }
    if ($exit -eq 4) { return $null }
    if ($exit -ne 0) {
        $out | ForEach-Object { Write-Host "  $_" }
        throw "hostgate: the live run of window-strategies.input failed (exit code $exit)"
    }
    New-Item -ItemType Directory -Force $baseline | Out-Null
    Copy-Item -LiteralPath $made $record
    Write-Host "  (window-strategies.input recorded live by $Reference's viewer)"
    return $record
}

# A recording's key trace: its lines other than its frame lines, the settings (scene, precision, kernel, sampling,
# seed, strata, the display) each key led to, in turn. How many frames a live window shows between two keys depends on
# the clock; where each key leads doesn't
function Get-KeyTrace([string]$recording) {
    return , @([IO.File]::ReadAllLines($recording) | Where-Object { $_ -cnotmatch '^frame( |$)' })
}

# The viewer under test's live run of the input (-StrategiesLive), in a window: its recording's key trace and what its
# console printed, progress lines (Done, Rays traced: times) left out; $null when postponed
function Get-StrategiesLive {
    $folder = Join-Path $run "strategies-live-new"
    New-Folder $folder
    $made = Join-Path $folder "record.txt"
    $all = @("-Step", "$Step", "-Label", "window-strategies (live)", "-Window", "--", $newViewer, "cornell", "-w",
        "320", "--height", "180", "-s", "32", "--input", (Join-Path $project "tests\window-strategies.input"),
        "--record", $made)
    Push-Location -LiteralPath $folder
    try { $out = @(& $gpuScript @all); $exit = $LASTEXITCODE }
    finally { Pop-Location }
    if ($exit -eq 4) { return $null }
    if ($exit -ne 0) {
        $out | ForEach-Object { Write-Host "  $_" }
        throw "hostgate: the live run of window-strategies.input by the viewer under test failed (exit code $exit)"
    }
    $console = @($out | Where-Object { $_ -notmatch '^gpu\.ps1:' -and $_ -notmatch '^(Done|Rays traced):' } |
        ForEach-Object { Hide-Paths $_ $folder })
    [IO.File]::WriteAllLines((Join-Path $folder "console.txt"), [string[]]$console)
    return @{ trace = (Get-KeyTrace $made); console = $console }
}

# One replay of the recording without a window: what it prints, its own recording's SHA-256 and every frame's
function Get-StrategiesLines([string]$viewer, [string]$recording, [string]$folder, [string]$suffix) {
    New-Folder $folder
    $frames = Join-Path $folder "frames"
    New-Item -ItemType Directory -Force $frames | Out-Null
    $record = Join-Path $folder "record.txt"
    Push-Location -LiteralPath $folder
    try {
        $out = Invoke-Queued "window-strategies$suffix" $viewer @("--headless", "--script", $recording, "--record",
            $record, "--frames", $frames)
    }
    finally { Pop-Location }
    if ($script:code -ne 0) {
        $out | ForEach-Object { Write-Host "  $_" }
        throw "hostgate: the replay of window-strategies$suffix failed (exit code $($script:code))"
    }
    $lines = @($out | Where-Object { $_ -notmatch '^gpu\.ps1:' } |
        ForEach-Object { Hide-Paths ($_ -replace ', [0-9.]+ s$', '') $folder })
    $lines += Get-Hashes $frames "f*.png"
    $lines += "record " + (Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash
    return , $lines
}

if ($runParts -contains "strategies") {
    $recording = Get-StrategiesRecording
    $live = $null
    if ($null -ne $recording -and $StrategiesLive) { $live = Get-StrategiesLive }
    if ($null -eq $recording -or ($StrategiesLive -and $null -eq $live)) {
        Write-Output ("window, window-strategies.input: postponed: the queue opens no window while a fullscreen " +
            "application is in front (or the GPU is too hot): run -Strategies (-StrategiesLive) again later")
        Write-Output "Host gate: POSTPONED"
        exit 4
    }
    if ($StrategiesLive) {
        Write-Output ("window, window-strategies.input live by the viewer under test: its console, progress lines " +
            "left out, for review:")
        $live.console | ForEach-Object { Write-Output "  | $_" }
        $failures += Compare-WithBaseline ("window, window-strategies.input live: the key trace (the recording's " +
            "settings, frame lines left out)") $live.trace "window-strategies-keys.txt" { Get-KeyTrace $recording }
    }
    $strategiesLines = Get-StrategiesLines $newViewer $recording (Join-Path $run "strategies") ""
    $failures += Compare-WithBaseline "window, window-strategies.input (its recording replayed)" $strategiesLines `
        "window-strategies.txt" {
        Get-StrategiesLines $refViewer $recording (Join-Path $run "strategies-ref") " (reference)"
    }
}

# --- 1. The command lines --------------------------------------------------------------------------------------------

# Splits a case's arguments at spaces, keeping "quoted parts" together
function Split-Arguments([string]$text) {
    $parts = @()
    foreach ($m in [regex]::Matches($text, '"([^"]*)"|(\S+)')) {
        $parts += if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
    }
    return , $parts
}

# Runs a program that must end at once (30 s at most); its output, the root folder's path masked
function Invoke-Text([string]$exe, [string[]]$arguments, [string]$caseLine) {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $exe
    $info.Arguments = ($arguments | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join " "
    $info.WorkingDirectory = $project
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Text.Encoding]::UTF8
    $info.StandardErrorEncoding = [Text.Encoding]::UTF8
    $p = [Diagnostics.Process]::Start($info)
    $out = $p.StandardOutput.ReadToEndAsync()
    $err = $p.StandardError.ReadToEndAsync()
    if (-not $p.WaitForExit(30000)) {
        $p.Kill()
        throw "hostgate: '$caseLine' ran for 30 s: a case that renders?"
    }
    $p.WaitForExit()
    $text = "> $caseLine`n" + $out.Result
    if ($err.Result) { $text += "--- stderr`n" + $err.Result }
    $text += "--- exit code $($p.ExitCode)`n"
    $text = $text -replace "`r`n", "`n"
    $text = $text.Replace($project, "<root>").Replace(($project -replace '\\', '/'), "<root>")
    # A case that renders a tiny image prints its times: masked
    $text = $text -replace '(?m), [0-9.]+ s$', '' -replace ', [0-9.]+ million per second', ''
    return $text
}

if ($runParts -contains "cli") {
    $cases = @(Get-Content -LiteralPath (Join-Path $PSScriptRoot "cli-cases.txt") -Encoding UTF8 |
        Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") })
    $cliRun = Join-Path $run "cli"
    New-Folder (Join-Path $cliRun "ref")
    New-Folder (Join-Path $cliRun "new")
    $sha1 = [Security.Cryptography.SHA1]::Create()
    # The newest accepted texts of an earlier step
    $accepted = @(Get-ChildItem -LiteralPath (Join-Path $stateDir "cli") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step(\d+)$' -and [int]$Matches[1] -lt $Step } |
        Sort-Object { [int]($_.Name.Substring(4)) } -Descending)
    $madeBaseline = 0
    foreach ($case in $cases) {
        $program = $case.Substring(0, 1)
        $argText = $case.Substring(1).Trim()
        $hash = -join ($sha1.ComputeHash([Text.Encoding]::UTF8.GetBytes($case)) | Select-Object -First 5 |
            ForEach-Object { $_.ToString("x2") })
        $name = "$program-$hash.txt"
        $arguments = Split-Arguments $argText
        $newExe = if ($program -ceq "G") { $newGpu } elseif ($program -ceq "V") { $newViewer }
        else { throw "bad case: $case" }
        [IO.File]::WriteAllText((Join-Path $cliRun "new\$name"), (Invoke-Text $newExe $arguments $case))
        # The previous step's text, or the reference exes' (kept in State\baseline\cli)
        $ref = $null
        foreach ($dir in $accepted) {
            $candidate = Join-Path $dir.FullName $name
            if (Test-Path -LiteralPath $candidate) { $ref = $candidate; break }
        }
        if (-not $ref) {
            $ref = Join-Path $baseline "cli\$name"
            if (-not (Test-Path -LiteralPath $ref)) {
                New-Item -ItemType Directory -Force (Split-Path -Parent $ref) | Out-Null
                $refExe = if ($program -eq "G") { $refGpu } else { $refViewer }
                [IO.File]::WriteAllText($ref, (Invoke-Text $refExe $arguments $case))
                $madeBaseline++
            }
        }
        Copy-Item -LiteralPath $ref (Join-Path $cliRun "ref\$name")
    }
    if ($madeBaseline -gt 0) { Write-Output "  ($madeBaseline texts of the reference exes made)" }
    Push-Location -LiteralPath $cliRun
    $ErrorActionPreference = "Continue"
    $diff = (& git -c core.autocrlf=false diff --no-index --no-color -U0 --text ref new 2>&1 |
        ForEach-Object { "$_" }) -join "`n"
    $ErrorActionPreference = "Stop"
    Pop-Location
    $diff = $diff.TrimEnd()
    $allowedPath = Join-Path $stateDir ("cli-allowed\{0}.diff" -f $Step)
    $allowed = ""
    if (Test-Path -LiteralPath $allowedPath) {
        $allowed = ([IO.File]::ReadAllText($allowedPath) -replace "`r`n", "`n").TrimEnd()
    }
    # A run whose differences aren't the allowed ones leaves them here for review; -Accept accepts exactly these
    $pending = Join-Path $stateDir ("cli-allowed\{0}.pending.diff" -f $Step)
    $diffLines = ($diff -split "`n" | Where-Object { $_ }).Count
    if ($diff -ceq $allowed) {
        Write-Output ("command lines: {0} cases, {1}" -f $cases.Count,
            $(if ($diff) { "only the allowed differences ($diffLines lines)" } else { "identical" }))
        if ($Accept) {
            $keep = Join-Path $stateDir ("cli\step{0}" -f $Step)
            New-Folder $keep
            Copy-Item -Path (Join-Path $cliRun "new\*") -Destination $keep
            Write-Output "  (kept as Step $Step's texts)"
        }
    }
    elseif ($Accept -and (Test-Path -LiteralPath $pending) -and
        ([IO.File]::ReadAllText($pending) -replace "`r`n", "`n").TrimEnd() -ceq $diff) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $allowedPath) | Out-Null
        Move-Item -LiteralPath $pending $allowedPath -Force
        $keep = Join-Path $stateDir ("cli\step{0}" -f $Step)
        New-Folder $keep
        Copy-Item -Path (Join-Path $cliRun "new\*") -Destination $keep
        Write-Output ("command lines: {0} cases, the reviewed differences accepted as Step {1}'s ({2} lines)" -f
            $cases.Count, $Step, $diffLines)
    }
    else {
        New-Item -ItemType Directory -Force (Split-Path -Parent $pending) | Out-Null
        [IO.File]::WriteAllText($pending, $diff + "`n")
        $why = if ($Accept) { " (this run's differences aren't the ones reviewed)" } else { "" }
        Write-Output ("command lines: {0} cases, DIFFERENT from the allowed: review {1}, then -Accept{2}" -f
            $cases.Count, $pending, $why)
        $diff -split "`n" | Select-Object -First 60 | ForEach-Object { Write-Output "  $_" }
        $failures++
    }
}

# --- 2. The window's regression script -------------------------------------------------------------------------------

function Get-ScriptFrames([string]$viewer, [string]$label, [string]$scriptFile, [string]$folder, [string]$record) {
    New-Folder $folder
    $arguments = @("--headless", "--script", $scriptFile, "--frames", $folder)
    if ($record) { $arguments += @("--record", $record) }
    $out = Invoke-Queued $label $viewer $arguments
    if ($script:code -ne 0) { $out | ForEach-Object { Write-Host "  $_" }; throw "hostgate: $label failed" }
    # What the window says on the console, times masked: each scene's strategy and precision, each image's samples.
    # (Not for the tour, whose baseline holds the GIF's frames and record only: one scene, said once.)
    $lines = @()
    if (-not $record) {
        $lines = @($out | Where-Object { $_ -notmatch '^gpu\.ps1:' } |
            ForEach-Object { $_ -replace ', [0-9.]+ s$', '' })
    }
    $lines += Get-Hashes $folder "f*.png"
    if ($record) { $lines += "record " + (Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash }
    return , $lines
}

if ($runParts -contains "regress") {
    $frames = Get-ScriptFrames $newViewer "regress-viewer" "tests\regress-viewer.txt" (Join-Path $run "regress") ""
    $failures += Compare-WithBaseline "window, regress-viewer.txt" $frames "regress-viewer.txt" {
        Get-ScriptFrames $refViewer "regress-viewer (reference)" "tests\regress-viewer.txt" `
            (Join-Path $run "regress-ref") ""
    }
}

# --- 3. RayTracingGPU's small renders --------------------------------------------------------------------------------

$smokes = @(
    @("weekend", "--fp64", "-w", "64", "-s", "16"),
    @("cornell", "-p", "uniform", "-w", "64", "-s", "16"),
    @("final", "--strata", "4", "-s", "32", "--launch-ms", "1", "-w", "64"),
    @("glass", "-p", "lights", "-w", "64", "-s", "16"),
    @("materials", "--all-books-kernel", "-w", "64", "-s", "16"))
function Get-Smokes([string]$gpu, [string]$folder, [string]$suffix) {
    New-Folder $folder
    $lines = @()
    for ($k = 0; $k -lt $smokes.Count; $k++) {
        $pfm = Join-Path $folder "smoke$k.pfm"
        $png = Join-Path $folder "smoke$k.png"
        $out = Invoke-Queued ("smoke $k$suffix") $gpu ($smokes[$k] + @("-o", $pfm, "-o", $png))
        if ($script:code -ne 0) { $out | ForEach-Object { Write-Host "  $_" }; throw "hostgate: smoke $k failed" }
        $lines += "> " + ($smokes[$k] -join " ")
        $lines += @($out | Where-Object { $_ -match '^(Scene|View|Kernel|Mean radiance|Rays traced):?' } |
            ForEach-Object { $_ -replace ', [0-9.]+ million per second', '' })
        $lines += "pfm " + (Get-FileHash -LiteralPath $pfm -Algorithm SHA256).Hash
        $lines += "png " + (Get-FileHash -LiteralPath $png -Algorithm SHA256).Hash
    }
    return , $lines
}
if ($runParts -contains "smoke") {
    $smokeLines = Get-Smokes $newGpu (Join-Path $run "smoke") ""
    $failures += Compare-WithBaseline "RayTracingGPU, $($smokes.Count) small renders" $smokeLines "smoke.txt" {
        Get-Smokes $refGpu (Join-Path $run "smoke-ref") " (reference)"
    }
}

# --- 4. The GIF's tour -----------------------------------------------------------------------------------------------

if ($runParts -contains "tour") {
    $tourLines = Get-ScriptFrames $newViewer "tour" "docs\viewer-tour.txt" (Join-Path $run "tour") `
        (Join-Path $run "tour-record.txt")
    $failures += Compare-WithBaseline "window, the GIF's tour (frames and --record)" $tourLines "tour.txt" {
        Get-ScriptFrames $refViewer "tour (reference)" "docs\viewer-tour.txt" (Join-Path $run "tour-ref") `
            (Join-Path $run "tour-ref-record.txt")
    }
}

# --- 6. The chained sections, step to step ---------------------------------------------------------------------------
# Each line of chained.txt, rendered by this step's exes, against the newest earlier step's accepted lines
# (State\chained\stepNN\<name>.txt), else the reference exes' (baseline reference) or nothing (baseline none). A
# difference is left in State\chained\NN.<name>.pending.txt for review; -Accept then accepts exactly it, and keeps the
# lines as step NN's, as with the command lines' texts. Stamped sections: see the top of this file.

# A line of a RayTracingGPU render with its times and speeds masked (a first OptiX compile, say, takes longer)
function Hide-Times([string]$text) {
    return $text -replace ', [0-9.]+ million per second', '' -replace '\b[0-9]+(\.[0-9]+)? m?s\b', '<time>'
}

# One run of a chained section by the given exes, in its own folder: the lines it keeps, without a stamp
function Get-ChainedLines($section, [string]$viewer, [string]$gpu, [string]$folder, [string]$suffix) {
    New-Folder $folder
    $lines = @()
    if ($section.kind -ceq "V") {
        # A viewer script: what the window prints (times masked), every frame and the --record file
        $frames = Join-Path $folder "frames"
        $record = Join-Path $folder "record.txt"
        New-Item -ItemType Directory -Force $frames | Out-Null
        $out = Invoke-Queued ($section.name + $suffix) $viewer @("--headless", "--script", $section.file, "--frames",
            $frames, "--record", $record)
        if ($script:code -ne 0) {
            $out | ForEach-Object { Write-Host "  $_" }
            throw "hostgate: $($section.name)$suffix failed"
        }
        $lines += @($out | Where-Object { $_ -notmatch '^gpu\.ps1:' } |
            ForEach-Object { Hide-Paths ($_ -replace ', [0-9.]+ s$', '') $folder })
        $lines += Get-Hashes $frames "f*.png"
        $lines += "record " + (Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash
    }
    else {
        # A list of RayTracingGPU commands: each one's lines that say what it rendered, and every file it wrote (a
        # folder of its own per command, so the -denoised files count too)
        $commands = @(Get-Content -LiteralPath (Join-Path $project $section.file) -Encoding UTF8 |
            Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") } | ForEach-Object { $_.Trim() })
        for ($k = 0; $k -lt $commands.Count; $k++) {
            $name = "c{0:D2}" -f ($k + 1)
            $dir = Join-Path $folder $name
            New-Item -ItemType Directory -Force $dir | Out-Null
            # Split-Arguments emits its array as one object: cast, not @(...), which would nest it as one argument
            $arguments = [string[]](Split-Arguments $commands[$k]) + @("-o", (Join-Path $dir "$name.pfm"), "-o",
                (Join-Path $dir "$name.png"))
            $out = Invoke-Queued ("{0} {1}{2}" -f $section.name, $name, $suffix) $gpu $arguments
            if ($script:code -ne 0) {
                $out | ForEach-Object { Write-Host "  $_" }
                throw "hostgate: $($section.name) $name$suffix failed ($($commands[$k]))"
            }
            $lines += "> " + $commands[$k]
            $lines += @($out |
                Where-Object { $_ -match '^(Scene|View|Kernel|Mean radiance|Rays traced|Shadow rays|Denoised):?' } |
                ForEach-Object { Hide-Paths (Hide-Times $_) $folder })
            $lines += Get-Hashes $dir "*"
        }
    }
    return , $lines
}

# How many lines differ, place by place
function Get-DifferentLines([string]$a, [string]$b) {
    $x = @($a -split "`n")
    $y = @($b -split "`n")
    $n = 0
    for ($i = 0; $i -lt [Math]::Max($x.Count, $y.Count); $i++) {
        if ($i -ge $x.Count -or $i -ge $y.Count -or $x[$i] -cne $y[$i]) { $n++ }
    }
    return $n
}

$chainedSections = @()
if ($runParts -contains "chained") { $chainedSections = $sections }
foreach ($section in $chainedSections) {
    $what = "chained {0} ({1})" -f $section.name, $section.file
    $file = $section.name + ".txt"
    $lines = Get-ChainedLines $section $newViewer $newGpu (Join-Path $run ("chained\" + $section.name)) ""
    $body = ($lines -join "`n").TrimEnd()
    $text = if ($section.stamped) { "stamp: $stamp`n$body" } else { $body }
    $pending = Join-Path $chainedDir ("{0}.{1}.pending.txt" -f $Step, $section.name)
    $keepDir = Join-Path $chainedDir ("step{0}" -f $Step)
    $keep = Join-Path $keepDir $file
    # The lines of this very run were reviewed: -Accept takes them
    $reviewed = $Accept -and (Test-Path -LiteralPath $pending) -and (Read-Text $pending) -ceq $text

    # What this run is compared with, without a stamp; the stamp it was accepted under (a baseline has the run's own)
    $previous = $null
    $previousStamp = $stamp
    $from = "no earlier step"
    $earlier = Get-EarlierKept $section.name
    if ($earlier) {
        $from = "Step {0}'s" -f $earlier.step
        if ($section.stamped) {
            $kept = Read-StampedKept $earlier.path
            $previousStamp = $kept[0]
            $previous = $kept[1]
        }
        else { $previous = Read-Text $earlier.path }
    }
    elseif ($section.baseline -eq "reference") {
        $basePath = Join-Path $baseline ("chained-{0}.txt" -f $section.name)
        if (-not (Test-Path -LiteralPath $basePath)) {
            $refFolder = Join-Path $run ("chained\" + $section.name + "-ref")
            $made = Get-ChainedLines $section $refViewer $refGpu $refFolder " (reference)"
            New-Item -ItemType Directory -Force $baseline | Out-Null
            [IO.File]::WriteAllText($basePath, ($made -join "`n").TrimEnd() + "`n")
            Write-Output "  (baseline made with $Reference)"
        }
        $previous = Read-Text $basePath
        $from = "the reference exes'"
    }

    if ($null -ne $previous -and $previousStamp -cne $stamp) {
        # Only a stamped section gets here: another driver or other OptiX programs, so neither a pass nor a failure
        if ($reviewed) {
            New-Item -ItemType Directory -Force $keepDir | Out-Null
            Move-Item -LiteralPath $pending $keep -Force
            Write-Output ("{0}: the reviewed lines accepted as Step {1}'s, with the new stamp ({2} lines)" -f $what,
                $Step, $lines.Count)
        }
        else {
            New-Item -ItemType Directory -Force $chainedDir | Out-Null
            [IO.File]::WriteAllText($pending, $text + "`n")
            $n = Get-DifferentLines $previous $body
            $news = if ($n -eq 0) { "the same lines as $from" } else { "$n lines differ from $from" }
            Write-Output ("{0}: RE-VERIFY {1} (driver or OptiX programs changed): {2}; review {3}, then -Accept" -f
                $what, $section.name, $news, $pending)
            $reverify++
        }
    }
    elseif ($null -ne $previous -and $previous -ceq $body) {
        Write-Output ("{0}: identical to {1} ({2} lines)" -f $what, $from, $lines.Count)
        if ($Accept) {
            New-Item -ItemType Directory -Force $keepDir | Out-Null
            [IO.File]::WriteAllText($keep, $text + "`n")
        }
    }
    elseif ($reviewed) {
        New-Item -ItemType Directory -Force $keepDir | Out-Null
        Move-Item -LiteralPath $pending $keep -Force
        Write-Output ("{0}: the reviewed lines accepted as Step {1}'s ({2} lines)" -f $what, $Step, $lines.Count)
    }
    else {
        New-Item -ItemType Directory -Force $chainedDir | Out-Null
        [IO.File]::WriteAllText($pending, $text + "`n")
        if ($null -eq $previous) {
            Write-Output ("{0}: NEW, no earlier step has accepted its lines: review {1}, then -Accept" -f $what,
                $pending)
            $lines | Select-Object -First 40 | ForEach-Object { Write-Output "  + $_" }
        }
        else {
            Write-Output ("{0}: DIFFERENT from {1}: review {2}, then -Accept" -f $what, $from, $pending)
            Compare-Object -ReferenceObject @($previous -split "`n") -DifferenceObject @($body -split "`n") `
                -CaseSensitive | Select-Object -First 40 | ForEach-Object {
                    $sign = if ($_.SideIndicator -eq "=>") { "+" } else { "-" }
                    Write-Output ("  {0} {1}" -f $sign, $_.InputObject) }
        }
        $failures++
    }
}

if ($failures -gt 0) {
    Write-Output ("Host gate: FAIL ({0}){1}" -f $failures, $onlyNote)
    exit 1
}
if ($reverify -gt 0) {
    Write-Output ("Host gate: RE-VERIFY ({0}){1}" -f $reverify, $onlyNote)
    exit 3
}
Write-Output ("Host gate: PASS{0}" -f $onlyNote)
exit 0
