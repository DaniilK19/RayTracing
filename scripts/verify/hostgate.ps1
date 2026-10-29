# The host gate: the GPU programs' host code (options, scene translation, the window's session and scripts, saving)
# behaves as before. The SASS gate proves the kernels unchanged; this proves what they are given and what is done with
# their results. Compared with the previous step's (Step 70's exes in out\m4\bin70 to start with):
#   1. text: the command lines of cli-cases.txt, each ending before it renders: their output and exit code, apart
#      from the differences the step allows (out\m4\cli-allowed\NN.diff, a git diff of the texts);
#   2. the window: tests\regress-viewer.txt rendered headless, every frame's bytes (the display kernel's) and what the
#      window prints (each scene's strategy and precision, each image's samples);
#   3. RayTracingGPU: five small renders (float64; a strategy; rounds of strata with short launches; the lights list;
#      the kernel for all three books on a first-book scene), the PFM's and the PNG's bytes, and the Scene, View,
#      Kernel, mean radiance and rays-traced lines;
#   4. with -Tour: docs\viewer-tour.txt, the README GIF's script, rendered headless: every frame and the --record file;
#   5. tests\regress-tone.txt (from Step 75): the window's tone maps, against the step before's (see below).
# Every render goes through gpu.ps1 (the queue and the GPU log). Results go to out\m4\runs\stepNN\host\.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\hostgate.ps1 -Step 71 [-Tour] [-Accept]
# Texts that differ from the allowed differences fail the gate and are left in out\m4\cli-allowed\NN.pending.diff for
# review. -Accept then accepts exactly those (a run whose differences changed since is refused), and keeps the texts
# as step NN's, which the next step compares with. Run it once at every step's commit, also when nothing differs.
# Exit code 0 when everything matches.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [switch]$Tour,
    [switch]$Accept,
    [string]$Build = "out\build\x64-release",
    [string]$Reference = "out\m4\bin70"
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8 # the programs write UTF-8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
$m4 = Join-Path $project "out\m4"
$run = Join-Path $m4 ("runs\step{0}\host" -f $Step)
$baseline = Join-Path $m4 "baseline"
$failures = 0

function Resolve-Exe([string]$folder, [string]$name) {
    $path = Join-Path $project (Join-Path $folder $name)
    if (-not (Test-Path -LiteralPath $path)) { throw "hostgate: no $path" }
    return $path
}
$newGpu = Resolve-Exe $Build "RayTracingGPU.exe"
$newViewer = Resolve-Exe $Build "RayTracingViewer.exe"
$refGpu = Resolve-Exe $Reference "RayTracingGPU.exe"
$refViewer = Resolve-Exe $Reference "RayTracingViewer.exe"

function New-Folder([string]$path) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    New-Item -ItemType Directory -Force $path | Out-Null
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
    return @(Get-ChildItem -LiteralPath $folder -Filter $filter | Sort-Object Name |
        ForEach-Object { "{0} {1}" -f $_.Name, (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash })
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

$cases = @(Get-Content -LiteralPath (Join-Path $PSScriptRoot "cli-cases.txt") -Encoding UTF8 |
    Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") })
$cliRun = Join-Path $run "cli"
New-Folder (Join-Path $cliRun "ref")
New-Folder (Join-Path $cliRun "new")
$sha1 = [Security.Cryptography.SHA1]::Create()
# The newest accepted texts of an earlier step
$accepted = @(Get-ChildItem -LiteralPath (Join-Path $m4 "cli") -Directory -ErrorAction SilentlyContinue |
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
    # The previous step's text, or the reference exes' (kept in out\m4\baseline\cli)
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
$allowedPath = Join-Path $m4 ("cli-allowed\{0}.diff" -f $Step)
$allowed = ""
if (Test-Path -LiteralPath $allowedPath) {
    $allowed = ([IO.File]::ReadAllText($allowedPath) -replace "`r`n", "`n").TrimEnd()
}
# A run whose differences aren't the allowed ones leaves them here for review; -Accept accepts exactly these
$pending = Join-Path $m4 ("cli-allowed\{0}.pending.diff" -f $Step)
$diffLines = ($diff -split "`n" | Where-Object { $_ }).Count
if ($diff -ceq $allowed) {
    Write-Output ("command lines: {0} cases, {1}" -f $cases.Count,
        $(if ($diff) { "only the allowed differences ($diffLines lines)" } else { "identical" }))
    if ($Accept) {
        $keep = Join-Path $m4 ("cli\step{0}" -f $Step)
        New-Folder $keep
        Copy-Item -Path (Join-Path $cliRun "new\*") -Destination $keep
        Write-Output "  (kept as Step $Step's texts)"
    }
}
elseif ($Accept -and (Test-Path -LiteralPath $pending) -and
    ([IO.File]::ReadAllText($pending) -replace "`r`n", "`n").TrimEnd() -ceq $diff) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $allowedPath) | Out-Null
    Move-Item -LiteralPath $pending $allowedPath -Force
    $keep = Join-Path $m4 ("cli\step{0}" -f $Step)
    New-Folder $keep
    Copy-Item -Path (Join-Path $cliRun "new\*") -Destination $keep
    Write-Output ("command lines: {0} cases, the reviewed differences accepted as Step {1}'s ({2} lines)" -f
        $cases.Count, $Step, $diffLines)
}
else {
    New-Item -ItemType Directory -Force (Split-Path -Parent $pending) | Out-Null
    [IO.File]::WriteAllText($pending, $diff + "`n")
    Write-Output ("command lines: {0} cases, DIFFERENT from the allowed: review {1}, then -Accept{2}" -f $cases.Count,
        $pending, $(if ($Accept) { " (this run's differences aren't the ones reviewed)" } else { "" }))
    $diff -split "`n" | Select-Object -First 60 | ForEach-Object { Write-Output "  $_" }
    $failures++
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
        $lines = @($out | Where-Object { $_ -notmatch '^gpu\.ps1:' } | ForEach-Object { $_ -replace ', [0-9.]+ s$', '' })
    }
    $lines += Get-Hashes $folder "f*.png"
    if ($record) { $lines += "record " + (Get-FileHash -LiteralPath $record -Algorithm SHA256).Hash }
    return , $lines
}

$frames = Get-ScriptFrames $newViewer "regress-viewer" "tests\regress-viewer.txt" (Join-Path $run "regress") ""
$failures += Compare-WithBaseline "window, regress-viewer.txt" $frames "regress-viewer.txt" {
    Get-ScriptFrames $refViewer "regress-viewer (reference)" "tests\regress-viewer.txt" (Join-Path $run "regress-ref") `
        ""
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
$smokeLines = Get-Smokes $newGpu (Join-Path $run "smoke") ""
$failures += Compare-WithBaseline "RayTracingGPU, $($smokes.Count) small renders" $smokeLines "smoke.txt" {
    Get-Smokes $refGpu (Join-Path $run "smoke-ref") " (reference)"
}

# --- 4. The GIF's tour -----------------------------------------------------------------------------------------------

if ($Tour) {
    $tourLines = Get-ScriptFrames $newViewer "tour" "docs\viewer-tour.txt" (Join-Path $run "tour") `
        (Join-Path $run "tour-record.txt")
    $failures += Compare-WithBaseline "window, the GIF's tour (frames and --record)" $tourLines "tour.txt" {
        Get-ScriptFrames $refViewer "tour (reference)" "docs\viewer-tour.txt" (Join-Path $run "tour-ref") `
            (Join-Path $run "tour-ref-record.txt")
    }
}

# --- 5. The window's tone maps, step to step -------------------------------------------------------------------------
# Step 70's window has no tone maps, so tests\regress-tone.txt (from Step 75) is compared with the newest earlier step's
# accepted lines (out\m4\chained\stepNN\regress-tone.txt): its console (times masked), every frame and its --record. A
# difference is left in out\m4\chained\NN.pending.txt for review; -Accept then accepts exactly it, and keeps the lines
# as step NN's, as with the command lines' texts.

$toneScript = "tests\regress-tone.txt"
if (Test-Path -LiteralPath (Join-Path $project $toneScript)) {
    $toneFolder = Join-Path $run "regress-tone"
    $toneRecord = Join-Path $run "regress-tone-record.txt"
    New-Folder $toneFolder
    $toneOut = Invoke-Queued "regress-tone" $newViewer @("--headless", "--script", $toneScript, "--frames", $toneFolder,
        "--record", $toneRecord)
    if ($script:code -ne 0) { $toneOut | ForEach-Object { Write-Host "  $_" }; throw "hostgate: regress-tone failed" }
    $toneLines = @($toneOut | Where-Object { $_ -notmatch '^gpu\.ps1:' } |
        ForEach-Object { $_ -replace ', [0-9.]+ s$', '' })
    $toneLines += Get-Hashes $toneFolder "f*.png"
    $toneLines += "record " + (Get-FileHash -LiteralPath $toneRecord -Algorithm SHA256).Hash
    $toneText = ($toneLines -join "`n").TrimEnd()

    $chained = Join-Path $m4 "chained"
    $toneEarlier = @(Get-ChildItem -LiteralPath $chained -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^step(\d+)$' -and [int]$Matches[1] -lt $Step -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "regress-tone.txt")) } |
        Sort-Object { [int]($_.Name.Substring(4)) } | Select-Object -Last 1)
    $tonePrevious = $null
    $toneFrom = "no earlier step"
    if ($toneEarlier.Count -gt 0) {
        $tonePrevious = ([IO.File]::ReadAllText((Join-Path $toneEarlier[0].FullName "regress-tone.txt")) -replace
            "`r`n", "`n").TrimEnd()
        $toneFrom = "Step " + $toneEarlier[0].Name.Substring(4)
    }
    $tonePending = Join-Path $chained ("{0}.pending.txt" -f $Step)
    $toneKeep = Join-Path $chained ("step{0}" -f $Step)
    if ($null -ne $tonePrevious -and $tonePrevious -ceq $toneText) {
        Write-Output ("window, regress-tone.txt: identical to {0}'s ({1} lines)" -f $toneFrom, $toneLines.Count)
        if ($Accept) {
            New-Item -ItemType Directory -Force $toneKeep | Out-Null
            [IO.File]::WriteAllText((Join-Path $toneKeep "regress-tone.txt"), $toneText + "`n")
        }
    }
    elseif ($Accept -and (Test-Path -LiteralPath $tonePending) -and
        ([IO.File]::ReadAllText($tonePending) -replace "`r`n", "`n").TrimEnd() -ceq $toneText) {
        New-Item -ItemType Directory -Force $toneKeep | Out-Null
        Move-Item -LiteralPath $tonePending (Join-Path $toneKeep "regress-tone.txt") -Force
        Write-Output ("window, regress-tone.txt: the reviewed lines accepted as Step {0}'s ({1} lines)" -f $Step,
            $toneLines.Count)
    }
    else {
        New-Item -ItemType Directory -Force $chained | Out-Null
        [IO.File]::WriteAllText($tonePending, $toneText + "`n")
        Write-Output ("window, regress-tone.txt: DIFFERENT from {0}'s: review {1}, then -Accept" -f $toneFrom,
            $tonePending)
        if ($null -eq $tonePrevious) {
            $toneLines | Select-Object -First 40 | ForEach-Object { Write-Output "  + $_" }
        }
        else {
            Compare-Object -ReferenceObject @($tonePrevious -split "`n") -DifferenceObject @($toneText -split "`n") `
                -CaseSensitive | Select-Object -First 40 | ForEach-Object {
                    $sign = if ($_.SideIndicator -eq "=>") { "+" } else { "-" }
                    Write-Output ("  {0} {1}" -f $sign, $_.InputObject) }
        }
        $failures++
    }
}

if ($failures -gt 0) {
    Write-Output "Host gate: FAIL ($failures)"
    exit 1
}
Write-Output "Host gate: PASS"
exit 0
