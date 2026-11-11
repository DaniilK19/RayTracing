# The panel's gate: what the window does when it is asked by name, by an action or (from Step 101) by its panel, is
# what its keys do, and asking changes no image. Each part runs the viewer under test live, in small windows, through
# the queue (gpu.ps1 -Window: postponed while a fullscreen application is in front), in folders of its own under
# -State (out\m6\runs\stepNN\panel\, never committed). Its parts, each named as -Only names it:
#   actions (Step 100): tests\window-actions-keys.input reaches a state by keys and tests\window-actions.input the
#     same state by actions. A live window splits its samples over frames by the clock, so the two runs' frames needn't
#     agree; what the clock doesn't change must: the recordings' lines other than frames (the settings each key or
#     action led to, in order), their last frame line (the image's passes and view when it was saved), what was saved
#     with its command, and the files P and the save action wrote, byte for byte. The console's Scene, View, Display
#     and Note lines of the actions' run must be lines of the keys' run, in the same order: keys step, and print each
#     state they pass (M past uniform to material), where an action prints the one it names.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\panel.ps1 -Step 100 [-Only actions]
# Exit code 0 when everything matches ("Panel gate: PASS"), 1 when something differs ("Panel gate: FAIL (n)"), 2 for an
# error ("Panel gate: ERROR"), 4 when a live window is postponed ("Panel gate: POSTPONED").
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [string[]]$Only = @(),
    [string]$Build = "out\build\x64-release",
    [string]$State = "out\m6"
)
$ErrorActionPreference = "Stop"
trap {
    Write-Output "Error: $($_.Exception.Message)"
    Write-Output "Panel gate: ERROR"
    exit 2
}
[Console]::OutputEncoding = [Text.Encoding]::UTF8 # the programs write UTF-8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$gpuScript = Join-Path $PSScriptRoot "gpu.ps1"

function Get-FullPath([string]$path) {
    if ([IO.Path]::IsPathRooted($path)) { return $path }
    return Join-Path $project $path
}
$viewer = Join-Path (Get-FullPath $Build) "RayTracingViewer.exe"
if (-not (Test-Path -LiteralPath $viewer)) { throw "panel: no $viewer" }
$run = Join-Path (Get-FullPath $State) ("runs\step{0}\panel" -f $Step)
$script:failures = 0

$allParts = @("actions")
$named = @(($Only -join ",") -split '[,\s]+' | Where-Object { $_ })
foreach ($part in $named) {
    if ($allParts -notcontains $part) { throw "panel: -Only takes $($allParts -join ', '), not '$part'" }
}
$runParts = if ($named.Count -gt 0) { $named } else { $allParts }

function New-Folder([string]$path) {
    if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

function Test-Equal([string]$what, [string[]]$a, [string[]]$b) {
    $same = ($a.Count -eq $b.Count)
    for ($i = 0; $same -and $i -lt $a.Count; $i++) { $same = $a[$i] -ceq $b[$i] }
    if ($same) {
        Write-Output ("  {0}: the same ({1} lines), PASS" -f $what, $a.Count)
    }
    else {
        Write-Output ("  {0}: DIFFERENT, FAIL" -f $what)
        Compare-Object $a $b | ForEach-Object { Write-Output ("    {0} {1}" -f $_.SideIndicator, $_.InputObject) }
        $script:failures++
    }
}

# Every line of PART is a line of WHOLE, in the same order
function Test-Within([string]$what, [string[]]$part, [string[]]$whole) {
    $at = 0
    $missing = @()
    foreach ($line in $part) {
        $found = -1
        for ($i = $at; $i -lt $whole.Count; $i++) {
            if ($whole[$i] -ceq $line) { $found = $i; break }
        }
        if ($found -lt 0) { $missing += $line } else { $at = $found + 1 }
    }
    if ($missing.Count -eq 0) {
        Write-Output ("  {0}: {1} lines, each among the other run's {2}, in order, PASS" -f $what, $part.Count,
            $whole.Count)
    }
    else {
        Write-Output ("  {0}: lines the other run doesn't have, or has earlier, FAIL" -f $what)
        $missing | ForEach-Object { Write-Output "    $_" }
        $script:failures++
    }
}

# The viewer live in a window, in FOLDER (where P saves its views): its console's lines; $null when postponed
function Invoke-Window([string]$label, [string]$folder, [string[]]$arguments) {
    New-Folder $folder
    Push-Location -LiteralPath $folder
    try {
        $all = @("-Step", "$Step", "-Label", $label, "-Window", "--", $viewer) + $arguments
        $lines = @(& powershell -NoProfile -ExecutionPolicy Bypass -File $gpuScript @all)
        $code = $LASTEXITCODE
    }
    finally { Pop-Location }
    if ($code -eq 4) { return $null }
    if ($code -ne 0) {
        $lines | ForEach-Object { Write-Output "  | $_" }
        throw "panel: '$label' ended with exit code $code"
    }
    return , $lines
}

function Get-Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
}

# --- actions: the same state by keys and by actions ------------------------------------------------------------------

function Get-ActionsRun([string]$name, [string]$inputFile) {
    $folder = Join-Path $run $name
    $console = Invoke-Window "window-$name" $folder @("cornell", "-w", "320", "--height", "180", "-s", "32",
        "--input", (Join-Path $project $inputFile), "--record", "record.txt")
    if ($null -eq $console) { return $null }
    $record = @(Get-Content -LiteralPath (Join-Path $folder "record.txt"))
    $frames = @($record | Where-Object { $_ -like "frame *" })
    $last = ""
    if ($frames.Count -gt 0) {
        # Without its generation, which counts every view the clock let the window show
        $last = $frames[-1] -replace ' gen \d+ ', ' gen * '
    }
    return @{
        folder   = $folder
        settings = @($record | Where-Object { $_ -notlike "frame *" })
        last     = $last
        console  = @($console | Where-Object { $_ -match '^(Scene|View|Display|Note|Denoiser|Error):' })
        saved    = @($console | Where-Object { $_ -match '^(Saved |RayTracingGPU )' } |
                ForEach-Object { $_ -replace [regex]::Escape($folder), '<folder>' })
    }
}

if ($runParts -contains "actions") {
    Write-Output "actions: tests\window-actions-keys.input against tests\window-actions.input, each live in a window"
    $keys = Get-ActionsRun "actions-keys" "tests\window-actions-keys.input"
    $acts = if ($null -ne $keys) { Get-ActionsRun "actions" "tests\window-actions.input" } else { $null }
    if ($null -eq $keys -or $null -eq $acts) {
        Write-Output ("actions: postponed: the queue opens no window while a fullscreen application is in front (or " +
            "the GPU is too hot): run it again later")
        Write-Output "Panel gate: POSTPONED"
        exit 4
    }
    Test-Equal "the recordings' settings" $keys.settings $acts.settings
    Test-Equal "the last frame recorded" @($keys.last) @($acts.last)
    Test-Within "the actions' Scene, View, Display and Note lines" $acts.console $keys.console
    Test-Equal "what was saved, and its command" $keys.saved $acts.saved
    foreach ($file in @("light-view-001.png", "light-view-001.pfm")) {
        $a = Join-Path $keys.folder $file
        $b = Join-Path $acts.folder $file
        if (-not (Test-Path -LiteralPath $a) -or -not (Test-Path -LiteralPath $b)) {
            Write-Output "  ${file}: MISSING, FAIL"
            $script:failures++
        }
        elseif ((Get-Sha256 $a) -ceq (Get-Sha256 $b)) {
            Write-Output ("  {0}: the same bytes ({1}), PASS" -f $file, (Get-Sha256 $a).Substring(0, 16))
        }
        else {
            Write-Output "  ${file}: DIFFERENT bytes, FAIL"
            $script:failures++
        }
    }
    Write-Output "  the keys' run, for review:"
    $keys.console | ForEach-Object { Write-Output "    | $_" }
    $keys.settings | ForEach-Object { Write-Output "    r $_" }
    Write-Output "    r $($keys.last)"
}

if ($script:failures -gt 0) {
    Write-Output "Panel gate: FAIL ($($script:failures))"
    exit 1
}
Write-Output "Panel gate: PASS"
exit 0
