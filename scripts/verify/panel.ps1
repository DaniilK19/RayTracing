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
#   panel (Step 101): tests\window-panel.input shows the panel, then drags and turns the wheel over it: --stats must
#     show one view to the end (the camera never moved), and no frame in the 1.4 s after the image is done (the
#     window, the panel shown, sleeps). tests\window-panel-off.input, the same events without the panel, must move the
#     camera: the check can fail. The last frame of the window as it was (--frames-hud) is named for review.
#   clicks (Step 102): tests\window-panel-clicks.input clicks the panel's controls (a scene, the strategy's and the
#     tone map's boxes, the exposure, the samples, Save) in a 1280x720 window, tests\window-panel-asked.input asks for
#     the same by actions at the same frames: the same settings, the same last frame, the same files saved.
#   drop (Step 103): tests\window-drop.input drops a glTF model on a live window (shown, with OptiX), then README.md
#     and a .glb cut short (each refused with its reason, the model still shown), and saves the model's image.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\panel.ps1 -Step 103 [-Only actions,panel,clicks,drop]
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

$allParts = @("actions", "panel", "clicks", "drop")
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

# Two runs that must end in the same state: the recordings' settings and last frame, what was saved, and its bytes
function Test-SameState($a, $b, [string]$stem = "light") {
    Test-Equal "the recordings' settings" $a.settings $b.settings
    Test-Equal "the last frame recorded" @($a.last) @($b.last)
    Test-Equal "what was saved, and its command" $a.saved $b.saved
    foreach ($file in @("$stem-view-001.png", "$stem-view-001.pfm")) {
        $x = Join-Path $a.folder $file
        $y = Join-Path $b.folder $file
        if (-not (Test-Path -LiteralPath $x) -or -not (Test-Path -LiteralPath $y)) {
            Write-Output "  ${file}: MISSING, FAIL"
            $script:failures++
        }
        elseif ((Get-Sha256 $x) -ceq (Get-Sha256 $y)) {
            Write-Output ("  {0}: the same bytes ({1}), PASS" -f $file, (Get-Sha256 $x).Substring(0, 16))
        }
        else {
            Write-Output "  ${file}: DIFFERENT bytes, FAIL"
            $script:failures++
        }
    }
}

function Get-ActionsRun([string]$name, [string]$inputFile, [string]$width = "320", [string]$height = "180") {
    $folder = Join-Path $run $name
    $console = Invoke-Window "window-$name" $folder @("cornell", "-w", $width, "--height", $height, "-s", "32",
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
    Test-SameState $keys $acts
    Test-Within "the actions' Scene, View, Display and Note lines" $acts.console $keys.console
    Write-Output "  the keys' run, for review:"
    $keys.console | ForEach-Object { Write-Output "    | $_" }
    $keys.settings | ForEach-Object { Write-Output "    r $_" }
    Write-Output "    r $($keys.last)"
}

# --- panel: what happens over the panel is the panel's, and the window sleeps under it -------------------------------

# A live run of an input file with --stats and the frames as the window shows them (--frames-hud): the stats' rows
function Get-PanelRun([string]$name, [string]$inputFile) {
    $folder = Join-Path $run $name
    $console = Invoke-Window "window-$name" $folder @("cornell", "-w", "320", "--height", "180", "-s", "32",
        "--input", (Join-Path $project $inputFile), "--stats", "stats.csv", "--frames", "frames", "--frames-hud")
    if ($null -eq $console) { return $null }
    return @{ folder = $folder; rows = @(Import-Csv -LiteralPath (Join-Path $folder "stats.csv")) }
}

if ($runParts -contains "panel") {
    Write-Output "panel: tests\window-panel.input and tests\window-panel-off.input, each live in a window"
    $on = Get-PanelRun "panel" "tests\window-panel.input"
    $off = if ($null -ne $on) { Get-PanelRun "panel-off" "tests\window-panel-off.input" } else { $null }
    if ($null -eq $on -or $null -eq $off) {
        Write-Output ("panel: postponed: the queue opens no window while a fullscreen application is in front (or " +
            "the GPU is too hot): run it again later")
        Write-Output "Panel gate: POSTPONED"
        exit 4
    }
    $views = @($on.rows | ForEach-Object { $_.generation } | Sort-Object -Unique)
    if ($views.Count -eq 1) {
        Write-Output "  a drag and a wheel step over the panel: one view to the end (generation $($views[0])), PASS"
    }
    else {
        Write-Output "  a drag and a wheel step over the panel: $($views.Count) views, the camera moved, FAIL"
        $script:failures++
    }
    $offViews = @($off.rows | ForEach-Object { $_.generation } | Sort-Object -Unique)
    if ($offViews.Count -gt 1) {
        Write-Output "  the same without the panel: $($offViews.Count) views (the check can fail), PASS"
    }
    else {
        Write-Output "  the same without the panel: one view: the events never reached the camera, FAIL"
        $script:failures++
    }
    $late = @($on.rows | Where-Object { [double]$_.time_ms -ge 2500 -and [double]$_.time_ms -lt 3900 })
    if ($late.Count -eq 0) {
        Write-Output ("  the image done, the panel shown: no frame from 2.5 s to 3.9 s ({0} frames before), PASS" -f
            $on.rows.Count)
    }
    else {
        Write-Output ("  the image done, the panel shown: {0} frames from 2.5 s to 3.9 s, the window doesn't sleep, " +
            "FAIL" -f $late.Count)
        $script:failures++
    }
    $frames = @(Get-ChildItem -LiteralPath (Join-Path $on.folder "frames") -Filter "*.png" | Sort-Object Name)
    if ($frames.Count -gt 0) { Write-Output "  the window as it was, for review: $($frames[-1].FullName)" }
}

# --- clicks: the panel's controls clicked lead where the actions lead -------------------------------------------------

if ($runParts -contains "clicks") {
    Write-Output "clicks: tests\window-panel-clicks.input against tests\window-panel-asked.input, each live in a window"
    $clicked = Get-ActionsRun "clicks" "tests\window-panel-clicks.input" "1280" "720"
    $asked = if ($null -ne $clicked) { Get-ActionsRun "asked" "tests\window-panel-asked.input" "1280" "720" } else { $null }
    if ($null -eq $clicked -or $null -eq $asked) {
        Write-Output ("clicks: postponed: the queue opens no window while a fullscreen application is in front (or " +
            "the GPU is too hot): run it again later")
        Write-Output "Panel gate: POSTPONED"
        exit 4
    }
    Test-SameState $clicked $asked "checker"
    Write-Output "  the clicks' run, for review:"
    $clicked.console | ForEach-Object { Write-Output "    | $_" }
    $clicked.settings | ForEach-Object { Write-Output "    r $_" }
    Write-Output "    r $($clicked.last)"
}

# --- drop: files dropped on the window ---------------------------------------------------------------------------------

if ($runParts -contains "drop") {
    Write-Output "drop: tests\window-drop.input, live in a window"
    $folder = Join-Path $run "drop"
    $console = Invoke-Window "window-drop" $folder @("cornell", "-w", "320", "--height", "180", "-s", "16",
        "--input", (Join-Path $project "tests\window-drop.input"), "--record", "record.txt")
    if ($null -eq $console) {
        Write-Output ("drop: postponed: the queue opens no window while a fullscreen application is in front (or the " +
            "GPU is too hot): run it again later")
        Write-Output "Panel gate: POSTPONED"
        exit 4
    }
    $scenes = @($console | Where-Object { $_ -like "Scene: *" })
    $expect = @(
        @{ what = "the model dropped is shown, with OptiX"; ok = ($scenes.Count -eq 2 -and
                $scenes[1] -like "Scene: *brass_vase_01_1k.gltf, sampling: mis*OptiX*") },
        @{ what = "README.md refused: not a model"; ok = [bool]($console | Where-Object {
                    $_ -ceq "Note: README.md isn't a glTF 2.0 model (.gltf or .glb)" }) },
        @{ what = "cut-short.glb refused by the loader"; ok = [bool]($console | Where-Object {
                    $_ -like "Note: can't show cut-short.glb: *" }) },
        @{ what = "the model still shown when saved"; ok = (Test-Path -LiteralPath (Join-Path $folder "brass_vase_01_1k-view-001.png")) }
    )
    foreach ($e in $expect) {
        if ($e.ok) { Write-Output "  $($e.what): PASS" }
        else { Write-Output "  $($e.what): FAIL"; $script:failures++ }
    }
    Write-Output "  its console, for review:"
    $console | Where-Object { $_ -match '^(Scene|View|Display|Note|Error|Saved|Warning):' } |
        ForEach-Object { Write-Output "    | $_" }
}

if ($script:failures -gt 0) {
    Write-Output "Panel gate: FAIL ($($script:failures))"
    exit 1
}
Write-Output "Panel gate: PASS"
exit 0
