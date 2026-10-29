# The one queue for builds and GPU work: every build and every program that uses the GPU runs through this script, so
# that only one of them runs at a time, each is logged with its time, and a step's GPU work stays within its budget.
#   - A named mutex (Global\RayTracingGpuQueue) serializes the calls, from any checkout or worktree.
#   - Each call appends kind,step,label,seconds,exit,command to the log, out\m4\logs\gpu-time.csv of the main
#     checkout (kind: gpu or build; builds are logged, never counted against the GPU budget).
#   - Before GPU work it waits while the GPU is above 83 degrees C (nvidia-smi), and refuses while a fullscreen
#     application is in front: exit code 4, try again later. Every GPU job is refused then, not only the ones that
#     open a window (-Window marks those): someone may be playing on the same GPU.
#   - A step's GPU work is planned at most a minute: the script warns once the step's logged GPU time passes 45 s, and
#     refuses to start GPU work beyond 55 s (exit code 3) unless approved: -Approved, or for the gate scripts that run
#     this one, the environment variable RAYTRACING_GPU_APPROVED set to the step's number.
#   - A program that can't be found is exit code 2, before anything runs.
#
#   gpu.ps1 -Step 71 -Label ref-cornell -- out\build\x64-release\RayTracingGPU.exe cornell -w 480 ...
#   gpu.ps1 -Step 71 -Build x64-release [-Target RayTracingGPU]     (cmake --build in the Visual Studio environment)
#   gpu.ps1 -Step 71 -Configure x64-release                          (cmake --preset, for a new build folder)
#   gpu.ps1 -Report 71                                               ("k GPU runs, s seconds of GPU")
# The program's output passes through; the exit code is the program's.
#
# The options are read by hand, not by a param() block: PowerShell would take the program's own options (-w, -s...)
# for abbreviations of this script's. They end at "--" or at the first argument that isn't one of them.
$ErrorActionPreference = "Stop"
$Step = -1; $Label = ""; $Kind = "gpu"; $Build = ""; $Target = ""; $Configure = ""; $Report = -1
$Window = $false; $Approved = $false
$Command = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    switch -exact ($a) {
        "--" { $Command = @($args | Select-Object -Skip ($i + 1)); $i = $args.Count; break }
        "-Step" { $Step = [int]$args[++$i]; continue }
        "-Label" { $Label = [string]$args[++$i]; continue }
        "-Kind" { $Kind = [string]$args[++$i]; continue }
        "-Build" { $Build = [string]$args[++$i]; continue }
        "-Target" { $Target = [string]$args[++$i]; continue }
        "-Configure" { $Configure = [string]$args[++$i]; continue }
        "-Report" { $Report = [int]$args[++$i]; continue }
        "-Window" { $Window = $true; continue }
        "-Approved" { $Approved = $true; continue }
        default { $Command = @($args | Select-Object -Skip $i); $i = $args.Count }
    }
}
$Command = @($Command | ForEach-Object { [string]$_ })
if ($Kind -ne "gpu" -and $Kind -ne "build") { throw "gpu.ps1: -Kind is gpu or build, not '$Kind'" }
if ($Step -ge 0 -and $env:RAYTRACING_GPU_APPROVED -eq "$Step") { $Approved = $true }
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# The main checkout holds the log, whichever worktree this copy of the script belongs to
$common = (& git -C $project rev-parse --path-format=absolute --git-common-dir 2>$null)
$main = if ($LASTEXITCODE -eq 0 -and $common) { Split-Path -Parent $common } else { $project }
$logDir = Join-Path $main "out\m4\logs"
$log = Join-Path $logDir "gpu-time.csv"
$budgetWarn = 45
$budgetStop = 55

function Read-Log {
    if (-not (Test-Path -LiteralPath $log)) { return @() }
    return @(Import-Csv -LiteralPath $log)
}

function Get-StepGpuSeconds([int]$n) {
    $sum = 0.0
    foreach ($row in Read-Log) {
        if ($row.kind -eq "gpu" -and [int]$row.step -eq $n) { $sum += [double]$row.seconds }
    }
    return $sum
}

if ($Report -ge 0) {
    $rows = @(Read-Log | Where-Object { [int]$_.step -eq $Report })
    $gpu = @($rows | Where-Object { $_.kind -eq "gpu" })
    $builds = @($rows | Where-Object { $_.kind -eq "build" })
    $gpuSeconds = ($gpu | Measure-Object -Property seconds -Sum).Sum
    $buildSeconds = ($builds | Measure-Object -Property seconds -Sum).Sum
    "Step {0}: {1} GPU runs, {2:N1} seconds of GPU (and {3} builds, {4:N0} s, not counted)" -f $Report, $gpu.Count,
        [double]$gpuSeconds, $builds.Count, [double]$buildSeconds
    exit 0
}

if ($Step -lt 0) { throw "gpu.ps1: -Step is required" }
if ($Build -or $Configure) { $Kind = "build" }
if ($Kind -eq "gpu" -and $Command.Count -eq 0) { throw "gpu.ps1: no program to run (after --)" }

# Whether a fullscreen application (a game, a presentation) is in front: the shell's own answer, and as a check the
# foreground window covering its whole monitor without a title bar
if (-not ("RayTracingScreen" -as [type])) {
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class RayTracingScreen
{
    [DllImport("shell32.dll")] static extern int SHQueryUserNotificationState(out int state);
    [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr window, out RECT rect);
    [DllImport("user32.dll")] static extern IntPtr MonitorFromWindow(IntPtr window, uint flags);
    [DllImport("user32.dll")] static extern bool GetMonitorInfo(IntPtr monitor, ref MONITORINFO info);
    [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr window, int index);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr window,
        System.Text.StringBuilder name, int size);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
    [StructLayout(LayoutKind.Sequential)] public struct MONITORINFO { public int Size; public RECT Monitor, Work;
        public uint Flags; }
    public static bool FullscreenInFront()
    {
        int state;
        // QUNS_BUSY (a fullscreen application), QUNS_RUNNING_D3D_FULL_SCREEN, QUNS_PRESENTATION_MODE
        if (SHQueryUserNotificationState(out state) == 0 && (state == 2 || state == 3 || state == 4)) return true;
        IntPtr window = GetForegroundWindow();
        if (window == IntPtr.Zero) return false;
        var name = new System.Text.StringBuilder(256);
        GetClassName(window, name, name.Capacity);
        string c = name.ToString();
        if (c == "Progman" || c == "WorkerW" || c == "Shell_TrayWnd") return false;
        RECT r;
        if (!GetWindowRect(window, out r)) return false;
        var info = new MONITORINFO();
        info.Size = Marshal.SizeOf(info);
        if (!GetMonitorInfo(MonitorFromWindow(window, 2), ref info)) return false;
        const int WS_CAPTION = 0x00C00000;
        bool covers = r.Left <= info.Monitor.Left && r.Top <= info.Monitor.Top && r.Right >= info.Monitor.Right &&
                      r.Bottom >= info.Monitor.Bottom;
        return covers && (GetWindowLong(window, -16) & WS_CAPTION) != WS_CAPTION;
    }
}
"@
}

function Get-GpuTemperature {
    try {
        $text = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        return [double]::Parse($text.Trim(), [Globalization.CultureInfo]::InvariantCulture)
    }
    catch { return -1 }
}

function Quote([string]$s) {
    if ($s -match '[\s"]') { return '"' + ($s -replace '"', '\"') + '"' }
    return $s
}

$mutex = $null
try { $mutex = New-Object System.Threading.Mutex($false, "Global\RayTracingGpuQueue") }
catch { $mutex = New-Object System.Threading.Mutex($false, "Local\RayTracingGpuQueue") }
$owned = $false
try {
    try { $owned = $mutex.WaitOne() } catch [System.Threading.AbandonedMutexException] { $owned = $true }

    if ($Kind -eq "gpu") {
        if ([RayTracingScreen]::FullscreenInFront()) {
            Write-Output "gpu.ps1: a fullscreen application is in front: no GPU work now (try again later)"
            exit 4
        }
        $spent = Get-StepGpuSeconds $Step
        if ($spent -ge $budgetStop -and -not $Approved) {
            Write-Output ("gpu.ps1: Step {0} has used {1:N1} s of GPU, over its budget: ask before more (-Approved)" -f
                $Step, $spent)
            exit 3
        }
        for ($waited = 0; ; $waited += 30) {
            $t = Get-GpuTemperature
            if ($t -le 83) { break }
            if ($waited -ge 1200) { Write-Output "gpu.ps1: the GPU stayed above 83 degrees C for 20 minutes"; exit 4 }
            Write-Output ("gpu.ps1: the GPU is at {0} degrees C: waiting 30 s" -f $t)
            Start-Sleep -Seconds 30
        }
    }

    $exe = ""
    $arguments = @()
    if ($Build -or $Configure) {
        # The Visual Studio environment (cl.exe, the Windows SDK, Ninja) for this process and the build it starts
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
        Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" |
            Out-Null
        if (-not $env:CUDA_PATH) {
            $cuda = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -Filter "v*" |
                Sort-Object Name | Select-Object -Last 1
            if ($cuda) { $env:CUDA_PATH = $cuda.FullName }
        }
        if ($env:CUDA_PATH) { $env:PATH = "$env:CUDA_PATH\bin;$env:PATH" }
        # Below normal priority: the compilers it starts inherit it, and the desktop stays responsive
        (Get-Process -Id $PID).PriorityClass = "BelowNormal"
        Set-Location -LiteralPath $project
        $exe = "cmake"
        if ($Configure) { $arguments = @("--preset", $Configure); if (-not $Label) { $Label = "configure $Configure" } }
        else {
            $arguments = @("--build", "--preset", $Build)
            if ($Target) { $arguments += @("--target", $Target) }
            if (-not $Label) { $Label = "build $Build" }
        }
        if ($Command.Count -gt 0) { $arguments += $Command }
    }
    else {
        $exe = $Command[0]
        if (Test-Path -LiteralPath $exe -PathType Leaf) { $exe = (Resolve-Path -LiteralPath $exe).Path }
        elseif (-not (Get-Command $exe -CommandType Application -ErrorAction SilentlyContinue)) {
            Write-Output "gpu.ps1: no program '$exe'"
            exit 2
        }
        $arguments = @($Command | Select-Object -Skip 1)
        if (-not $Label) { $Label = [IO.Path]::GetFileNameWithoutExtension($exe) }
    }

    $watch = [Diagnostics.Stopwatch]::StartNew()
    $ErrorActionPreference = "Continue" # a native program's stderr must not stop the script
    $global:LASTEXITCODE = 0
    & $exe @arguments 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    $watch.Stop()

    if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
    if (-not (Test-Path -LiteralPath $log)) {
        Set-Content -LiteralPath $log -Encoding UTF8 -Value "kind,step,label,seconds,exit,command"
    }
    $line = (@(Quote $exe) + @($arguments | ForEach-Object { Quote $_ })) -join " "
    $row = '{0},{1},"{2}",{3},{4},"{5}"' -f $Kind, $Step, ($Label -replace '"', "'"),
        $watch.Elapsed.TotalSeconds.ToString("F3", [Globalization.CultureInfo]::InvariantCulture), $code,
        ($line -replace '"', "'")
    Add-Content -LiteralPath $log -Encoding UTF8 -Value $row

    if ($Kind -eq "gpu") {
        $spent = Get-StepGpuSeconds $Step
        if ($spent -gt $budgetWarn) {
            Write-Output ("gpu.ps1: Step {0} has now used {1:N1} s of GPU (budget 60 s)" -f $Step, $spent)
        }
    }
    exit $code
}
finally {
    if ($owned) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
