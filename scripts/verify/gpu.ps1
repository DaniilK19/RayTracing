# The one queue for builds and GPU work: every build and every program that uses the GPU runs through this script, so
# that only one of them runs at a time, each is logged with its time, and a step's GPU work stays within its budget.
#   - A named mutex (Global\RayTracingGpuQueue) serializes the calls, from any checkout or worktree.
#   - Each call appends kind,step,label,seconds,exit,command,cpu_seconds,driver to the log, out\m5\logs\gpu-time.csv
#     of the main checkout (kind: gpu or build; builds are logged, never counted against the GPU budget). seconds is
#     the wall time. cpu_seconds is the started process's own processor time (TotalProcessorTime): OptiX compiles its
#     programs on the CPU inside the program's process, so this is where a first compile shows. It counts that process
#     alone, not the ones it starts: for a build, cmake's own time, not the compilers'; under compute-sanitizer, the
#     sanitizer's, not its target's. driver is the graphics driver's version (nvidia-smi), read before each GPU job and
#     empty for builds: it is written to this local log only, never into anything committed.
#   - Every job, build or GPU, runs with OPTIX_CACHE_PATH set to out\m5\optix-cache of the main checkout (created when
#     missing), in the started process's environment only: the queue's runs share one cache of OptiX's compiled
#     programs, so a step pays a changed module's compile once, in its optix-warm run. Programs run outside the queue
#     keep OptiX's own cache.
#   - Before GPU work it waits while the GPU is above 83 degrees C (nvidia-smi), and refuses while a fullscreen
#     application is in front: exit code 4, try again later. Every GPU job is refused then, not only the ones that
#     open a window (-Window marks those): someone may be playing on the same GPU.
#   - A step's GPU work is planned at most a minute: the script warns once the step's logged GPU time passes 45 s, and
#     refuses to start GPU work (exit code 3) beyond 55 s, and also a job that would take the step past 60 s by the
#     last logged duration of a job with its label (in any step; 0 for a label never logged), printing both numbers,
#     since the 55 s are checked before a job starts and one long job could end far past them. Unless approved:
#     -Approved, or for the gate scripts that run this one, the environment variable RAYTRACING_GPU_APPROVED set to
#     the step's number.
#   - A program that can't be found is exit code 2, before anything runs.
#
#   gpu.ps1 -Step 86 -Label ref-cornell -- out\build\x64-release\RayTracingGPU.exe cornell -w 480 ...
#   gpu.ps1 -Step 86 -Build x64-release [-Target RayTracingGPU]     (cmake --build in the Visual Studio environment)
#   gpu.ps1 -Step 86 -Configure x64-release                          (cmake --preset, for a new build folder)
#   gpu.ps1 -Report 86          ("k GPU runs, s seconds of GPU, c of them OptiX's first compile (x s, y s of CPU)", the
#                                c being the runs labelled optix-warm..., then the builds)
# The program's output passes through as lines: its standard output's as they come, then its standard error's. The
# exit code is the program's. Its arguments reach it exactly, spaces and quotes included (quoted as the C runtime reads
# a command line), and it starts in the current folder. A window opens as it would from a console: only the output is
# redirected.
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
$invariant = [Globalization.CultureInfo]::InvariantCulture

# The main checkout holds the log and OptiX's cache, whichever worktree this copy of the script belongs to
$common = (& git -C $project rev-parse --path-format=absolute --git-common-dir 2>$null)
$main = if ($LASTEXITCODE -eq 0 -and $common) { Split-Path -Parent $common } else { $project }
$logDir = Join-Path $main "out\m5\logs"
$log = Join-Path $logDir "gpu-time.csv"
$optixCache = Join-Path $main "out\m5\optix-cache"
$budgetWarn = 45
$budgetStop = 55
$budgetHard = 60

function Read-Log {
    if (-not (Test-Path -LiteralPath $log)) { return @() }
    return @(Import-Csv -LiteralPath $log)
}

# A column's sum over some rows of the log; an empty cell counts 0
function Get-Sum($rows, [string]$column) {
    $sum = 0.0
    foreach ($row in $rows) {
        $cell = $row.$column
        if ($cell) { $sum += [double]$cell }
    }
    return $sum
}

function Get-StepGpuSeconds([int]$n) {
    return Get-Sum @(Read-Log | Where-Object { $_.kind -eq "gpu" -and [int]$_.step -eq $n }) "seconds"
}

# The wall seconds of the last GPU job logged with a label, in any step; 0 for a label never logged
function Get-LastSeconds([string]$label) {
    $logged = $label -replace '"', "'"
    $rows = @(Read-Log | Where-Object { $_.kind -eq "gpu" -and $_.label -eq $logged })
    if ($rows.Count -eq 0 -or -not $rows[-1].seconds) { return 0.0 }
    return [double]$rows[-1].seconds
}

if ($Report -ge 0) {
    $rows = @(Read-Log | Where-Object { [int]$_.step -eq $Report })
    $gpu = @($rows | Where-Object { $_.kind -eq "gpu" })
    $warm = @($gpu | Where-Object { $_.label -like "optix-warm*" })
    $builds = @($rows | Where-Object { $_.kind -eq "build" })
    [string]::Format($invariant, "Step {0}: {1} GPU runs, {2:N1} seconds of GPU, {3} of them OptiX's first compile " +
        "({4:N1} s, {5:N1} s of CPU) (and {6} builds, {7:N0} s, not counted)", $Report, $gpu.Count,
        (Get-Sum $gpu "seconds"), $warm.Count, (Get-Sum $warm "seconds"), (Get-Sum $warm "cpu_seconds"),
        $builds.Count, (Get-Sum $builds "seconds"))
    exit 0
}

if ($Step -lt 0) { throw "gpu.ps1: -Step is required" }
if ($Build -or $Configure) { $Kind = "build" }
if ($Kind -eq "gpu" -and $Command.Count -eq 0) { throw "gpu.ps1: no program to run (after --)" }
# A GPU job's label, as logged, before the budget looks it up: the program's name when none is given
if ($Kind -eq "gpu" -and -not $Label) { $Label = [IO.Path]::GetFileNameWithoutExtension($Command[0]) }

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

# The GPU's temperature and the graphics driver's version, from one call of nvidia-smi (-1 and "" when it fails)
function Get-GpuState {
    $result = @{ temperature = -1.0; driver = "" }
    try {
        $text = & nvidia-smi --query-gpu=temperature.gpu,driver_version --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        $parts = @(([string]$text).Split(","))
        if ($parts.Count -ge 2) { $result.driver = $parts[1].Trim() -replace '[",]', '' }
        $result.temperature = [double]::Parse($parts[0].Trim(), $invariant)
    }
    catch { }
    return $result
}

# One argument of a command line, as the C runtime (and CommandLineToArgvW) reads it back: in quotes when it is empty
# or holds a space or a quote, each quote escaped, and the backslashes before a quote or before the closing quote
# doubled (a lone backslash elsewhere is a backslash)
function Format-Argument([string]$s) {
    if ($s.Length -gt 0 -and $s -notmatch '[\s"]') { return $s }
    $b = New-Object Text.StringBuilder
    [void]$b.Append([char]34)
    $slashes = 0
    foreach ($c in $s.ToCharArray()) {
        if ($c -eq [char]92) { $slashes++; continue }
        if ($c -eq [char]34) { [void]$b.Append([char]92, 2 * $slashes + 1) }
        else { [void]$b.Append([char]92, $slashes) }
        [void]$b.Append($c)
        $slashes = 0
    }
    [void]$b.Append([char]92, 2 * $slashes)
    [void]$b.Append([char]34)
    return $b.ToString()
}

$mutex = $null
try { $mutex = New-Object System.Threading.Mutex($false, "Global\RayTracingGpuQueue") }
catch { $mutex = New-Object System.Threading.Mutex($false, "Local\RayTracingGpuQueue") }
$owned = $false
try {
    try { $owned = $mutex.WaitOne() } catch [System.Threading.AbandonedMutexException] { $owned = $true }

    $driver = ""
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
        # The 55 s are checked before a job starts: a long job could end far past the 60 s, so its last duration counts
        $last = Get-LastSeconds $Label
        if ($spent + $last -gt $budgetHard -and -not $Approved) {
            Write-Output (("gpu.ps1: Step {0} has used {1:N1} s of GPU, and '{2}' took {3:N1} s when last logged: " +
                "together past the {4} s budget: ask before it (-Approved)") -f $Step, $spent, $Label, $last,
                $budgetHard)
            exit 3
        }
        for ($waited = 0; ; $waited += 30) {
            $gpuState = Get-GpuState
            $t = $gpuState.temperature
            if ($t -le 83) { break }
            if ($waited -ge 1200) { Write-Output "gpu.ps1: the GPU stayed above 83 degrees C for 20 minutes"; exit 4 }
            Write-Output ("gpu.ps1: the GPU is at {0} degrees C: waiting 30 s" -f $t)
            Start-Sleep -Seconds 30
        }
        $driver = $gpuState.driver
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
        $arguments = @($Command | Select-Object -Skip 1)
    }
    # The program's file: a path (logged in full), or a name found on the PATH as the console would find it
    $path = ""
    if (Test-Path -LiteralPath $exe -PathType Leaf) { $path = (Resolve-Path -LiteralPath $exe).Path; $exe = $path }
    else {
        $found = Get-Command $exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $found) {
            Write-Output "gpu.ps1: no program '$exe'"
            exit 2
        }
        $path = $found.Path
    }
    if (-not $Label) { $Label = [IO.Path]::GetFileNameWithoutExtension($exe) }
    if (-not (Test-Path -LiteralPath $optixCache)) { New-Item -ItemType Directory -Force $optixCache | Out-Null }

    # Started as a process of its own rather than with PowerShell's call operator, so that its processor time can be
    # read when it ends. Its output is decoded as the console's, as the call operator decodes it (the gate scripts set
    # UTF-8, which the programs write). Standard error is read on another thread while standard output passes through
    # line by line, so neither pipe can fill up and stop the program.
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $path
    $info.Arguments = @($arguments | ForEach-Object { Format-Argument $_ }) -join " "
    $info.WorkingDirectory = (Get-Location -PSProvider FileSystem).ProviderPath
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = [Console]::OutputEncoding
    $info.StandardErrorEncoding = [Console]::OutputEncoding
    $info.EnvironmentVariables["OPTIX_CACHE_PATH"] = $optixCache

    $watch = [Diagnostics.Stopwatch]::StartNew()
    $process = [Diagnostics.Process]::Start($info)
    $stderr = $process.StandardError.ReadToEndAsync()
    while ($null -ne ($text = $process.StandardOutput.ReadLine())) { $text }
    $process.WaitForExit()
    $watch.Stop()
    $reader = New-Object IO.StringReader($stderr.Result)
    while ($null -ne ($text = $reader.ReadLine())) { $text }
    $code = $process.ExitCode
    $cpu = ""
    try { $cpu = $process.TotalProcessorTime.TotalSeconds.ToString("F3", $invariant) } catch { $cpu = "" }
    $process.Dispose()

    if (-not (Test-Path -LiteralPath $logDir)) { New-Item -ItemType Directory -Force $logDir | Out-Null }
    if (-not (Test-Path -LiteralPath $log)) {
        Set-Content -LiteralPath $log -Encoding UTF8 -Value "kind,step,label,seconds,exit,command,cpu_seconds,driver"
    }
    $line = @(@($exe) + $arguments | ForEach-Object { Format-Argument $_ }) -join " "
    $row = '{0},{1},"{2}",{3},{4},"{5}",{6},{7}' -f $Kind, $Step, ($Label -replace '"', "'"),
        $watch.Elapsed.TotalSeconds.ToString("F3", $invariant), $code, ($line -replace '"', "'"), $cpu, $driver
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
