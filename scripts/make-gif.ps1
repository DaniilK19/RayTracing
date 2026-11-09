# Makes docs/viewer.gif, the README's GIF of the window: RayTracingViewer renders the camera script docs/viewer-tour.txt
# without a window (--headless) and saves every frame as a PNG (--frames), and gifski (gif.ski) turns them into a GIF of
# 12.5 frames a second. The same frames and the same gifski make the same file, byte for byte; the frames of OptiX and
# the denoiser depend on the graphics driver. gifski is an external program, run but never built into the project
# (`scoop install gifski`, or from gif.ski); the GIF in the repository was made with version 1.34.0.
#
# After building the x64-release preset (paths given are taken from the current folder; the defaults are the
# project's):
#   powershell -ExecutionPolicy Bypass -File scripts\make-gif.ps1 [-Gifski path\to\gifski.exe] [-Output file.gif]
#              [-Frames folder]
# -Frames encodes the f*.png frames of a folder already rendered instead of rendering them: the folder is copied, and
# only the copy gets the last frame's held copies.
param(
    [string]$Gifski = "gifski",
    [string]$Viewer = "",
    [string]$Output = "",
    [string]$Frames = ""
)
$ErrorActionPreference = "Stop"

# Paths from the current folder, before the script moves to the project's: there the scenes find the Earth's texture
$project = [IO.Path]::GetDirectoryName($PSScriptRoot)
if (Test-Path -LiteralPath $Gifski -PathType Leaf) {
    $Gifski = (Resolve-Path -LiteralPath $Gifski).Path
}
else {
    $Gifski = (Get-Command $Gifski -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
}
if ($Viewer) {
    $Viewer = (Resolve-Path -LiteralPath $Viewer).Path
}
else {
    $Viewer = Join-Path $project "out\build\x64-release\RayTracingViewer.exe"
}
if ($Frames) {
    $Frames = (Resolve-Path -LiteralPath $Frames).Path
}
if ($Output) {
    $Output = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
}
else {
    $Output = Join-Path $project "docs\viewer.gif"
}

$version = (& $Gifski --version) -join " "
if ($version -ne "gifski 1.34.0") {
    Write-Warning "$version, not gifski 1.34.0: the GIF may differ from the repository's in its bytes"
}

# The frames: the window's image only (no numbers on screen), one PNG per frame of the script
$work = Join-Path ([IO.Path]::GetTempPath()) ("raytracing-gif-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory $work | Out-Null
Push-Location -LiteralPath $project
try {
    if ($Frames) {
        Copy-Item -Path (Join-Path $Frames "f*.png") -Destination $work
    }
    else {
        & $Viewer --headless --script docs\viewer-tour.txt --frames $work
        if ($LASTEXITCODE -ne 0) { throw "RayTracingViewer failed (exit code $LASTEXITCODE)" }
    }
    $files = @(Get-ChildItem -LiteralPath $work -Filter "f*.png" | Sort-Object Name)
    if ($files.Count -eq 0) { throw "no f*.png frames to encode" }

    # The finished image stays for 2 s at the end: a script's frames each add samples, so the last one is copied 25
    # times, and gifski merges the copies into the GIF's long last frame
    $last = $files[-1]
    for ($i = 1; $i -le 25; $i++) {
        Copy-Item -LiteralPath $last.FullName (Join-Path $work ("f{0:D5}.png" -f ($files.Count - 1 + $i)))
    }
    $names = @(Get-ChildItem -LiteralPath $work -Filter "f*.png" | Sort-Object Name | ForEach-Object { $_.Name })

    # Quality 100 keeps the denoised frames clean (lower ones leave blotches of the frame before) and the GIF under 8 MB.
    # --width 960 is a maximum: it keeps the frames' 960x540 whatever size gifski would choose by itself (about 800x600
    # by default)
    Push-Location -LiteralPath $work
    try {
        & $Gifski --fps 12.5 --quality 100 --motion-quality 100 --width 960 --quiet -o $Output $names
        if ($LASTEXITCODE -ne 0) { throw "gifski failed (exit code $LASTEXITCODE)" }
    }
    finally { Pop-Location }
}
finally {
    Pop-Location
    Remove-Item -LiteralPath $work -Recurse -Force
}

# The GIF's size from its header (bytes 6 to 9: width and height, little-endian), its bytes and their hash
$bytes = [IO.File]::ReadAllBytes($Output)
$width = [BitConverter]::ToUInt16($bytes, 6)
$height = [BitConverter]::ToUInt16($bytes, 8)
$hash = (Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash
"{0}: {1}x{2}, from {3} PNG frames, {4:N2} MB, SHA256 {5}" -f $Output, $width, $height, $names.Count, ($bytes.Length / 1MB), $hash
if ($width -ne 960 -or $height -ne 540) { throw "the GIF is ${width}x$height, not 960x540" }
if ($bytes.Length -gt 8MB) { throw "the GIF is larger than 8 MB" }
