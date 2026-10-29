# Gate F: the files the books' kernels, the display kernel and the CPU programs are built from are unchanged since
# Step 70 (5c2d048), so the frozen renderers can't have changed through their sources. It proves:
#   - the frozen files are byte for byte Step 70's (git diff of the working tree, untracked files included): the CPU
#     modules and programs, the books' device headers, display.cu, the third-party code, the CMake modules and presets,
#     the manifest, the GIF and its script;
#   - renderer.cu still holds, verbatim, the device code of the two books' kernels and the host lines they compile
#     against (block_width/block_height, launch_work, kernel_function): only host code around them may change;
#   - CMakeLists.txt starts with Step 70's lines up to the ImageCompare target, and no later line names one of those
#     targets in a command that changes it (target_*, set_target_properties, set_property) or changes the C++ flags
#     every target reads when the build is generated (CMAKE_CXX_FLAGS and the other CMAKE_CXX_* and CMAKE_MSVC_*
#     variables, add_definitions, add_compile_definitions): the CPU programs are built as they were.
# Not covered here: renderer.h, which the kernels include and later steps extend with host types (the SASS gate proves
# the kernels' code unchanged whatever the headers), and the GPU programs' host code (the host gate).
# It renders nothing and needs no GPU. Exit code 0 when all holds, 1 otherwise.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\frozen.ps1 [-Base 5c2d048]
param([string]$Base = "5c2d048")
$ErrorActionPreference = "Stop"
# git writes UTF-8 (the sources' comments are full of Greek letters and symbols): read it as UTF-8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location -LiteralPath $project
$failures = 0
try {
    $frozen = @(
        "src/aabb.ixx", "src/bvh.ixx", "src/camera.ixx", "src/cli.ixx", "src/constant_medium.ixx", "src/hittable.ixx",
        "src/image.ixx", "src/instance.ixx", "src/interval.ixx", "src/material.ixx", "src/onb.ixx", "src/pdf.ixx",
        "src/perlin.ixx", "src/quad.ixx", "src/random.ixx", "src/ray.ixx", "src/scenes.ixx", "src/sphere.ixx",
        "src/texture.ixx", "src/vec3.ixx", "src/main.cpp", "src/monte_carlo.cpp", "src/image_compare.cpp",
        "src/utf8.manifest", "src/gpu/bvh.cuh", "src/gpu/camera.cuh", "src/gpu/constant_medium.cuh",
        "src/gpu/lights.cuh", "src/gpu/material.cuh", "src/gpu/pdf.cuh", "src/gpu/perlin.cuh", "src/gpu/quad.cuh",
        "src/gpu/random.cuh", "src/gpu/scene.cuh", "src/gpu/sphere.cuh", "src/gpu/texture.cuh", "src/gpu/vec3.cuh",
        "src/gpu/display.cu", "src/gpu/display.cuh", "external", "cmake", "CMakePresets.json", "assets/earthmap.jpg",
        "docs/viewer.gif", "docs/viewer-tour.txt", "scripts/make-gif.ps1")

    # Tracked files changed since the base (committed or not)
    $changed = @(git diff --name-only $Base -- @frozen)
    if ($LASTEXITCODE -ne 0) { throw "git diff failed" }
    # Files added under the frozen folders but not committed
    $untracked = @(git ls-files --others --exclude-standard -- @frozen)
    foreach ($file in $changed + $untracked) {
        Write-Output "CHANGED: $file"
        $failures++
    }
    Write-Output ("frozen files: {0} paths checked, {1} changed" -f $frozen.Count, ($changed.Count + $untracked.Count))

    # Text as git stores it (LF), the working tree's with its line ends made LF too (core.autocrlf)
    function Get-BaseLines([string]$path) {
        $text = (git show "${Base}:$path") -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "git show ${Base}:$path failed" }
        return , ($text -split "`n")
    }
    function Get-WorkingText([string]$path) {
        return ([IO.File]::ReadAllText((Join-Path $project $path))) -replace "`r`n", "`n"
    }

    # renderer.cu: Step 70's lines that must reappear verbatim (1-based, inclusive). String.Contains is ordinal.
    $renderer = Get-WorkingText "src/gpu/renderer.cu"
    $baseRenderer = Get-BaseLines "src/gpu/renderer.cu"
    $spans = @(
        @{ name = "block_width and block_height"; first = 64; last = 65 },
        @{ name = "launch_work"; first = 249; last = 255 },
        @{ name = "the device code, without_nan to all_books_kernel"; first = 389; last = 529 },
        @{ name = "kernel_function"; first = 532; last = 543 })
    foreach ($span in $spans) {
        $text = ($baseRenderer[($span.first - 1)..($span.last - 1)]) -join "`n"
        if ($renderer.Contains($text)) {
            Write-Output ("renderer.cu, {0} (Step 70's lines {1}-{2}): verbatim" -f $span.name, $span.first, $span.last)
        }
        else {
            Write-Output ("CHANGED: renderer.cu, {0} (Step 70's lines {1}-{2})" -f $span.name, $span.first, $span.last)
            $failures++
        }
    }

    # CMakeLists.txt: Step 70's text up to the ImageCompare target's flags is the start of today's
    $cmake = Get-WorkingText "CMakeLists.txt"
    $baseCmake = (Get-BaseLines "CMakeLists.txt") -join "`n"
    $end = "target_compile_options(ImageCompare PRIVATE /W4)"
    $prefix = $baseCmake.Substring(0, $baseCmake.IndexOf($end, [StringComparison]::Ordinal) + $end.Length)
    if ($cmake.StartsWith($prefix, [StringComparison]::Ordinal)) {
        Write-Output "CMakeLists.txt, the CPU section (through the ImageCompare target): verbatim"
        # The rest, without its comments: nothing that changes the CPU targets or the flags they are built with
        $rest = @($cmake.Substring($prefix.Length) -split "`n" | ForEach-Object { ($_ -replace '#.*$', '').Trim() } |
            Where-Object { $_ })
        $targets = '(raytracer|RayTracing|MonteCarlo|ImageCompare|std_module)'
        $commands = '(target_\w+|set_target_properties|set_property|add_dependencies)'
        $touching = @($rest | Where-Object {
                $_ -cmatch "^$commands\s*\(\s*(TARGET\s+)?$targets(\s|\))" -or
                $_ -cmatch '\bCMAKE_(CXX|MSVC|C|EXE_LINKER|STATIC_LINKER)_\w*' -or
                $_ -cmatch '^(add_definitions|add_compile_definitions|add_link_options)\s*\(' })
        foreach ($line in $touching) {
            Write-Output "CHANGED: CMakeLists.txt after the CPU section: $line"
            $failures++
        }
        if ($touching.Count -eq 0) {
            Write-Output "CMakeLists.txt after the CPU section: no command changes the CPU targets or the C++ flags"
        }
    }
    else {
        Write-Output "CHANGED: CMakeLists.txt, the CPU section (through the ImageCompare target)"
        $failures++
    }
}
finally {
    Pop-Location
}
if ($failures -gt 0) {
    Write-Output "Gate F: FAIL ($failures)"
    exit 1
}
Write-Output "Gate F: PASS"
exit 0
