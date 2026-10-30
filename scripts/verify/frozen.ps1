# Gate F: the files the frozen kernels and the CPU programs are built from are unchanged, so those can't have changed
# through their sources. It proves:
#   - for each (base, paths) pair below, the files under the paths are byte for byte the base commit's (git diff of the
#     working tree against the base; an untracked file under a frozen path fails too):
#       5c2d048 (Step 70): the CPU modules and programs, the books' device headers, display.cu, stb (external/stb), the
#         SDL3 module (cmake/sdl3.cmake), the CMake presets, the manifest, the README's GIF, its script and its tour;
#       65f297b (Step 85): every header beyond the books' that the quality and
#         tone-map kernels compile, the tone maps, the lamp and environment tables, the nine test suites of milestone 4
#         and the headers and regression lists they share, the HDR environment maps;
#   - the blob pins below: a file frozen by the step that wrote it (a commit can't name its own hash) is still that
#     blob: `git hash-object` of the working file, through git's own line-end filters, equals the pinned SHA-1; a
#     pinned file that is missing or untracked fails too;
#   - renderer.cu still holds, verbatim, the device code of the two books' kernels and the host lines they compile
#     against (block_width/block_height, launch_work, kernel_function): only host code around them may change;
#   - CMakeLists.txt starts with Step 70's lines up to the ImageCompare target, and with all of Step 85's (212) lines.
#     The commands after them, and those of every CMake module under cmake/ but sdl3.cmake, are read as CMake reads
#     them: a command whole however many lines it spans (and several on one line), its comments dropped, its name in any
#     case, each argument with its quotes or bracket delimiters taken off ("RayTracing", [[RayTracing]] and RayTracing
#     are one name) and an unquoted one split at its semicolons; a text whose commands can't be read (a parenthesis,
#     quote or bracket left open) fails. After Step 70's section, and in those modules, no command
#       - changes a CPU target (raytracer, RayTracing, MonteCarlo, ImageCompare, std_module; in any case): named as a
#         target_* command's first argument, as any target of set_target_properties or set_property(TARGET ...), as any
#         argument of add_dependencies, or after TARGET in add_custom_command;
#       - sets source properties (set_source_files_properties, set_property(SOURCE ...)) of a CPU source: a path with a
#         src/ folder not followed by gpu/ (backslashes, "." and ".." resolved), or the file name of std.ixx,
#         utf8.manifest or any file under src/ outside src/gpu/; in any case, as CMake on Windows compares paths;
#       - writes a variable the C++ targets read when the build is generated, CMAKE_CXX_*, CMAKE_C_*, CMAKE_MSVC_*,
#         CMAKE_EXE_LINKER_* or CMAKE_STATIC_LINKER_* (what a write is: below), calls add_definitions,
#         add_compile_definitions or add_link_options, or sets a global RULE_LAUNCH_* property (a launcher in front of
#         every compile or link);
#       - sets directory-wide include paths or definitions, which reach the targets already defined too
#         (include_directories, remove_definitions, set_directory_properties, set_property(DIRECTORY ...)):
#     the CPU programs are built as they were.
#     And after Step 85's lines, and in those modules, no command touches the renderer's existing translation units:
#     the source properties of renderer.cu, display.cu, quality.cu, tonemap.cu, lamp_table.cu or environment_table.cu
#     (by file name, in any case, whatever path surrounds it), gpu_renderer (named as a CPU target is above), or a write
#     of a CMAKE_CUDA_* variable. target_sources(gpu_renderer PRIVATE ...) is allowed: new translation units, which the
#     SASS gate shows leave the existing kernels alone.
#     A list of targets or sources is read up to the keyword that ends it in CMake, spelled exactly (CMake's keywords
#     are case-sensitive: a "properties" is one more name): PROPERTIES for set_target_properties and
#     set_source_files_properties (whose old form, without it, is read whole), APPEND, APPEND_STRING or PROPERTY for
#     set_property (a source list's DIRECTORY or TARGET_DIRECTORY part is read too, the stricter way). A write of a
#     variable is an argument that is its name written out, or begins with it (CACHE{ allowed before it), in any command
#     but a condition (if, elseif, else, endif, while, endwhile) or message(), and anywhere but in the list or path that
#     list(GET|LENGTH|FIND|JOIN|SUBLIST ...) and cmake_path(GET|HAS_*|IS_*|HASH|NATIVE_PATH ...) read by its name and in
#     foreach's lists: set, unset, string(APPEND ...), list(APPEND ...), separate_arguments, set_property(CACHE|GLOBAL
#     ...) and every other command that takes a variable's name; a read, ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}, is
#     none.
#     Not proved, since the text doesn't spell it out: a target, source or variable named through a variable (${name},
#     CMAKE_${lang}_FLAGS) or a generator expression, and a command run through cmake_language (CALL, EVAL, DEFER). Nor
#     CMake code outside CMakeLists.txt and cmake/*.cmake (a file included from elsewhere, a subdirectory, fetched
#     content).
# Not covered here: renderer.h, which the kernels include and later steps extend with host types, and the headers of
# the test harness (gpu_test.h, host_services.h): the SASS gate proves every kernel's code unchanged whatever they
# declare. The GPU programs' host code is the host gate's.
# It renders nothing and needs no GPU. Exit code 0 when all holds, 1 otherwise.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\frozen.ps1 [-Root DIR] [-SelfTest]
# -Root checks another checkout of the project (a worktree's top folder) with this script's rules. -SelfTest checks the
# gate itself: in a temporary detached worktree of HEAD (out\m5\runs\frozen-selftest, never this checkout, where other
# processes build), it plants each case of the self-test table below alone, runs the checks on it as -Root would, and
# puts the worktree back. Every fault must fail; the worktree as checked out, a file beside a frozen folder and the
# CMake lines later steps write (a library linked into gpu_scene, a source added to GpuTests, a read of a CMAKE_CUDA_*
# variable, OptiX's lines) must pass. It prints expected and got per case and exits 0 only when every case
# behaved; the worktree is removed at the end, after an error too.
param(
    [string]$Root = "",
    [switch]$SelfTest)
$ErrorActionPreference = "Stop"
# git writes UTF-8 (the sources' comments are full of Greek letters and symbols): read it as UTF-8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# --- What is frozen ---------------------------------------------------------------------------------------------------

# (base, paths): tracked files changed since the base, and untracked files under the paths, fail
$frozenSets = @(
    @{
        base = "5c2d048"; name = "Step 70"
        paths = @(
            "src/aabb.ixx", "src/bvh.ixx", "src/camera.ixx", "src/cli.ixx", "src/constant_medium.ixx",
            "src/hittable.ixx", "src/image.ixx", "src/instance.ixx", "src/interval.ixx", "src/material.ixx",
            "src/onb.ixx", "src/pdf.ixx", "src/perlin.ixx", "src/quad.ixx", "src/random.ixx", "src/ray.ixx",
            "src/scenes.ixx", "src/sphere.ixx", "src/texture.ixx", "src/vec3.ixx", "src/main.cpp",
            "src/monte_carlo.cpp", "src/image_compare.cpp", "src/utf8.manifest", "src/gpu/bvh.cuh",
            "src/gpu/camera.cuh", "src/gpu/constant_medium.cuh", "src/gpu/lights.cuh", "src/gpu/material.cuh",
            "src/gpu/pdf.cuh", "src/gpu/perlin.cuh", "src/gpu/quad.cuh", "src/gpu/random.cuh", "src/gpu/scene.cuh",
            "src/gpu/sphere.cuh", "src/gpu/texture.cuh", "src/gpu/vec3.cuh", "src/gpu/display.cu",
            "src/gpu/display.cuh", "external/stb", "cmake/sdl3.cmake", "CMakePresets.json", "assets/earthmap.jpg",
            # Until Step 97 remakes them and moves them to blob pins
            "docs/viewer.gif", "docs/viewer-tour.txt", "scripts/make-gif.ps1")
    },
    @{
        base = "65f297b"; name = "Step 85"
        paths = @(
            "src/gpu/quality.cuh", "src/gpu/quality_view.cuh", "src/gpu/quality_path.cuh", "src/gpu/tracer.cuh",
            "src/gpu/shadow.cuh", "src/gpu/bsdf.cuh", "src/gpu/ggx.cuh", "src/gpu/ggx_energy.cuh", "src/gpu/mis.cuh",
            "src/gpu/light_sampler.cuh", "src/gpu/lamps.cuh", "src/gpu/background.cuh", "src/gpu/environment.cuh",
            "src/gpu/environment_data.h", "src/gpu/lamps_on_device.cuh", "src/gpu/environment_on_device.cuh",
            "src/gpu/microfacets_on_device.cuh", "src/gpu/tonemap.cu", "src/gpu/tonemap.cuh", "src/gpu/tone_curves.cuh",
            "src/gpu/lamp_table.cu", "src/gpu/environment_table.cu", "src/gpu/device_memory.h",
            "tests/bsdf_tests.cu", "tests/env_tests.cu", "tests/ggx_tests.cu", "tests/gloss_tests.cu",
            "tests/harness_tests.cu", "tests/lamp_tests.cu", "tests/routing_tests.cu", "tests/services_tests.cu",
            "tests/tone_tests.cu", "tests/gpu_test.cuh", "tests/statistics.h", "tests/test_scene.h",
            "tests/regress-viewer.txt", "tests/regress-tone.txt", "assets/*.hdr")
    })

# ---- BLOB PINS: (path, git blob SHA-1 of the file as committed) ------------------------------------------------------
# A step that freezes its own output pins it in its own commit (`git hash-object PATH` after its last edit); this
# script is pinned nowhere, so writing the ids changes no pinned file. Step 97 adds docs/viewer.gif,
# docs/viewer-tour.txt and scripts/make-gif.ps1 (taking them out of the Step 70 list above).
$blobPins = @(
    # Step 87: quality.cu with the one accessor OptiX's programs and the denoiser's guides read the quality kernel's view
    # through (host code: the SASS gate shows quality_kernel unchanged), and that accessor's header
    @{ path = "src/gpu/quality.cu"; blob = "f759c49a631608a432d03d779459a1f699087b31" },
    @{ path = "src/gpu/quality_access.cuh"; blob = "8b6ac3b37ca7e4d2cfa5dbd2d576e683e5713ff3" }
)
# ---- end of BLOB PINS ------------------------------------------------------------------------------------------------

# renderer.cu: Step 70's lines that must reappear verbatim (1-based, inclusive)
$rendererSpans = @(
    @{ name = "block_width and block_height"; first = 64; last = 65 },
    @{ name = "launch_work"; first = 249; last = 255 },
    @{ name = "the device code, without_nan to all_books_kernel"; first = 389; last = 529 },
    @{ name = "kernel_function"; first = 532; last = 543 })

# CMake: the bases of the two prefixes, and what the rules for the commands after them look for (the rules themselves
# are Get-CpuBreaks and Get-GpuBreaks below)
$cmakeStep70 = "5c2d048"
$cmakeStep70End = "target_compile_options(ImageCompare PRIVATE /W4)"
$cmakeStep85 = "65f297b"
# After Step 70's CPU section, and in every module: nothing that changes the CPU targets, their sources' properties or
# the flags they're built with. The targets are compared in any case (-contains); the CPU sources are matched by path
# and by these names, to which Test-Frozen adds the name of every file under src/ outside src/gpu/; the variables are
# case-sensitive regular expressions on a written name, as CMake's variable names are
$cpuTargets = @("raytracer", "RayTracing", "MonteCarlo", "ImageCompare", "std_module")
$cpuSourceNames = @("std.ixx", "utf8.manifest")
$cpuVariables = '^(CACHE\{)?CMAKE_(CXX|MSVC|C|EXE_LINKER|STATIC_LINKER)(?![A-Za-z0-9])'
# After Step 85's lines, and in every module: nothing that changes the renderer's existing translation units (a file
# name in any case, wherever it stands in the path), gpu_renderer, or the CUDA flags
$frozenUnits = '(?<![\w.-])(renderer|display|quality|tonemap|lamp_table|environment_table)\.cu(?![\w.])'
$gpuTarget = "gpu_renderer"
$cudaVariables = '^(CACHE\{)?CMAKE_CUDA(?![A-Za-z0-9])'

# ---- SELF-TEST CASES: each planted alone in the worktree W, which is put back to HEAD after each ---------------------
# A later step adds its own cases here: Step 97 "a byte of viewer.gif changed" (FAIL).
# $selfTestPin is pinned at its HEAD blob for the self-test only, to show the pins' mechanism on a file no other rule
# covers.
$selfTestPin = "README.md"
$selfTestCases = @(
    @{ name = "the worktree as checked out"; expect = "PASS"; plant = { param($w) } },
    @{ name = "a line appended to src/gpu/quality.cu, after the accessor (its blob pin)"; expect = "FAIL"
       plant = { param($w) Add-LineTo $w "src/gpu/quality.cu" "// planted by frozen.ps1 -SelfTest" } },
    @{ name = "one byte of src/gpu/quality_access.cuh changed (its blob pin)"; expect = "FAIL"
       plant = { param($w) Edit-OneByte $w "src/gpu/quality_access.cuh" } },
    @{ name = "one byte of $selfTestPin changed (blob-pinned for the self-test)"; expect = "FAIL"
       plant = { param($w) Edit-OneByte $w $selfTestPin } },
    @{ name = "an untracked external/stb/x.h"; expect = "FAIL"
       plant = { param($w) New-FileIn $w "external/stb/x.h" "// planted by frozen.ps1 -SelfTest`n" } },
    @{ name = "an untracked external/x.h (outside external/stb)"; expect = "PASS"
       plant = { param($w) New-FileIn $w "external/x.h" "// planted by frozen.ps1 -SelfTest`n" } }
)
# CMake cases: the text appended to CMakeLists.txt (file "CMakeLists.txt"), or alone in a new module (file
# "cmake/x.cmake"); named "<file>: <text>" unless a name is given
$cmakeCases = @(
    @{ expect = "FAIL"; file = "cmake/x.cmake"; text = 'add_compile_definitions(X)' },
    # (Step 86 review) the CPU rules: a quoted or bracketed target, a CPU target after another one, a CPU source's
    # properties, a C++ flag variable
    @{ expect = "FAIL"; file = "CMakeLists.txt"; text = 'target_compile_definitions("RayTracing" PRIVATE X)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"
       text = 'set_property(TARGET "ImageCompare" APPEND PROPERTY COMPILE_OPTIONS /fp:fast)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"; text = 'target_compile_options([[MonteCarlo]] PRIVATE /Od)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"
       text = 'set_target_properties(PfmStats RayTracing PROPERTIES COMPILE_OPTIONS /Od)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"
       text = 'set_property(TARGET gpu_scene raytracer APPEND PROPERTY COMPILE_DEFINITIONS X)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"
       text = 'set_source_files_properties(src/main.cpp PROPERTIES COMPILE_DEFINITIONS X)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"
       text = 'set_property(SOURCE src/vec3.ixx APPEND PROPERTY COMPILE_OPTIONS /Od)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"; text = 'string(APPEND CMAKE_CXX_FLAGS " /fp:fast")' },
    # The renderer's rules: a CUDA flag variable however it's written, a quoted or second gpu_renderer, a frozen unit's
    # properties in another case or through a quoted scope
    @{ expect = "FAIL"; file = "CMakeLists.txt"
       text = 'set_source_files_properties(src/gpu/quality.cu PROPERTIES COMPILE_OPTIONS -fmad=false)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"; text = 'string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/fp:fast")' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"; text = 'list(APPEND CMAKE_CUDA_FLAGS -lineinfo)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"; text = 'target_compile_options("gpu_renderer" PRIVATE -O0)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"
       text = 'set_target_properties(gpu_optix gpu_renderer PROPERTIES CUDA_ARCHITECTURES 75)' },
    @{ expect = "FAIL"; file = "CMakeLists.txt"
       text = 'set_source_files_properties(src/gpu/Quality.cu PROPERTIES COMPILE_OPTIONS -fmad=false)' },
    @{ expect = "FAIL"; file = "cmake/x.cmake"
       text = 'set_property("SOURCE" src/gpu/quality.cu APPEND PROPERTY COMPILE_OPTIONS -fmad=false)' },
    # What later steps write, which must pass
    @{ expect = "PASS"; file = "CMakeLists.txt"; text = 'target_link_libraries(gpu_scene PUBLIC mesh_io)' },
    @{ expect = "PASS"; file = "cmake/x.cmake"; text = 'target_sources(GpuTests PRIVATE tests/x.cpp)' },
    @{ expect = "PASS"; file = "CMakeLists.txt"
       text = 'target_include_directories(GpuTests PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})' },
    @{ expect = "PASS"; file = "cmake/x.cmake"
       name = "cmake/x.cmake: OptiX's lines (cmake/optix.cmake's kind), and nvcc's folder read from CMAKE_CUDA_COMPILER"
       text = (@(
           'if(CMAKE_CUDA_COMPILER AND optix_include)',
           '    cmake_path(GET CMAKE_CUDA_COMPILER PARENT_PATH nvcc_folder)',
           '    add_library(optix_programs OBJECT src/gpu/optix/programs.cu)',
           '    set_target_properties(optix_programs PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES OFF)',
           '    target_compile_options(optix_programs PRIVATE --gpu-architecture=compute_75)',
           '    add_library(gpu_optix STATIC src/gpu/optix/runtime.cpp src/gpu/optix/scene.cu',
           '        "${CMAKE_CURRENT_BINARY_DIR}/optix/optix_programs_ptx.cpp")',
           '    add_dependencies(gpu_optix optix_programs)',
           '    target_link_libraries(gpu_optix PUBLIC gpu_renderer PRIVATE optix_headers)',
           '    target_include_directories(gpu_optix SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})',
           '    target_compile_definitions(gpu_optix PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)',
           '    target_compile_options(gpu_optix PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4> ' +
               '$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)',
           'endif()',
           'target_link_libraries(gpu_scene PUBLIC gpu_optix)',
           'target_sources(gpu_renderer PRIVATE src/gpu/guides.cu src/gpu/denoise.cu)') -join "`n") }
)
foreach ($case in $cmakeCases) {
    if (-not $case.name) { $case.name = "{0}: {1}" -f $case.file, $case.text }
}
$selfTestCases += $cmakeCases
# ---- end of SELF-TEST CASES ------------------------------------------------------------------------------------------

# --- The checks -------------------------------------------------------------------------------------------------------

# git in the checkout DIR, its output lines; a failure throws. git's warnings on stderr pass through without stopping
# the script (Windows PowerShell turns them into errors when the console's output is redirected); core.safecrlf=false
# only silences the line-end warnings, the conversions stay git's.
function Invoke-Git([string]$Dir, [string[]]$Arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $output = & git -C $Dir -c core.safecrlf=false @Arguments }
    finally { $ErrorActionPreference = $saved }
    if ($LASTEXITCODE -ne 0) {
        throw ("git {0} failed in {1} (exit code {2})" -f ($Arguments -join " "), $Dir, $LASTEXITCODE)
    }
    if ($null -ne $output) { $output }
}

# The top folder of the checkout DIR is in, as a full Windows path
function Get-TopLevel([string]$Dir) {
    $top = @(Invoke-Git $Dir @("rev-parse", "--show-toplevel"))[0]
    return [IO.Path]::GetFullPath($top)
}

# A file's text as git stores it (LF): at the base, and in the checkout with its line ends made LF (core.autocrlf)
function Get-BaseLines([string]$Dir, [string]$Base, [string]$Path) {
    $lines = @(Invoke-Git $Dir @("show", "${Base}:$Path"))
    return , $lines
}
function Get-WorkingText([string]$Dir, [string]$Path) {
    return ([IO.File]::ReadAllText((Join-Path $Dir $Path))) -replace "`r`n", "`n"
}

# A CMake text's commands, read as CMake reads them (the CMake language's syntax: quoted and bracket arguments,
# escapes, line and bracket comments), so a command split over lines or sharing one is seen whole. Each is { name,
# arguments, text }: its name in lower case (CMake reads a command's name in any case); its arguments as CMake hands
# them to the command, quotes and bracket delimiters taken off ("RayTracing", [[RayTracing]] and RayTracing are the
# same argument), escapes resolved, an unquoted argument split at its semicolons and a nested parenthesis an argument
# of its own; and the two on one line, for the report. $null when the text ends inside a command, a quote or a
# bracket, or closes a parenthesis it didn't open.
$bracketOpen = [regex]'\G\[(=*)\['
function Get-CMakeCommands([string]$Text) {
    $commands = New-Object System.Collections.Generic.List[object]
    $head = New-Object System.Text.StringBuilder        # what precedes a command's "(", its name last
    $arguments = New-Object System.Collections.Generic.List[string]
    $word = New-Object System.Text.StringBuilder        # the argument being read
    $quoted = $false                                     # it has a quoted or bracket part (an empty one counts)
    $depth = 0
    $i = 0
    $n = $Text.Length
    while ($i -lt $n) {
        $c = $Text[$i]
        if ($depth -eq 0) {
            # Between commands: a name, white space, comments
            if ($c -eq [char]'#') {
                $i = Skip-CMakeComment $Text $i
                if ($i -lt 0) { return $null }
                [void]$head.Append(" ")
            }
            elseif ($c -eq [char]'(') {
                $depth = 1
                $arguments.Clear()
                $i++
            }
            elseif ($c -eq [char]')') { return $null }
            else {
                [void]$head.Append($c)
                $i++
            }
            continue
        }
        if ($c -eq [char]'\') {
            # An escape: the next character is no syntax (a line end after it continues the line)
            if ($i + 1 -lt $n -and $Text[$i + 1] -ne [char]10) { [void]$word.Append($Text[$i + 1]) }
            $i += 2
            continue
        }
        if ($c -eq [char]'"') {
            # A quoted argument (or a quoted part of one), up to the next unescaped quote, line ends and all; never
            # split
            $j = $i + 1
            while ($j -lt $n -and $Text[$j] -ne [char]'"') {
                if ($Text[$j] -eq [char]'\') {
                    $j++
                    if ($j -lt $n -and $Text[$j] -ne [char]10) { [void]$word.Append($Text[$j]) }
                }
                else { [void]$word.Append($Text[$j]) }
                $j++
            }
            if ($j -ge $n) { return $null }
            $quoted = $true
            $i = $j + 1
            continue
        }
        if ($c -eq [char]'[' -and $word.Length -eq 0 -and -not $quoted) {
            $bracket = $bracketOpen.Match($Text, $i)
            if ($bracket.Success) {
                # A bracket argument, [[ ... ]] or [=[ ... ]=]: nothing inside is syntax, and a line end right after
                # the opening is no part of it
                $close = "]" + $bracket.Groups[1].Value + "]"
                $start = $i + $bracket.Length
                $end = $Text.IndexOf($close, $start, [StringComparison]::Ordinal)
                if ($end -lt 0) { return $null }
                if ($start -lt $end -and $Text[$start] -eq [char]10) { $start++ }
                [void]$word.Append($Text, $start, $end - $start)
                $quoted = $true
                $i = $end + $close.Length
                continue
            }
        }
        $ends = [char]::IsWhiteSpace($c) -or $c -eq [char]'#' -or $c -eq [char]';' -or $c -eq [char]'(' -or
                $c -eq [char]')'
        if (-not $ends) {
            [void]$word.Append($c)
            $i++
            continue
        }
        # The argument read so far is whole (an unquoted argument's empty parts are none, as in CMake)
        if ($word.Length -gt 0 -or $quoted) {
            $arguments.Add($word.ToString())
            [void]$word.Clear()
            $quoted = $false
        }
        if ($c -eq [char]'#') {
            $i = Skip-CMakeComment $Text $i
            if ($i -lt 0) { return $null }
            continue
        }
        if ($c -eq [char]'(') {
            $depth++
            $arguments.Add("(")
        }
        elseif ($c -eq [char]')') {
            $depth--
            if ($depth -gt 0) { $arguments.Add(")") }
            else {
                $name = @(($head.ToString().Trim()) -split '\s+')[-1]
                $list = $arguments.ToArray()
                $line = "{0}({1})" -f $name, ($list -join " ")
                $commands.Add([pscustomobject]@{ name = $name.ToLowerInvariant(); arguments = $list; text = $line })
                [void]$head.Clear()
            }
        }
        $i++
    }
    if ($depth -ne 0) { return $null }
    return , $commands.ToArray()
}

# The index after the comment that starts at I (a "#"): a bracket comment, #[[ ... ]] or #[=[ ... ]=], or a line comment
# up to its line end; -1 when a bracket comment is left open
function Skip-CMakeComment([string]$Text, [int]$I) {
    $bracket = $bracketOpen.Match($Text, $I + 1)
    if ($bracket.Success) {
        $close = "]" + $bracket.Groups[1].Value + "]"
        $end = $Text.IndexOf($close, $bracket.Index + $bracket.Length, [StringComparison]::Ordinal)
        if ($end -lt 0) { return -1 }
        return $end + $close.Length
    }
    $end = $Text.IndexOf([char]10, $I)
    if ($end -lt 0) { return $Text.Length }
    return $end
}

# A command's arguments from FIRST up to the first of the keywords STOP, compared as CMake compares them: exactly
function Get-ArgumentsUntil([string[]]$Arguments, [int]$First, [string[]]$Stop) {
    $list = New-Object System.Collections.Generic.List[string]
    for ($k = $First; $k -lt $Arguments.Count; $k++) {
        if ($Stop -ccontains $Arguments[$k]) { break }
        $list.Add($Arguments[$k])
    }
    return , $list.ToArray()
}

# The names in a list of arguments: each argument split at its semicolons (a quoted one too, the stricter way: some
# commands split it themselves)
function Split-Names([string[]]$Arguments) {
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($argument in $Arguments) {
        foreach ($name in $argument.Split([char]';')) { if ($name) { $names.Add($name) } }
    }
    return , $names.ToArray()
}

# The targets a command changes: a target_* command's first argument, the targets of set_target_properties (before
# PROPERTIES) and of set_property(TARGET ...) (before APPEND, APPEND_STRING or PROPERTY), every argument of
# add_dependencies, the one after TARGET in add_custom_command
function Get-ChangedTargets($Command) {
    $arguments = $Command.arguments
    $names = @()
    if ($Command.name -match '^target_\w+$') {
        if ($arguments.Count -gt 0) { $names = @($arguments[0]) }
    }
    elseif ($Command.name -eq "set_target_properties") {
        $names = Get-ArgumentsUntil $arguments 0 @("PROPERTIES")
    }
    elseif ($Command.name -eq "set_property") {
        if ($arguments.Count -gt 0 -and $arguments[0] -ieq "TARGET") {
            $names = Get-ArgumentsUntil $arguments 1 @("APPEND", "APPEND_STRING", "PROPERTY")
        }
    }
    elseif ($Command.name -eq "add_dependencies") {
        $names = $arguments
    }
    elseif ($Command.name -eq "add_custom_command") {
        for ($k = 0; $k + 1 -lt $arguments.Count; $k++) {
            if ($arguments[$k] -ieq "TARGET") { $names += $arguments[$k + 1] }
        }
    }
    return , (Split-Names $names)
}

# The source files a command sets properties of: those of set_source_files_properties (before PROPERTIES, or all of its
# arguments in its old form without it) and of set_property(SOURCE ...) (before APPEND, APPEND_STRING or PROPERTY; a
# DIRECTORY or TARGET_DIRECTORY part is read as files too, the stricter way)
function Get-PropertySources($Command) {
    $arguments = $Command.arguments
    $files = @()
    if ($Command.name -eq "set_source_files_properties") {
        $files = Get-ArgumentsUntil $arguments 0 @("PROPERTIES")
    }
    elseif ($Command.name -eq "set_property" -and $arguments.Count -gt 0 -and $arguments[0] -ieq "SOURCE") {
        $files = Get-ArgumentsUntil $arguments 1 @("APPEND", "APPEND_STRING", "PROPERTY")
    }
    return , (Split-Names $files)
}

# Whether FILE, a source as a command names it, is a CPU source: its path (backslashes made slashes, "." and "x/.."
# taken out) has a src/ folder not followed by gpu/, or it holds a CPU source's file name (NAMES, a regular expression);
# in any case, as CMake on Windows compares paths
function Test-CpuSource([string]$File, [string]$Names) {
    $parts = New-Object System.Collections.Generic.List[string]
    foreach ($part in $File.Replace([char]'\', [char]'/').Split([char]'/')) {
        if ($part -eq "." -or ($part -eq "" -and $parts.Count -gt 0)) { continue }
        if ($part -eq ".." -and $parts.Count -gt 0 -and $parts[$parts.Count - 1] -ne "..") {
            $parts.RemoveAt($parts.Count - 1)
            continue
        }
        $parts.Add($part)
    }
    return (($parts -join "/") -imatch '(^|/)src/(?!gpu(/|$))[^/]') -or ($File -imatch $Names)
}

# The variables matching PATTERN (a case-sensitive regular expression, as CMake's variable names are) that a command
# writes: each argument that is such a name written out, but in a condition or a message, and where a command only
# reads a variable by its name: the list or path of list(GET|LENGTH|FIND|JOIN|SUBLIST ...) and
# cmake_path(GET|HAS_*|IS_*|HASH|NATIVE_PATH ...), and foreach's lists (its loop variables, before IN, it writes). A
# read, ${NAME}, is no such argument.
function Get-WrittenVariables($Command, [string]$Pattern) {
    $arguments = $Command.arguments
    if (@("if", "elseif", "else", "endif", "while", "endwhile", "message") -contains $Command.name) { return , @() }
    $last = $arguments.Count - 1
    $read = -1
    if ($Command.name -eq "foreach") {
        $last = 0
        $in = [Array]::IndexOf($arguments, "IN")
        if ($in -gt 0) { $last = $in - 1 }
    }
    elseif ($arguments.Count -gt 1 -and
            (($Command.name -eq "list" -and $arguments[0] -cmatch '^(GET|LENGTH|FIND|JOIN|SUBLIST)$') -or
             ($Command.name -eq "cmake_path" -and $arguments[0] -cmatch '^(GET|HAS_\w+|IS_\w+|HASH|NATIVE_PATH)$'))) {
        $read = 1
    }
    $names = New-Object System.Collections.Generic.List[string]
    for ($k = 0; $k -le $last; $k++) {
        if ($k -eq $read) { continue }
        foreach ($name in $arguments[$k].Split([char]';')) { if ($name -cmatch $Pattern) { $names.Add($name) } }
    }
    return , $names.ToArray()
}

# What in a command breaks the CPU rules, as reasons (SOURCES: the CPU sources' file names, a regular expression)
function Get-CpuBreaks($Command, [string]$Sources) {
    $reasons = New-Object System.Collections.Generic.List[string]
    $arguments = $Command.arguments
    $scope = ""
    if ($Command.name -eq "set_property" -and $arguments.Count -gt 0) { $scope = $arguments[0].ToUpperInvariant() }
    foreach ($target in (Get-ChangedTargets $Command)) {
        if ($cpuTargets -contains $target) { $reasons.Add("changes the CPU target $target") }
    }
    foreach ($file in (Get-PropertySources $Command)) {
        if (Test-CpuSource $file $Sources) { $reasons.Add("sets properties of the CPU source $file") }
    }
    foreach ($name in (Get-WrittenVariables $Command $cpuVariables)) { $reasons.Add("writes $name") }
    if ($Command.name -match '^(add_definitions|add_compile_definitions|add_link_options)$') {
        $reasons.Add("gives every target of the directory definitions or link options")
    }
    if ($Command.name -match '^(include_directories|remove_definitions|set_directory_properties)$' -or
        $scope -eq "DIRECTORY") {
        $reasons.Add("sets the directory's include paths, definitions or properties, which its targets read")
    }
    if ($scope -eq "GLOBAL" -and @((Split-Names $arguments) -match '^RULE_LAUNCH_').Count -gt 0) {
        $reasons.Add("puts a launcher in front of every compile or link")
    }
    return , $reasons.ToArray()
}

# What in a command breaks the renderer's rules, as reasons
function Get-GpuBreaks($Command) {
    $reasons = New-Object System.Collections.Generic.List[string]
    if ($Command.name -ne "target_sources") {
        foreach ($target in (Get-ChangedTargets $Command)) {
            if ($target -ieq $gpuTarget) { $reasons.Add("changes $gpuTarget") }
        }
    }
    foreach ($file in (Get-PropertySources $Command)) {
        if ($file -imatch $frozenUnits) { $reasons.Add("sets properties of the renderer's unit $file") }
    }
    foreach ($name in (Get-WrittenVariables $Command $cudaVariables)) { $reasons.Add("writes $name") }
    return , $reasons.ToArray()
}

# What in a CMake text breaks the rules RULES ("cpu", "gpu"; SOURCES as for Get-CpuBreaks): each command that breaks
# one, once, with its reasons
function Find-CMakeRuleBreaks([string]$Text, [string[]]$Rules, [string]$Sources) {
    $found = New-Object System.Collections.Generic.List[string]
    $seen = New-Object 'System.Collections.Generic.HashSet[string]'
    $commands = Get-CMakeCommands $Text
    if ($null -eq $commands) {
        $found.Add("(its commands can't be read: a parenthesis, quote or bracket is left open)")
        return , $found.ToArray()
    }
    foreach ($command in $commands) {
        $reasons = @()
        if ($Rules -contains "cpu") { $reasons += Get-CpuBreaks $command $Sources }
        if ($Rules -contains "gpu") { $reasons += Get-GpuBreaks $command }
        if ($reasons.Count -gt 0 -and $seen.Add($command.text)) {
            $found.Add(("{0}: {1}" -f $command.text, ($reasons -join "; ")))
        }
    }
    return , $found.ToArray()
}

function Add-Line($Result, [string]$Text) { $Result.lines.Add($Text) }
function Add-Failure($Result, [string]$Text) {
    $Result.lines.Add($Text)
    $Result.failures += 1
}

# Every check on the checkout DIR (its top folder) with the blob pins PINS: { failures, lines }
function Test-Frozen([string]$Dir, [object[]]$Pins) {
    $result = @{ failures = 0; lines = (New-Object System.Collections.Generic.List[string]) }

    # The (base, paths) pairs: tracked files changed since the base (committed or not), and files added under the
    # paths but not committed
    foreach ($set in $frozenSets) {
        $changed = @(Invoke-Git $Dir (@("diff", "--name-only", $set.base, "--") + $set.paths))
        $untracked = @(Invoke-Git $Dir (@("ls-files", "--others", "--exclude-standard", "--") + $set.paths))
        foreach ($file in $changed) { Add-Failure $result "CHANGED since $($set.name) ($($set.base)): $file" }
        foreach ($file in $untracked) { Add-Failure $result "UNTRACKED under a path frozen since $($set.name): $file" }
        Add-Line $result ("frozen since {0} ({1}): {2} paths checked, {3} changed" -f $set.name, $set.base,
                          $set.paths.Count, ($changed.Count + $untracked.Count))
    }

    # The blob pins
    $pinsChanged = 0
    foreach ($pin in $Pins) {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $pin.path) -PathType Leaf)) {
            Add-Failure $result "CHANGED: $($pin.path), pinned at blob $($pin.blob), is missing"
            $pinsChanged++
            continue
        }
        if (@(Invoke-Git $Dir @("ls-files", "--", $pin.path)).Count -eq 0) {
            Add-Failure $result "CHANGED: $($pin.path), pinned at blob $($pin.blob), is untracked"
            $pinsChanged++
            continue
        }
        $blob = @(Invoke-Git $Dir @("hash-object", "--", $pin.path))[0]
        if ($blob -ne $pin.blob) {
            Add-Failure $result "CHANGED: $($pin.path) is blob $blob, pinned at $($pin.blob)"
            $pinsChanged++
        }
    }
    Add-Line $result ("blob pins: {0} files checked, {1} changed" -f @($Pins).Count, $pinsChanged)

    # renderer.cu's verbatim spans (String.Contains is ordinal)
    if (Test-Path -LiteralPath (Join-Path $Dir "src/gpu/renderer.cu") -PathType Leaf) {
        $renderer = Get-WorkingText $Dir "src/gpu/renderer.cu"
        $baseRenderer = Get-BaseLines $Dir $cmakeStep70 "src/gpu/renderer.cu"
        foreach ($span in $rendererSpans) {
            $text = ($baseRenderer[($span.first - 1)..($span.last - 1)]) -join "`n"
            $where = "renderer.cu, {0} (Step 70's lines {1}-{2})" -f $span.name, $span.first, $span.last
            if ($renderer.Contains($text)) { Add-Line $result "${where}: verbatim" }
            else { Add-Failure $result "CHANGED: $where" }
        }
    }
    else {
        Add-Failure $result "CHANGED: src/gpu/renderer.cu is missing"
    }

    # CMakeLists.txt
    if (-not (Test-Path -LiteralPath (Join-Path $Dir "CMakeLists.txt") -PathType Leaf)) {
        Add-Failure $result "CHANGED: CMakeLists.txt is missing"
        return $result
    }
    $cmake = Get-WorkingText $Dir "CMakeLists.txt"
    # The CPU sources' file names for the CMake rules: std.ixx, utf8.manifest and every file under src/ outside src/gpu/
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($name in $cpuSourceNames) { $names.Add($name) }
    $src = Join-Path $Dir "src"
    if (Test-Path -LiteralPath $src -PathType Container) {
        $gpu = [IO.Path]::GetFullPath((Join-Path $src "gpu")) + "\"
        foreach ($file in @(Get-ChildItem -LiteralPath $src -Recurse -File)) {
            if (-not $file.FullName.StartsWith($gpu, [StringComparison]::OrdinalIgnoreCase)) { $names.Add($file.Name) }
        }
    }
    $sources = '(?<![\w.-])(' + (($names | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')(?![\w.])'
    # Step 70's text up to the ImageCompare target's flags is the start of today's; after it, the CPU rules
    $base70 = (Get-BaseLines $Dir $cmakeStep70 "CMakeLists.txt") -join "`n"
    $prefix70 = $base70.Substring(0, $base70.IndexOf($cmakeStep70End, [StringComparison]::Ordinal) +
                                     $cmakeStep70End.Length)
    if ($cmake.StartsWith($prefix70, [StringComparison]::Ordinal)) {
        Add-Line $result "CMakeLists.txt, Step 70's CPU section (through the ImageCompare target): verbatim"
        $breaks = Find-CMakeRuleBreaks ($cmake.Substring($prefix70.Length)) @("cpu") $sources
        foreach ($hit in $breaks) { Add-Failure $result "CHANGED: CMakeLists.txt after Step 70's CPU section: $hit" }
        if ($breaks.Count -eq 0) {
            Add-Line $result ("CMakeLists.txt after Step 70's CPU section: no command changes the CPU targets or " +
                              "their sources, the C++ flags or the directory's include paths and definitions")
        }
    }
    else {
        Add-Failure $result "CHANGED: CMakeLists.txt, Step 70's CPU section (through the ImageCompare target)"
    }
    # Step 85's lines, all of them, are today's first lines; after them, the renderer's rules
    $base85 = Get-BaseLines $Dir $cmakeStep85 "CMakeLists.txt"
    $lines = $cmake -split "`n"
    $same = $lines.Count -ge $base85.Count
    for ($k = 0; $same -and $k -lt $base85.Count; $k++) { $same = $lines[$k] -ceq $base85[$k] }
    if ($same) {
        Add-Line $result ("CMakeLists.txt, Step 85's {0} lines: verbatim" -f $base85.Count)
        $after85 = ""
        if ($lines.Count -gt $base85.Count) { $after85 = ($lines[$base85.Count..($lines.Count - 1)]) -join "`n" }
        $breaks = Find-CMakeRuleBreaks $after85 @("gpu") $sources
        foreach ($hit in $breaks) { Add-Failure $result "CHANGED: CMakeLists.txt after Step 85's lines: $hit" }
        if ($breaks.Count -eq 0) {
            Add-Line $result ("CMakeLists.txt after Step 85's lines: no command touches the renderer's existing " +
                              "units, gpu_renderer or the CUDA flags")
        }
    }
    else {
        Add-Failure $result ("CHANGED: CMakeLists.txt, Step 85's {0} lines are not its first lines" -f $base85.Count)
    }

    # The CMake modules but sdl3.cmake (frozen since Step 70 above), the new ones included: every rule
    $modules = @()
    if (Test-Path -LiteralPath (Join-Path $Dir "cmake") -PathType Container) {
        $modules = @(Get-ChildItem -LiteralPath (Join-Path $Dir "cmake") -Filter "*.cmake" -Recurse -File |
            Where-Object { $_.FullName -ne [IO.Path]::GetFullPath((Join-Path $Dir "cmake\sdl3.cmake")) } |
            Sort-Object FullName)
    }
    $moduleBreaks = 0
    foreach ($module in $modules) {
        $name = "cmake/" + $module.FullName.Substring((Join-Path $Dir "cmake").Length + 1).Replace("\", "/")
        $text = ([IO.File]::ReadAllText($module.FullName)) -replace "`r`n", "`n"
        $breaks = Find-CMakeRuleBreaks $text @("cpu", "gpu") $sources
        foreach ($hit in $breaks) { Add-Failure $result "CHANGED: ${name}: $hit" }
        $moduleBreaks += $breaks.Count
    }
    Add-Line $result ("CMake modules but sdl3.cmake: {0} read, {1} commands break a rule" -f $modules.Count,
                      $moduleBreaks)
    return $result
}

# --- The self-test's hands --------------------------------------------------------------------------------------------

function Add-LineTo([string]$Work, [string]$Path, [string]$Line) {
    $full = Join-Path $Work $Path
    $bytes = [IO.File]::ReadAllBytes($full)
    $start = ""
    if ($bytes.Length -gt 0 -and $bytes[$bytes.Length - 1] -ne 10) { $start = "`n" }
    [IO.File]::AppendAllText($full, $start + $Line + "`n")
}
function Edit-OneByte([string]$Work, [string]$Path) {
    $full = Join-Path $Work $Path
    $bytes = [IO.File]::ReadAllBytes($full)
    if ($bytes.Length -eq 0) { throw "$Path is empty: no byte to change" }
    $bytes[0] = $bytes[0] -bxor 1
    [IO.File]::WriteAllBytes($full, $bytes)
}
function New-FileIn([string]$Work, [string]$Path, [string]$Text) {
    $full = Join-Path $Work $Path
    $folder = Split-Path -Parent $full
    if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
    [IO.File]::WriteAllText($full, $Text)
}
# A CMake case's text: appended to CMakeLists.txt, or as a new module
function Add-CMakeCase([string]$Work, [string]$File, [string]$Text) {
    if ($File -ceq "CMakeLists.txt") { Add-LineTo $Work $File $Text }
    else { New-FileIn $Work $File ($Text + "`n") }
}
# The worktree back at HEAD: tracked files restored, untracked ones deleted (only files git lists in it), then checked
function Reset-SelfTestTree([string]$Work) {
    Invoke-Git $Work @("checkout", "-q", "HEAD", "--", ".") | Out-Null
    foreach ($file in @(Invoke-Git $Work @("ls-files", "--others", "--exclude-standard"))) {
        Remove-Item -LiteralPath (Join-Path $Work $file) -Force
    }
    $status = @(Invoke-Git $Work @("status", "--porcelain", "--untracked-files=all"))
    if ($status.Count -gt 0) { throw ("the self-test's worktree is not HEAD's: {0}" -f ($status -join "; ")) }
}
# The self-test's worktree gone: as git's worktree, or, left by an interrupted run that git no longer knows, as this
# script's own folder
function Remove-SelfTestTree([string]$Work) {
    if (-not (Test-Path -LiteralPath $Work)) { return }
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & git -C $project worktree remove --force $Work }
    finally { $ErrorActionPreference = $saved }
    if (Test-Path -LiteralPath $Work) { Remove-Item -LiteralPath $Work -Recurse -Force }
}

# --- Run --------------------------------------------------------------------------------------------------------------

if ($SelfTest) {
    $work = Join-Path $project "out\m5\runs\frozen-selftest"
    Write-Output "Gate F self-test: each case alone in a detached worktree of HEAD, $work"
    $behaved = 0
    $rows = New-Object System.Collections.Generic.List[string]
    try {
        Remove-SelfTestTree $work
        $runs = Split-Path -Parent $work
        if (-not (Test-Path -LiteralPath $runs)) { New-Item -ItemType Directory -Path $runs | Out-Null }
        # --force: a worktree of this path that git still lists though its folder is gone is replaced
        Invoke-Git $project @("worktree", "add", "--quiet", "--force", "--detach", $work, "HEAD") | Out-Null
        # Nothing below may touch any other checkout: the worktree's top folder must be $work itself
        $top = Get-TopLevel $work
        if ($top -ne [IO.Path]::GetFullPath($work)) { throw "the self-test's worktree is $top, not $work" }
        Reset-SelfTestTree $work
        $pinBlob = @(Invoke-Git $work @("rev-parse", "HEAD:$selfTestPin"))[0]
        $pins = @($blobPins) + @(@{ path = $selfTestPin; blob = $pinBlob })
        $rows.Add(("  {0,-9}{1,-7}{2}" -f "expected", "got", "case"))
        foreach ($case in $selfTestCases) {
            $details = @()
            try {
                if ($case.text) { Add-CMakeCase $work $case.file $case.text } else { & $case.plant $work }
                $checked = Test-Frozen $work $pins
                $got = "PASS"
                if ($checked.failures -gt 0) { $got = "FAIL" }
                $details = @($checked.lines | Where-Object { $_ -clike "CHANGED*" -or $_ -clike "UNTRACKED*" })
            }
            catch {
                $got = "ERROR"
                $details = @($_.Exception.Message)
            }
            Reset-SelfTestTree $work
            $mark = ""
            if ($got -ceq $case.expect) { $behaved++ } else { $mark = "   <- not as expected" }
            $rows.Add(("  {0,-9}{1,-7}{2}{3}" -f $case.expect, $got, $case.name, $mark))
            foreach ($detail in $details) { $rows.Add("                   $detail") }
        }
    }
    finally {
        Remove-SelfTestTree $work
        # The cases run so far, also when an error stopped the run
        $rows | ForEach-Object { Write-Output $_ }
    }
    if ($behaved -eq $selfTestCases.Count) {
        Write-Output ("Gate F self-test: PASS ({0} of {0} cases as expected)" -f $selfTestCases.Count)
        exit 0
    }
    Write-Output ("Gate F self-test: FAIL ({0} of {1} cases as expected)" -f $behaved, $selfTestCases.Count)
    exit 1
}

if (-not $Root) { $Root = $project }
$Root = (Resolve-Path -LiteralPath $Root).ProviderPath
$top = Get-TopLevel $Root
if ($top -ne [IO.Path]::GetFullPath($Root).TrimEnd("\")) {
    throw "-Root $Root is not the top folder of a checkout ($top)"
}
if ($Root -ne $project) { Write-Output "Gate F on $Root" }
$checked = Test-Frozen $Root $blobPins
$checked.lines | ForEach-Object { Write-Output $_ }
if ($checked.failures -gt 0) {
    Write-Output "Gate F: FAIL ($($checked.failures))"
    exit 1
}
Write-Output "Gate F: PASS"
exit 0
