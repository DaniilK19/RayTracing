# The documentation gate: README.md and docs\*.md hold no dead link, name no hardware, add no speed figure, and their
# commands run.
#   1. links: every relative link and image resolves to a file git keeps or will keep (tracked, or new and not
#      ignored), with the same case, as GitHub serves it; every #anchor to a heading of the page it names (GitHub's
#      anchors: the heading in lower case, without punctuation, spaces as hyphens). Link forms this script doesn't
#      read (reference-style definitions, HTML links and images, targets in <>) fail, rather than pass unread;
#   2. hardware: no page names the machine the project was measured on (a GPU or CPU model, a memory size, a count of
#      cores or multiprocessors, a compute capability), apart from the general statements listed in $allowed below,
#      which explain how a GPU works rather than which one this is (the rest of their line is searched still);
#   3. speed: every speed or time figure (per second, faster, slower, a number of seconds or milliseconds) stands in
#      the page as it was at -Baseline (the tag gpu-3: the last milestone that measured speed), with its words around
#      it, so no new one comes in unnoticed;
#   4. commands: the first RayTracingGPU command of each page that has one (with a scene first) runs as written, with
#      -w 64 -s 16 appended (the last of a repeated option counts), its PNG written into out\m5\runs\stepNN\docs\
#      instead of the page's -o. Renders go through gpu.ps1 (the queue and the GPU log); -NoRender skips them.
#   powershell -ExecutionPolicy Bypass -File scripts\verify\docs.ps1 -Step 85 [-NoRender] [-Baseline gpu-3]
# Exit code 0 when every check passes.
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [switch]$NoRender,
    [string]$Baseline = "gpu-3",
    [string]$Build = "out\build\x64-release"
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [Text.Encoding]::UTF8 # git's output, the pages' text
$project = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $project
$pages = @(Get-Item -LiteralPath (Join-Path $project "README.md")) +
    @(Get-ChildItem -LiteralPath (Join-Path $project "docs") -Filter *.md | Sort-Object Name)
$failures = 0
function Get-RepoPath([string]$full) { return $full.Substring($project.Length + 1).Replace("\", "/") }

# The files git keeps or will keep, by their exact names
$kept = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
foreach ($file in @(git ls-files --cached --others --exclude-standard)) { [void]$kept.Add($file) }
foreach ($file in @(git ls-files --cached --others --exclude-standard)) {
    # and their folders, which a link may name
    for ($dir = Split-Path -Parent $file; $dir; $dir = Split-Path -Parent $dir) { [void]$kept.Add($dir.Replace("\", "/")) }
}

# GitHub's anchor of a heading: lower case, punctuation dropped (letters, digits, spaces, hyphens and underscores
# stay), each space a hyphen; a repeated heading gets -1, -2, ...
function Get-Anchors([string]$path) {
    $anchors = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::Ordinal)
    $inCode = $false
    foreach ($line in [IO.File]::ReadAllLines($path)) {
        if ($line -match '^```') { $inCode = -not $inCode; continue }
        if ($inCode -or $line -notmatch '^#{1,6}\s+(.+?)\s*#*\s*$') { continue }
        $text = $Matches[1] -replace '`', '' -replace '\[([^\]]*)\]\([^)]*\)', '$1'
        $slug = ($text.ToLowerInvariant() -replace '[^\p{L}\p{Nd} _-]', '') -replace ' ', '-'
        $name = $slug
        for ($k = 1; $anchors.Contains($name); $k++) { $name = "$slug-$k" }
        [void]$anchors.Add($name)
    }
    return , $anchors
}

# --- 1. links ----------------------------------------------------------------------------------------------------------
$anchorCache = @{}
$linkCount = 0
foreach ($page in $pages) {
    $text = [IO.File]::ReadAllText($page.FullName) -replace '(?s)```.*?```', ''
    foreach ($m in [regex]::Matches($text, '(?m)^\s*\[[^\]]+\]:\s*\S+|<(img|a)\s[^>]*(src|href)=|\]\(<')) {
        Write-Output "docs: $($page.Name): a link this script doesn't read: '$($m.Value.Trim())'"
        $failures++
    }
    # Every ](target), whatever the brackets hold: an image inside a link gives both targets
    foreach ($m in [regex]::Matches($text, '\]\(([^)\s]+)(?:\s+(?:"[^"]*"|''[^'']*''))?\)')) {
        $target = $m.Groups[1].Value
        if ($target -match '^[a-z]+:') { continue } # http, https, mailto
        $linkCount++
        $file, $anchor = $target.Split("#", 2)
        $resolved = if ($file) { [IO.Path]::GetFullPath((Join-Path $page.DirectoryName ([Uri]::UnescapeDataString($file)))) }
                    else { $page.FullName }
        if (-not $resolved.StartsWith($project + "\") -or -not $kept.Contains((Get-RepoPath $resolved))) {
            Write-Output "docs: $($page.Name): '$target' isn't a file git keeps (or its case differs)"
            $failures++
            continue
        }
        if ($anchor -and $resolved.EndsWith(".md")) {
            if (-not $anchorCache.ContainsKey($resolved)) { $anchorCache[$resolved] = Get-Anchors $resolved }
            if (-not $anchorCache[$resolved].Contains($anchor)) {
                Write-Output "docs: $($page.Name): '$target': no heading has the anchor #$anchor"
                $failures++
            }
        }
    }
}
Write-Output "links: $linkCount relative links and images checked"

# --- 2. hardware -------------------------------------------------------------------------------------------------------
$hardware = '\b(RTX|GTX|GeForce|Quadro|Radeon|Ryzen|Threadripper|Core i[3579]|Xeon|Intel|AMD|Ada Lovelace|Ampere|' +
    'Blackwell|Turing)\b|\b\d+(\.\d+)? ?(GB|GiB|MB of VRAM)\b|\bVRAM\b|\b\d+ (cores|threads|SMs|multiprocessors)\b|' +
    '\bblocks of \d+ threads per multiprocessor\b|\bcompute capability\b|\bsm_\d{2,3}\b|\bcompute_\d{2,3}\b'
# General statements already in the repo (page, the phrase): how a GPU works, not which one this is
$allowed = @(
    @("gpu.md", "A GeForce card does arithmetic on 64-bit doubles"),
    @("gpu.md", "5 blocks of 128 threads per multiprocessor"))
foreach ($page in $pages) {
    $lines = [IO.File]::ReadAllLines($page.FullName)
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $rest = $lines[$i]
        foreach ($a in $allowed) { if ($a[0] -eq $page.Name) { $rest = $rest.Replace($a[1], "") } }
        if ($rest -match $hardware) {
            Write-Output ("docs: {0}:{1}: names hardware: '{2}'" -f $page.Name, ($i + 1), $Matches[0])
            $failures++
        }
    }
}
Write-Output "hardware: $($pages.Count) pages searched"

# --- 3. speed ----------------------------------------------------------------------------------------------------------
$speed = '(?i)per second|faster|slower|\b\d+(\.\d+)? ?(ms|s|seconds?|milliseconds?|minutes?|min)\b'
$speedCount = 0
foreach ($page in $pages) {
    # (through cmd, so that git's "not in that tag" for a new page isn't an error; UTF-8, as the pages are)
    $before = (cmd /c "git show ${Baseline}:$(Get-RepoPath $page.FullName) 2>nul") -join "`n"
    foreach ($line in [IO.File]::ReadAllLines($page.FullName)) {
        foreach ($m in [regex]::Matches($line, $speed)) {
            $speedCount++
            # The figure with up to 30 characters before it, or up to 30 after it (a sentence added beside an old
            # figure leaves one side as it was), cut at word boundaries, must be in the baseline page
            $from = [Math]::Max(0, $m.Index - 30)
            $left = $line.Substring($from, $m.Index + $m.Length - $from)
            if ($from -gt 0) { $left = $left -replace '^\S*\s', '' }
            $to = [Math]::Min($line.Length, $m.Index + $m.Length + 30)
            $right = $line.Substring($m.Index, $to - $m.Index)
            if ($to -lt $line.Length) { $right = $right -replace '\s\S*$', '' }
            $snippet = $line.Substring($from, $to - $from)
            if (-not ($before.Contains($left) -or $before.Contains($right))) {
                Write-Output "docs: $($page.Name): a speed or time figure not in the page at ${Baseline}: '$snippet'"
                $failures++
            }
        }
    }
}
Write-Output "speed: $speedCount figures, each in its page at $Baseline"

# --- 4. commands -------------------------------------------------------------------------------------------------------
if (-not $NoRender) {
    $gpuScript = Join-Path $PSScriptRoot "gpu.ps1"
    $gpu = Join-Path $project "$Build\RayTracingGPU.exe"
    $work = Join-Path $project ("out\m5\runs\step{0}\docs" -f $Step)
    New-Item -ItemType Directory -Force $work | Out-Null
    foreach ($page in $pages) {
        # The first command: in a code block or inline code, from the program's name (with or without its folder) on,
        # with a scene first (an option first is a template, such as "--strata K -s N")
        $command = $null
        foreach ($line in [IO.File]::ReadAllLines($page.FullName)) {
            $m = [regex]::Match($line, '(?:^|`)(?:out\\build\\x64-release\\)?RayTracingGPU(?:\.exe)? ([a-z][^`]*?)(?:`|$)')
            if ($m.Success) { $command = $m.Groups[1].Value.Trim(); break }
        }
        if (-not $command) { continue }
        $words = @([regex]::Matches($command, '"[^"]*"|\S+') | ForEach-Object { $_.Value.Trim('"') })
        # An output the page names goes to the work folder instead
        $arguments = @()
        for ($k = 0; $k -lt $words.Count; $k++) {
            if ($words[$k] -eq "-o" -or $words[$k] -eq "--output") { $k++; continue }
            $arguments += $words[$k]
        }
        $png = Join-Path $work ($page.BaseName + ".png")
        $all = @("-Step", "$Step", "-Label", "docs $($page.Name)", "--", $gpu) + $arguments +
            @("-w", "64", "-s", "16", "-o", $png)
        $out = @(& $gpuScript @all)
        if ($LASTEXITCODE -ne 0) {
            $out | ForEach-Object { Write-Output "  $_" }
            Write-Output "docs: $($page.Name): 'RayTracingGPU $command' failed (exit code $LASTEXITCODE)"
            $failures++
        }
        else {
            Write-Output "command of $($page.Name): RayTracingGPU $command (-w 64 -s 16): ran"
        }
    }
}

if ($failures -gt 0) {
    Write-Output "Docs gate: FAIL ($failures)"
    exit 1
}
Write-Output "Docs gate: PASS"
exit 0
