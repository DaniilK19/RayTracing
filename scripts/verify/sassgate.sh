#!/usr/bin/env bash
# The SASS gate: the books' four render kernels and the display kernel compiled into a new build are the same machine
# code as in Step 70, instruction for instruction. cuobjdump (CUDA's disassembler) prints the GPU code embedded in an
# executable; nothing runs on the GPU. It proves:
#   - first_book_kernel<float>, first_book_kernel<double>, all_books_kernel<float>, all_books_kernel<double> and
#     gpu::detail::rgba8_kernel are identical to the baseline dump's, for the same GPU architecture;
#   - their constant bank 2 (the constants the compiler put beside the code: doubles, jump tables), from
#     `cuobjdump -elf` of the new exe and of the baseline's exe, is identical too;
#   - with --prev DUMP, every other kernel found in both the new build and DUMP (the quality and tone-map kernels of a
#     step that says they are unchanged) is identical too.
# Every other kernel is listed as new, with its registers, stack, shared and local memory per thread.
#
# The kernels in renderer.cu live in an unnamed namespace, whose mangled name carries two hashes of the source file's
# path (_GLOBAL__N__<8 hex>_11_renderer_cu_<8 hex>), different in another checkout such as a worktree: they are
# replaced by x's, eight for eight, so the names' length prefixes stay valid and cu++filt still demangles them. Only the
# instructions are compared, without their addresses and encodings; branch targets stay absolute addresses, so a change
# in a kernel's size anywhere before a branch shows as a difference too.
#
#   bash scripts/verify/sassgate.sh NEW.exe BASE.sass.txt [--base-exe BASE.exe] [--prev PREV.sass.txt]
#                                   [--save OUT.sass.txt]
# BASE is `cuobjdump -sass` of the Step 70 build (out/m4/bin70/RayTracingGPU.sass.txt or RayTracingViewer.sass.txt);
# BASE.exe, the exe it was dumped from, is by default the same name with .exe (out/m4/bin70/RayTracingGPU.exe).
# --save keeps the new dump, unnormalized, for a later --prev. Exit code 0 when all compared kernels are identical.
set -u

if [ $# -lt 2 ]; then
    echo "usage: sassgate.sh NEW.exe BASE.sass.txt [--base-exe BASE.exe] [--prev PREV.sass.txt]" \
         "[--save OUT.sass.txt]" >&2
    exit 2
fi
NEW="$1"
BASE="$2"
shift 2
PREV=""
SAVE=""
BASE_EXE="${BASE%.sass.txt}.exe"
while [ $# -gt 0 ]; do
    case "$1" in
    --prev) PREV="$2"; shift 2 ;;
    --base-exe) BASE_EXE="$2"; shift 2 ;;
    --save) SAVE="$2"; shift 2 ;;
    *) echo "sassgate.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# cuobjdump from the CUDA Toolkit: CUDA_PATH's, or the newest installed version's
CU="${CUDA_PATH:-}"
if [ -z "$CU" ]; then
    CU="$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v"* 2>/dev/null | sort -V | tail -1)"
fi
CU="$(cygpath -u "$CU" 2>/dev/null || echo "$CU")/bin"
if [ ! -x "$CU/cuobjdump.exe" ]; then
    echo "sassgate.sh: cuobjdump.exe not found in $CU (set CUDA_PATH)" >&2
    exit 2
fi
for f in "$NEW" "$BASE" "$BASE_EXE" ${PREV:+"$PREV"}; do
    [ -f "$f" ] || { echo "sassgate.sh: no file $f" >&2; exit 2; }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

"$CU/cuobjdump.exe" -sass "$NEW" > "$WORK/new.raw" 2>&1 || { echo "cuobjdump -sass failed on $NEW" >&2; exit 2; }
"$CU/cuobjdump.exe" -res-usage "$NEW" > "$WORK/new.res" 2>&1 || { echo "cuobjdump -res-usage failed" >&2; exit 2; }
"$CU/cuobjdump.exe" -elf "$NEW" > "$WORK/new.elf.raw" 2>&1 || { echo "cuobjdump -elf failed on $NEW" >&2; exit 2; }
"$CU/cuobjdump.exe" -elf "$BASE_EXE" > "$WORK/base.elf.raw" 2>&1 ||
    { echo "cuobjdump -elf failed on $BASE_EXE" >&2; exit 2; }
if [ -n "$SAVE" ]; then
    mkdir -p "$(dirname "$SAVE")"
    cp "$WORK/new.raw" "$SAVE"
fi

normalize() {
    sed -E 's/_GLOBAL__N__[0-9a-f]{8}_([0-9]+_[A-Za-z0-9_]+_cu_)[0-9a-f]{8}/_GLOBAL__N__xxxxxxxx_\1xxxxxxxx/g' "$1" > "$2"
}
normalize "$WORK/new.raw" "$WORK/new.sass"
normalize "$BASE" "$WORK/base.sass"
normalize "$WORK/new.res" "$WORK/new.res.n"
normalize "$WORK/new.elf.raw" "$WORK/new.elf"
normalize "$WORK/base.elf.raw" "$WORK/base.elf"
[ -n "$PREV" ] && normalize "$PREV" "$WORK/prev.sass"

# One file per function: its instructions only, without the address comment before them and the encoding after
split_kernels() {
    local dump="$1" dir="$2"
    mkdir -p "$dir"
    awk -v dir="$dir" '
        /Function :/ { name = $3; file = dir "/" name ".sass"; printf "" > file; next }
        file && /^[ \t]+\/\*[0-9a-f]+\*\// {
            line = $0
            sub(/^[ \t]+\/\*[0-9a-f]+\*\/[ \t]*/, "", line)
            sub(/[ \t]*\/\*.*$/, "", line)
            sub(/[ \t]*;[ \t]*$/, "", line)
            print line > file
        }' "$dump"
}
split_kernels "$WORK/new.sass" "$WORK/new"
split_kernels "$WORK/base.sass" "$WORK/base"
[ -n "$PREV" ] && split_kernels "$WORK/prev.sass" "$WORK/prev"

failures=0

# The same GPU architectures in both
if ! diff <(grep -E '^arch = ' "$WORK/new.sass" | sort -u) <(grep -E '^arch = ' "$WORK/base.sass" | sort -u) >/dev/null; then
    echo "DIFFERENT architectures: new $(grep -E '^arch = ' "$WORK/new.sass" | sort -u | tr '\n' ' ')," \
         "base $(grep -E '^arch = ' "$WORK/base.sass" | sort -u | tr '\n' ' ')"
    failures=$((failures + 1))
fi

demangle() { "$CU/cu++filt.exe" "$1" 2>/dev/null || echo "$1"; }

# A function's constant bank 2 as -elf prints it: the hex words under the line ".nv.constant2.<name>", up to the blank
# line that ends the section (nothing for a function without one)
constants() {
    awk -v s=".nv.constant2.$2" '$0 == s { on = 1; next } on && NF == 0 { exit } on { print }' "$1"
}

# The frozen kernels: each pattern must name exactly one function of the baseline
frozen_names=()
for pattern in 'first_book_kernelIfE' 'first_book_kernelIdE' 'all_books_kernelIfE' 'all_books_kernelIdE' \
               '^_ZN3gpu6detail12rgba8_kernelEPKdiiidPh$'; do
    matches=$(ls "$WORK/base" | sed 's/\.sass$//' | grep -E -- "$pattern" || true)
    count=$(printf '%s' "$matches" | grep -c . || true)
    if [ "$count" -ne 1 ]; then
        echo "BASELINE: $count functions match $pattern (expected 1)"
        failures=$((failures + 1))
        continue
    fi
    name="$matches"
    frozen_names+=("$name")
    label="$(demangle "$name" | sed -E 's/\(.*//')"
    if [ ! -f "$WORK/new/$name.sass" ]; then
        echo "MISSING: $label"
        failures=$((failures + 1))
    elif ! cmp -s "$WORK/new/$name.sass" "$WORK/base/$name.sass"; then
        lines=$(diff "$WORK/base/$name.sass" "$WORK/new/$name.sass" | grep -c '^[<>]' || true)
        echo "DIFFERENT: $label ($lines lines differ)"
        failures=$((failures + 1))
    elif [ "$(constants "$WORK/new.elf" "$name")" != "$(constants "$WORK/base.elf" "$name")" ]; then
        echo "DIFFERENT constants: $label (its instructions are identical, its constant bank 2 isn't)"
        failures=$((failures + 1))
    else
        words=$(constants "$WORK/base.elf" "$name" | wc -w)
        echo "identical: $label ($(wc -l < "$WORK/new/$name.sass") instructions, $words words of constants)"
    fi
done

# The other kernels: new ones listed with their resources; with --prev, compared with the previous step's
resources() {
    awk -v f="Function $1:" '$0 ~ /^ Function / { found = ($1 " " $2 == f) ; next } found { print; exit }' \
        "$WORK/new.res.n" | sed -E 's/^ +//' | grep -oE 'REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+'
}
for file in "$WORK/new"/*.sass; do
    name="$(basename "$file" .sass)"
    printf '%s\n' "${frozen_names[@]}" | grep -qxF -- "$name" && continue
    label="$(demangle "$name" | sed -E 's/\(.*//')"
    res="$(resources "$name")"
    if [ -n "$PREV" ] && [ -f "$WORK/prev/$name.sass" ]; then
        if cmp -s "$file" "$WORK/prev/$name.sass"; then
            echo "identical to --prev: $label ($(wc -l < "$file") instructions; $res)"
        else
            lines=$(diff "$WORK/prev/$name.sass" "$file" | grep -c '^[<>]' || true)
            echo "DIFFERENT from --prev: $label ($lines lines differ; $res)"
            failures=$((failures + 1))
        fi
    else
        echo "new: $label ($(wc -l < "$file") instructions; $res)"
    fi
done
for name in "${frozen_names[@]}"; do
    echo "resources of $(demangle "$name" | sed -E 's/\(.*//'): $(resources "$name")"
done

if [ "$failures" -eq 0 ]; then
    echo "SASS gate: PASS ($(basename "$NEW"))"
    exit 0
fi
echo "SASS gate: FAIL ($failures, $(basename "$NEW"))"
exit 1
