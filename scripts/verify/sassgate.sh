#!/usr/bin/env bash
# The SASS gate: kernels compiled into a new build are the same machine code as in a baseline, instruction for
# instruction. cuobjdump (CUDA's disassembler) prints the GPU code embedded in an executable; nothing runs on the GPU.
#
#   bash scripts/verify/sassgate.sh --all BASE_DIR NEW_DIR [--prev PREV_DIR] [--save OUT_DIR] [--exes LIST] [--list]
# Every function of a baseline's three GPU programs, RayTracingGPU, RayTracingViewer and GpuTests (or the
# comma-separated LIST), in a new build. For each program it proves:
#   - every function of BASE's dump (in out/m5/bin85: the books' four kernels, rgba8_kernel, quality_kernel,
#     tone_map_kernel and every GpuTests test kernel) is in NEW with identical instructions and an identical constant
#     bank 2 (the constants the compiler put beside the code: doubles, jump tables; the words under
#     `.nv.constant2.<name>` in `cuobjdump -elf`), compiled for the same GPU architectures (the `arch =` lines); a
#     function MISSING from NEW, or DIFFERENT, fails;
#   - with --prev PREV_DIR, every other function of NEW that PREV has (milestone 5's own kernels, chained from the step
#     that adds them) is identical to PREV's in the same way; one DIFFERENT from --prev fails. PREV's functions that NEW
#     no longer has are listed, not failed: a step may retire a kernel of its own.
# NEW's remaining functions are listed as new, with their registers, stack, shared and local memory per thread
# (`cuobjdump -res-usage`). quality_kernel's instruction count is printed as a cross-check (6864 at Step 85), and each
# of the five Step 70 patterns of the form below must name exactly one function of BASE's RayTracingGPU: BASE is a real
# build, not an empty or foreign dump.
# BASE_DIR holds each program's <exe>.sass.txt (`cuobjdump -sass`) and <exe>.elf.txt (`cuobjdump -elf`); out/m5/bin85
# holds Step 85's, beside the exes and their <exe>.res.txt (`cuobjdump -res-usage`). NEW_DIR is a build folder, whose
# <exe>.exe is dumped here, or, when it holds <exe>.sass.txt, a folder of the three dumps, read as they are: so
# `--all out/m5/bin85 out/m5/bin85` and planted faults in copies of the dumps need no cuobjdump (dumps older than an
# <exe>.exe beside them belong to an older build and are refused). --save OUT_DIR writes
# NEW's three dumps per program, as cuobjdump printed them, for a later --prev (never into BASE_DIR or PREV_DIR, whose
# dumps it would replace). --list also prints every identical function. A name that occurs twice (one kernel compiled
# into two translation units) is matched copy by copy: each of BASE's copies needs an identical copy of its own in NEW.
# Every dump is read once (GpuTests' -sass dump is ~36 MB): split into a file per function, hashed with SHA-256, and
# matched by name in one awk pass, so the run grows linearly with the dumps.
# Exit code 0 when everything compared is identical, 1 when something isn't, 2 when a program, a dump or cuobjdump is
# missing.
#
#   bash scripts/verify/sassgate.sh NEW.exe BASE.sass.txt [--base-exe BASE.exe] [--prev PREV.sass.txt]
#                                   [--save OUT.sass.txt]
# Milestone 4's form: the books' four render kernels and the display kernel compiled into one new exe are the same
# machine code as in Step 70. It proves:
#   - first_book_kernel<float>, first_book_kernel<double>, all_books_kernel<float>, all_books_kernel<double> and
#     gpu::detail::rgba8_kernel are identical to the baseline dump's, for the same GPU architecture;
#   - their constant bank 2, from `cuobjdump -elf` of the new exe and of the baseline's exe, is identical too;
#   - with --prev DUMP, every other kernel found in both the new build and DUMP (the quality and tone-map kernels of a
#     step that says they are unchanged) is identical too.
# Every other kernel is listed as new, with its registers, stack, shared and local memory per thread.
# BASE is `cuobjdump -sass` of the Step 70 build (out/m4/bin70/RayTracingGPU.sass.txt or RayTracingViewer.sass.txt);
# BASE.exe, the exe it was dumped from, is by default the same name with .exe (out/m4/bin70/RayTracingGPU.exe).
# --save keeps the new dump, unnormalized, for a later --prev. Exit code 0 when all compared kernels are identical.
#
# In both forms: the kernels in renderer.cu (and those of the test suites) live in an unnamed namespace, whose mangled
# name carries two hashes of the source file's path (_GLOBAL__N__<8 hex>_11_renderer_cu_<8 hex>), different in another
# checkout such as a worktree: they are replaced by x's, eight for eight, so the names' length prefixes stay valid and
# cu++filt still demangles them. Only the instructions are compared, without their addresses and encodings; branch
# targets stay absolute addresses, so a change in a kernel's size anywhere before a branch shows as a difference too.
# --all also drops carriage returns, in case a dump was written with Windows line ends, and normalizes one more part
# of those names: in the test suites' files nvcc appends a number to the second hash (_GLOBAL__N__<8 hex>_13_lamp_tests_
# cu_<8 hex>_12828), the same for every name of one translation unit but changed by any edit of a header it includes
# (Step 86's gpu_test.h turned lamp_tests.cu's 12828 into 5988, and every other suite's likewise). Like the hashes it
# names the unit, not the code: it is dropped, and the name's length prefix written again for what is left, so the
# names still demangle.
set -u

# --- --all -----------------------------------------------------------------------------------------------------------

# A filter: every <length>_GLOBAL__N__<8 hex>_<n>_<file>_cu_<8 hex>[_<number>] of a mangled name, the length prefix
# taken as the 2 or 3 digits before it that give exactly such a name, becomes _GLOBAL__N__xxxxxxxx_<n>_<file>_cu_xxxxxxxx
# with its own length in front
all_normalize() {
    awk '
        function norm(s,    out, pre, run, rest, d, len, cand, done) {
            out = ""
            while (match(s, /[0-9]+_GLOBAL__N__/)) {
                pre = substr(s, 1, RSTART - 1)
                run = substr(s, RSTART, RLENGTH - 12)
                rest = substr(s, RSTART + RLENGTH - 12)
                done = 0
                for (d = 2; d <= 3 && d <= length(run); d++) {
                    len = substr(run, length(run) - d + 1) + 0
                    cand = substr(rest, 1, len)
                    if (length(cand) == len &&
                        cand ~ /^_GLOBAL__N__[0-9a-f]{8}_[0-9]+_[A-Za-z0-9_]+_cu_[0-9a-f]{8}(_[0-9]+)?$/) {
                        sub(/_cu_[0-9a-f]{8}(_[0-9]+)?$/, "_cu_xxxxxxxx", cand)
                        sub(/^_GLOBAL__N__[0-9a-f]{8}/, "_GLOBAL__N__xxxxxxxx", cand)
                        out = out pre substr(run, 1, length(run) - d) length(cand) cand
                        s = substr(rest, len + 1)
                        done = 1
                        break
                    }
                }
                if (!done) { out = out pre run "_GLOBAL__N__"; s = substr(rest, 13) }
            }
            return out s
        }
        { print norm($0) }'
}

all_usage() {
    echo "usage: sassgate.sh --all BASE_DIR NEW_DIR [--prev PREV_DIR] [--save OUT_DIR] [--exes LIST] [--list]" >&2
    exit 2
}

# A file the run can't do without
all_need() {
    [ -f "$1" ] || { echo "sassgate.sh: no file $1" >&2; exit 2; }
}

# The CUDA Toolkit's bin folder, CUDA_PATH's or the newest installed version's, into CUDA_BIN (empty when none)
all_find_cuda() {
    local cu="${CUDA_PATH:-}"
    if [ -z "$cu" ]; then
        cu="$(ls -d "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v"* 2>/dev/null | sort -V | tail -1)"
    fi
    CUDA_BIN=""
    if [ -n "$cu" ]; then
        CUDA_BIN="$(cygpath -u "$cu" 2>/dev/null || echo "$cu")/bin"
    fi
}

# cuobjdump OPTION EXE into OUT; when it fails, its own message and exit 2
all_dump() {
    "$CUDA_BIN/cuobjdump.exe" "$1" "$2" > "$3" 2> "$3.err" ||
        { cat "$3.err" >&2; echo "sassgate.sh: cuobjdump $1 failed on $2" >&2; exit 2; }
}

# One side's dumps, each read once, into DIR (names normalized, carriage returns dropped). A dump is a list of cubins,
# each under a line "Fatbin elf code:", in the same order in -sass, -elf and -res-usage; a name is unique within a
# cubin. So a function finds its constants and resources by its name, or, when the name occurs twice, by its name and
# its cubin's number, which then must agree across the dumps.
#   f<k>.sass   the k-th function of the -sass dump: its instructions, without their addresses and encodings;
#   sass.idx    "name TAB k TAB instructions TAB cubin", in the dump's order;  arch: the dump's `arch =` lines;
#   c<j>.const  the words of the j-th `.nv.constant2.<name>` section of the -elf dump (up to the blank line that ends
#               it);  const.idx: "name TAB j TAB words TAB cubin";
#   res.idx     "name TAB REG:… STACK:… SHARED:… LOCAL:… TAB cubin" from the -res-usage dump, when one is given;
#   cubins      the number of cubins in each dump: equal in dumps of one exe, and required to be when a name repeats;
#   hash        the SHA-256 of every f and c file.
all_index() {
    local sass="$1" elf="$2" res="$3" dir="$4"
    mkdir -p "$dir"
    tr -d '\r' < "$sass" | all_normalize | awk -v dir="$dir" '
        BEGIN { printf "" > (dir "/sass.idx"); printf "" > (dir "/arch") }
        /^Fatbin elf code:/ { cubin++ }
        /^arch = / { arch[$0] = 1 }
        /Function :/ {
            if (file) close(file)
            n++; name[n] = $3; count[n] = 0; in_cubin[n] = cubin
            file = dir "/f" n ".sass"; printf "" > file
            next
        }
        file && /^[ \t]+\/\*[0-9a-f]+\*\// {
            line = $0
            sub(/^[ \t]+\/\*[0-9a-f]+\*\/[ \t]*/, "", line)
            sub(/[ \t]*\/\*.*$/, "", line)
            sub(/[ \t]*;[ \t]*$/, "", line)
            print line > file
            count[n]++
        }
        END {
            if (file) close(file)
            for (i = 1; i <= n; i++) print name[i] "\t" i "\t" count[i] "\t" in_cubin[i] + 0 > (dir "/sass.idx")
            for (a in arch) print a > (dir "/arch")
            print "sass " cubin + 0 > (dir "/cubins")
        }'
    tr -d '\r' < "$elf" | all_normalize | awk -v dir="$dir" '
        BEGIN { printf "" > (dir "/const.idx") }
        on && NF == 0 { close(file); on = 0; next }
        on { print > file; words[j] += NF; next }
        /^Fatbin elf code:/ { cubin++ }
        /^\.nv\.constant2\./ && NF == 1 {
            j++; name[j] = substr($1, 15); words[j] = 0; in_cubin[j] = cubin
            file = dir "/c" j ".const"; printf "" > file; on = 1
            next
        }
        END {
            if (on) close(file)
            for (i = 1; i <= j; i++) print name[i] "\t" i "\t" words[i] "\t" in_cubin[i] + 0 > (dir "/const.idx")
            print "elf " cubin + 0 >> (dir "/cubins")
        }'
    if [ -n "$res" ]; then
        tr -d '\r' < "$res" | all_normalize | awk -v dir="$dir" '
            /^Fatbin elf code:/ { cubin++ }
            /^ Function / { name = $2; sub(/:$/, "", name); want = 1; next }
            want && match($0, /REG:[0-9]+ STACK:[0-9]+ SHARED:[0-9]+ LOCAL:[0-9]+/) {
                print name "\t" substr($0, RSTART, RLENGTH) "\t" cubin + 0
            }
            { want = 0 }
            END { print "res-usage " cubin + 0 >> (dir "/cubins") }' > "$dir/res.idx"
    fi
    if [ "$(cut -d' ' -f2 "$dir/cubins" | sort -u | wc -l)" -ne 1 ]; then
        if [ -n "$(cut -f1 "$dir/sass.idx" | sort | uniq -d | head -1)" ]; then
            echo "sassgate.sh: $sass, $elf${res:+ and $res} hold different numbers of cubins" \
                 "($(tr '\n' ' ' < "$dir/cubins")), and a name occurs twice: its copies can't be tied to their" \
                 "constants" >&2
            exit 2
        fi
        echo "note: $sass, $elf${res:+ and $res} hold different numbers of cubins ($(tr '\n' ' ' < "$dir/cubins"))," \
             "not dumps of one exe as they are; every name occurs once, so constants and resources go by name"
    fi
    local files=( "$dir"/f*.sass "$dir"/c*.const )
    if [ ${#files[@]} -gt 0 ]; then
        (cd "$dir" && sha256sum -- f*.sass c*.const) > "$dir/hash" ||
            { echo "sassgate.sh: sha256sum failed in $dir" >&2; exit 2; }
    else
        : > "$dir/hash"
    fi
}

# Readable names for the report, "name TAB label": cu++filt's, in one call for all names (the x's of the normalized
# names demangle too), without the return type and the parameter list, and with the translation unit of an unnamed
# namespace: "gpu::(anonymous namespace)::all_books_kernel<float> [renderer.cu]". The mangled names without cu++filt,
# or when its answers from standard input don't match its answer for the first name alone.
all_labels() {
    local names="$1" out="$2" filt=""
    [ -n "$CUDA_BIN" ] && filt="$CUDA_BIN/cu++filt.exe"
    if [ -n "$filt" ] && [ -x "$filt" ] && [ -s "$names" ] &&
       "$filt" < "$names" 2>/dev/null | tr -d '\r' > "$out.raw" &&
       [ "$(wc -l < "$out.raw")" -eq "$(wc -l < "$names")" ] &&
       [ "$("$filt" "$(head -1 "$names")" 2>/dev/null | tr -d '\r')" = "$(head -1 "$out.raw")" ]; then
        :
    else
        cp "$names" "$out.raw"
    fi
    paste "$names" "$out.raw" | awk -F '\t' '{
        s = ($2 == "") ? $1 : $2
        sub(/^void /, "", s)
        if (s ~ /\)$/) {
            d = 0
            for (i = length(s); i > 0; i--) {
                c = substr(s, i, 1)
                if (c == ")") d++
                else if (c == "(" && --d == 0) { s = substr(s, 1, i - 1); break }
            }
        }
        if (match($1, /_GLOBAL__N__xxxxxxxx_[0-9]+_[A-Za-z0-9_]+_cu_xxxxxxxx/)) {
            tu = substr($1, RSTART, RLENGTH)
            sub(/^_GLOBAL__N__xxxxxxxx_[0-9]+_/, "", tu)
            sub(/_cu_xxxxxxxx$/, ".cu", tu)
            s = s " [" tu "]"
        }
        print $1 "\t" s
    }' > "$out"
}

# One program's comparison, from the files all_index and all_labels wrote under DIR (DIR/base, DIR/new and, with
# --prev, DIR/prev). It prints records for the caller: "line TAB text"; "diff TAB text TAB file TAB file TAB tail" (an
# instruction difference, whose changed lines the caller counts with diff); "summary TAB text"; "result TAB failures";
# "error TAB text" instead of all that when a function or a constant section has no hash (an equal empty hash on both
# sides must never pass).
# Copies of one name are matched greedily, identical ones first: identity is an equivalence, so no better pairing
# exists; the copies left over are paired in order (DIFFERENT) or stand alone (MISSING, new).
all_match() {
    local w="$1"
    local files=( "$w/labels" "$w/base/hash" "$w/base/sass.idx" "$w/base/const.idx"
                  "$w/new/hash" "$w/new/sass.idx" "$w/new/const.idx" "$w/new/res.idx" )
    [ -n "$PREV_DIR" ] && files+=( "$w/prev/hash" "$w/prev/sass.idx" "$w/prev/const.idx" )
    awk -F '\t' -v haveprev="${PREV_DIR:+1}" -v list="$LIST" '
        function dir_of(p) { sub(/\/[^\/]*$/, "", p); return p }
        function lab(name) { return (name in label) ? label[name] : name }
        # Copy o of `name` on side s: its constant bank 2 (hash, "" without one) and its words. A name with one copy and
        # one section takes that section; copies of a repeated name take their own cubin'"'"'s
        function section(s, name, o) {
            return (copies[s, name] == 1 && sections[s, name] == 1) ? only_section[s, name] : cubin_of[s, name, o]
        }
        function ch(s, name, o) { return chash[s, name, section(s, name, o)] }
        function words(s, name, o) { return cwords[s, name, section(s, name, o)] + 0 }
        function same(s, o, t, k, name) {
            return ih[s, name, o] == ih[t, name, k] && ch(s, name, o) == ch(t, name, k)
        }
        # The first copy of `name` on side t not taken yet that equals copy o on side s, now taken; 0 when none
        function take_same(s, o, t, name,    k) {
            for (k = 1; k <= copies[t, name] + 0; k++)
                if (!((t, name, k) in taken) && same(s, o, t, k, name)) { taken[t, name, k] = 1; return k }
            return 0
        }
        # The first copy of `name` on side t not taken yet, now taken; 0 when none
        function take_any(t, name,    k) {
            for (k = 1; k <= copies[t, name] + 0; k++)
                if (!((t, name, k) in taken)) { taken[t, name, k] = 1; return k }
            return 0
        }
        function resources(name, k,    c) {
            if (copies["new", name] == 1 && listed[name] == 1) return only_res[name]
            c = cubin_of["new", name, k]
            return ((name, c) in res) ? res[name, c] : "no resources listed"
        }
        function out(text) { print "line\t" text }

        FNR == 1 { n = split(FILENAME, part, "/"); kind = part[n]; side = part[n - 1] }
        kind == "labels" { label[$1] = $2; next }
        kind == "hash" {
            h = $0; sub(/[ \t].*$/, "", h)
            f = $0; sub(/^[^ \t]+[ \t]+\*?/, "", f)
            hash[side, f] = h
            next
        }
        kind == "sass.idx" {
            if (!((side, "f" $2 ".sass") in hash)) { broken = side " f" $2 ".sass (" $1 ")"; exit }
            o = ++copies[side, $1]
            if (o == 1) names[side, ++named[side]] = $1
            ih[side, $1, o] = hash[side, "f" $2 ".sass"]
            count[side, $1, o] = $3
            cubin_of[side, $1, o] = $4
            file[side, $1, o] = dir_of(FILENAME) "/f" $2 ".sass"
            next
        }
        kind == "const.idx" {
            if (!((side, "c" $2 ".const") in hash)) { broken = side " c" $2 ".const (" $1 ")"; exit }
            chash[side, $1, $4] = hash[side, "c" $2 ".const"]
            cwords[side, $1, $4] = $3
            sections[side, $1]++
            only_section[side, $1] = $4
            next
        }
        kind == "res.idx" { res[$1, $3] = $2; listed[$1]++; only_res[$1] = $2; next }

        END {
            if (broken != "") { print "error\tno hash for " broken; exit }
            # BASE functions: each copy needs an identical copy in NEW
            for (i = 1; i <= named["base"] + 0; i++) {
                name = names["base", i]
                nbase += copies["base", name]
                left_count = 0
                for (o = 1; o <= copies["base", name]; o++) {
                    k = take_same("base", o, "new", name)
                    if (!k) { left[++left_count] = o; continue }
                    identical++
                    if (list)
                        out(sprintf("identical: %s (%d instructions, %d words of constants)", lab(name),
                                    count["new", name, k], words("new", name, k)))
                }
                for (u = 1; u <= left_count; u++) {
                    o = left[u]
                    k = take_any("new", name)
                    if (!k) {
                        missing++
                        out(sprintf("MISSING: %s (%d instructions in BASE)", lab(name), count["base", name, o]))
                    } else if (ih["base", name, o] != ih["new", name, k]) {
                        different++
                        print "diff\tDIFFERENT: " lab(name) "\t" file["base", name, o] "\t" file["new", name, k] "\t"
                    } else {
                        different++
                        out("DIFFERENT constants: " lab(name) " (its instructions are identical, its constant " \
                            "bank 2 isn'"'"'t)")
                    }
                }
                # PREV copies of BASE functions are no candidates for the functions NEW adds
                if (haveprev) for (o = 1; o <= copies["base", name]; o++) take_same("base", o, "prev", name)
            }
            # The functions NEW adds: compared with PREV where it has a copy left, otherwise new
            for (i = 1; i <= named["new"] + 0; i++) {
                name = names["new", i]
                left_count = 0
                for (k = 1; k <= copies["new", name]; k++) {
                    if (("new", name, k) in taken) continue
                    if (haveprev && take_same("new", k, "prev", name)) {
                        same_prev++
                        out(sprintf("identical to --prev: %s (%d instructions; %s)", lab(name), count["new", name, k],
                                    resources(name, k)))
                    } else left[++left_count] = k
                }
                for (u = 1; u <= left_count; u++) {
                    k = left[u]
                    p = haveprev ? take_any("prev", name) : 0
                    if (!p) {
                        added++
                        out(sprintf("new: %s (%d instructions; %s)", lab(name), count["new", name, k],
                                    resources(name, k)))
                    } else if (ih["prev", name, p] != ih["new", name, k]) {
                        different_prev++
                        print "diff\tDIFFERENT from --prev: " lab(name) "\t" file["prev", name, p] "\t" \
                              file["new", name, k] "\t; " resources(name, k)
                    } else {
                        different_prev++
                        out("DIFFERENT constants from --prev: " lab(name) " (its instructions are identical, its " \
                            "constant bank 2 isn'"'"'t; " resources(name, k) ")")
                    }
                }
            }
            # PREV functions NEW no longer has
            if (haveprev) for (i = 1; i <= named["prev"] + 0; i++) {
                name = names["prev", i]
                for (p = 1; p <= copies["prev", name]; p++)
                    if (!(("prev", name, p) in taken)) { gone++; out("gone since --prev: " lab(name)) }
            }
            # The cross-check
            for (i = 1; i <= named["base"] + 0; i++) {
                name = names["base", i]
                if (name !~ /^_ZN3gpu6detail14quality_kernelE/) continue
                out(sprintf("quality_kernel: %d instructions in BASE, %s in NEW (Step 85: 6864)",
                            count["base", name, 1], (("new", name, 1) in count) ? count["new", name, 1] : "none"))
            }
            summary = sprintf("%d of %d functions of BASE identical in instructions and constants", identical, nbase)
            if (missing) summary = summary sprintf(", %d MISSING", missing)
            if (different) summary = summary sprintf(", %d DIFFERENT", different)
            summary = summary sprintf("; %d new", added)
            if (haveprev)
                summary = summary sprintf(", %d identical to --prev, %d DIFFERENT from --prev, %d gone since --prev",
                                          same_prev, different_prev, gone)
            print "summary\t" summary
            print "result\t" missing + different + different_prev
        }' "${files[@]}"
}

# Keep a dump for a later --prev (nothing to do when it is already there)
all_save() {
    [ "$1" -ef "$2" ] && return 0
    cp "$1" "$2" || { echo "sassgate.sh: could not write $2" >&2; exit 2; }
}

all_mode() {
    shopt -s nullglob
    [ $# -ge 2 ] || all_usage
    BASE_DIR="$1"
    NEW_DIR="$2"
    shift 2
    PREV_DIR=""
    SAVE_DIR=""
    LIST=""
    local exes="RayTracingGPU,RayTracingViewer,GpuTests"
    while [ $# -gt 0 ]; do
        case "$1" in
        --prev) [ $# -ge 2 ] || all_usage; PREV_DIR="$2"; shift 2 ;;
        --save) [ $# -ge 2 ] || all_usage; SAVE_DIR="$2"; shift 2 ;;
        --exes) [ $# -ge 2 ] || all_usage; exes="$2"; shift 2 ;;
        --list) LIST=1; shift ;;
        *) echo "sassgate.sh: unknown argument '$1'" >&2; all_usage ;;
        esac
    done
    local -a programs
    IFS=',' read -r -a programs <<< "$exes"
    [ ${#programs[@]} -gt 0 ] || all_usage
    if [ -n "$SAVE_DIR" ] && [ -d "$SAVE_DIR" ]; then
        if [ "$SAVE_DIR" -ef "$BASE_DIR" ] || { [ -n "$PREV_DIR" ] && [ "$SAVE_DIR" -ef "$PREV_DIR" ]; }; then
            echo "sassgate.sh: --save $SAVE_DIR would replace the dumps of BASE_DIR or PREV_DIR" >&2
            exit 2
        fi
    fi

    # Every input first: a missing program or dump stops the run before anything is compared
    local exe dump_new=""
    for exe in "${programs[@]}"; do
        [ -n "$exe" ] || { echo "sassgate.sh: an empty name in --exes $exes" >&2; exit 2; }
        all_need "$BASE_DIR/$exe.sass.txt"
        all_need "$BASE_DIR/$exe.elf.txt"
        if [ -f "$NEW_DIR/$exe.sass.txt" ]; then
            all_need "$NEW_DIR/$exe.elf.txt"
            all_need "$NEW_DIR/$exe.res.txt"
            # Dumps beside an exe built after them are some older build's: never compared in its place
            if [ -f "$NEW_DIR/$exe.exe" ] && [ "$NEW_DIR/$exe.exe" -nt "$NEW_DIR/$exe.sass.txt" ]; then
                echo "sassgate.sh: $NEW_DIR/$exe.exe is newer than the dumps beside it, an older build's" >&2
                exit 2
            fi
        else
            all_need "$NEW_DIR/$exe.exe"
            dump_new=1
        fi
        if [ -n "$PREV_DIR" ]; then
            all_need "$PREV_DIR/$exe.sass.txt"
            all_need "$PREV_DIR/$exe.elf.txt"
        fi
    done
    all_find_cuda
    if [ -n "$dump_new" ] && { [ -z "$CUDA_BIN" ] || [ ! -x "$CUDA_BIN/cuobjdump.exe" ]; }; then
        echo "sassgate.sh: cuobjdump.exe not found in ${CUDA_BIN:-the CUDA Toolkit} (set CUDA_PATH)" >&2
        exit 2
    fi

    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
    local failures=0 w new_sass new_elf new_res new_from
    for exe in "${programs[@]}"; do
        w="$WORK/$exe"
        mkdir -p "$w"
        if [ -f "$NEW_DIR/$exe.sass.txt" ]; then
            new_sass="$NEW_DIR/$exe.sass.txt"
            new_elf="$NEW_DIR/$exe.elf.txt"
            new_res="$NEW_DIR/$exe.res.txt"
            new_from="the dumps in $NEW_DIR"
        else
            new_sass="$w/new.sass.txt"
            new_elf="$w/new.elf.txt"
            new_res="$w/new.res.txt"
            all_dump -sass "$NEW_DIR/$exe.exe" "$new_sass"
            all_dump -elf "$NEW_DIR/$exe.exe" "$new_elf"
            all_dump -res-usage "$NEW_DIR/$exe.exe" "$new_res"
            new_from="$NEW_DIR/$exe.exe"
        fi
        echo "$exe: BASE the dumps in $BASE_DIR, NEW $new_from${PREV_DIR:+, --prev the dumps in $PREV_DIR}"
        all_index "$BASE_DIR/$exe.sass.txt" "$BASE_DIR/$exe.elf.txt" "" "$w/base"
        all_index "$new_sass" "$new_elf" "$new_res" "$w/new"
        [ -n "$PREV_DIR" ] && all_index "$PREV_DIR/$exe.sass.txt" "$PREV_DIR/$exe.elf.txt" "" "$w/prev"
        if [ ! -s "$w/base/sass.idx" ]; then
            echo "sassgate.sh: no function in $BASE_DIR/$exe.sass.txt (not a cuobjdump -sass dump?)" >&2
            exit 2
        fi
        # Saved only now, after every side was read: --save can't change what this run compares
        if [ -n "$SAVE_DIR" ]; then
            mkdir -p "$SAVE_DIR"
            all_save "$new_sass" "$SAVE_DIR/$exe.sass.txt"
            all_save "$new_elf" "$SAVE_DIR/$exe.elf.txt"
            all_save "$new_res" "$SAVE_DIR/$exe.res.txt"
        fi

        # The same GPU architectures in both
        local arch_base arch_new
        arch_base="$(sort -u "$w/base/arch" | sed 's/^arch = //' | tr '\n' ' ')"
        arch_new="$(sort -u "$w/new/arch" | sed 's/^arch = //' | tr '\n' ' ')"
        if [ "$arch_base" = "$arch_new" ]; then
            echo "architectures: ${arch_new% }"
        else
            echo "DIFFERENT architectures: NEW ${arch_new% }, BASE ${arch_base% }"
            failures=$((failures + 1))
        fi

        # BASE is a real build: the five Step 70 kernels (the same patterns as the two-argument form's)
        if [ "$exe" = RayTracingGPU ]; then
            local pattern matched found=0
            for pattern in 'first_book_kernelIfE' 'first_book_kernelIdE' 'all_books_kernelIfE' 'all_books_kernelIdE' \
                           '^_ZN3gpu6detail12rgba8_kernelEPKdiiidPh$'; do
                matched=$(cut -f1 "$w/base/sass.idx" | grep -cE -- "$pattern")
                if [ "$matched" -eq 1 ]; then
                    found=$((found + 1))
                else
                    echo "BASELINE: $matched functions of BASE match $pattern (expected 1)"
                    failures=$((failures + 1))
                fi
            done
            echo "the Step 70 kernels: $found of 5 patterns name exactly one function of BASE"
        fi

        cut -f1 "$w"/*/sass.idx | sort -u > "$w/names"
        all_labels "$w/names" "$w/labels"
        local tag a b c d summary="" result=""
        while IFS=$'\t' read -r tag a b c d; do
            case "$tag" in
            line) echo "$a" ;;
            diff) echo "$a ($(diff "$b" "$c" | grep -c '^[<>]') lines differ$d)" ;;
            summary) summary="$a" ;;
            result) result="$a" ;;
            error) echo "sassgate.sh: $exe: $a" >&2; exit 2 ;;
            esac
        done < <(all_match "$w")
        # No verdict from the comparison (awk stopped) is an error, never a pass
        if [ -z "$result" ]; then
            echo "sassgate.sh: $exe: the comparison gave no result" >&2
            exit 2
        fi
        echo "$exe: $summary"
        failures=$((failures + result))
    done

    if [ "$failures" -eq 0 ]; then
        echo "SASS gate: PASS (--all: ${exes//,/, })"
        exit 0
    fi
    echo "SASS gate: FAIL ($failures, --all: ${exes//,/, })"
    exit 1
}

if [ "${1:-}" = "--all" ]; then
    shift
    all_mode "$@"
fi

# --- The two-argument form -------------------------------------------------------------------------------------------

if [ $# -lt 2 ]; then
    echo "usage: sassgate.sh NEW.exe BASE.sass.txt [--base-exe BASE.exe] [--prev PREV.sass.txt]" \
         "[--save OUT.sass.txt]" >&2
    echo "       sassgate.sh --all BASE_DIR NEW_DIR [--prev PREV_DIR] [--save OUT_DIR] [--exes LIST] [--list]" >&2
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
