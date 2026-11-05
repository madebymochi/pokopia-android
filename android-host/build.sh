#!/usr/bin/env bash
# Build the port's Android host (arm64) on a Windows PC (Git Bash) with the NDK, from vendor/nx + android-host/compat. android-host/build_android.py runs it.
# Usage: android-host/build.sh [compile|link|all]   (env: ANDROID_SDK or NDK, PY = python.exe with the packages of requirements.txt, PROFILE = profile dir with game.tsv
#   (default .cache/profile/0400A3A01EE4C000, from tools/make_profile.py), J = parallel jobs)
# Needs vendor/nx, from tools/setup_vendor.py. The build reads no game data: game_profile.h is generated from the profile.
# Output: .cache/android/obj-apk/*.o, .cache/android/libdmhost.so (the NativeActivity library; android-host/build_apk.sh packs it)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${NDK:-${ANDROID_SDK:?set ANDROID_SDK (or NDK); android-host/build_android.py does}/ndk/28.2.13676358}"
CC="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/clang.exe"
PROFILE="${PROFILE:-$ROOT/.cache/profile/0400A3A01EE4C000}"; PY="${PY:-python}"
DM="$ROOT/vendor/nx/dm"; H="$DM/host"; G="$DM/gfx"; C="$ROOT/android-host/compat"
GLUE="$NDK/sources/android/native_app_glue"
OUT="$ROOT/.cache/android"; OBJ="$OUT/obj-apk"; mkdir -p "$OBJ" "$OUT/asm" "$OUT/log" "$OUT/gen"
GLSLC="$NDK/shader-tools/windows-x86_64/glslc.exe"
gen_shaders() {   # gfx/present.{vert,frag} -> $OUT/gen/present_spv.h (uint32_t arrays for nvn_vk.c's present composite)
  local h="$OUT/gen/present_spv.h"; : > "$h.tmp"
  for s in vert frag; do
    "$GLSLC" -O --target-env=vulkan1.3 -mfmt=num -o "$OUT/gen/present_$s.inc" "$ROOT/android-host/gfx/present.$s" || return 1
    { echo "static const uint32_t present_${s}_spv[] = {"; cat "$OUT/gen/present_$s.inc"; echo "};"; } >> "$h.tmp"
  done; mv "$h.tmp" "$h"
}
gen_profile() {   # the profile's game.tsv -> $OUT/gen/game_profile.h (nxnative's dm.h includes it); written only when it changes
  DM_GAMES_DIR="$(cygpath -m "$(dirname "$PROFILE")")" DM_GAMES_FIRST=1 "$PY" -X utf8 -I "$DM/games/game.py" header "$(basename "$PROFILE")" "$(cygpath -m "$OUT/gen/game_profile.h")"
}
# The rewriter's identity (its sources, carriage returns dropped; android-host/build_android.py REWRITER_FILES hashes the same files in the same order and writes it to mods/rewriter.txt):
# the gate numbers and the thread pointer register (nso_asm.py) in the rewritten modules must be the ones this host handles
REWRITER_ID="$(cd "$DM/tools" && cat patch_nso.py nso.py nso_code.py nso_asm.py nso_match.py | tr -d '\r' | sha256sum | cut -c1-16)"
J="${J:-$(nproc 2>/dev/null || echo 8)}"
TARGET=--target=aarch64-linux-android31
PIC=-fPIC
PREFIX_MAP="-ffile-prefix-map=$(cygpath -m "$ROOT")=."   # no build folder in the library: __FILE__ in asserts would put it there
CFLAGS=($PREFIX_MAP -I"$OUT/gen" -O1 -g -Wall -Wno-unused-function -Wno-format -Wno-unused-variable -Wno-unused-but-set-variable -fno-strict-aliasing $PIC -I"$GLUE"
  -D_GNU_SOURCE -DDM_REWRITER_ID=\"$REWRITER_ID\" -include "$C/darwin_compat.h" -I"$C/include" -I"$PROFILE"
  -I"$G/nvn/include" -I"$DM/third_party/vulkan-headers/include" -I"$DM/third_party/libvpx" -I"$DM/third_party/stb_vorbis" -I"$H" -I"$G" -I"$G/nvn" -I"$G/tex")
# Apple-only sources replaced by android-host/stubs (Metal replay, Cocoa front end, AudioToolbox / VideoToolbox paths, Mach memory statistics, Vulkan-on-Metal)
SKIP=" hle_movie_avc.c nvn_capture.c hle_cheats.c "
SRC=()
for f in "$H"/*.c "$G"/*.c "$G"/nvn/*.c "$G"/tex/*.c "$C"/src/*.c "$ROOT"/android-host/stubs/*.c "$ROOT"/android-host/gfx/*.c "$ROOT"/android-host/app/*.c; do
  [ -f "$f" ] || continue; case "$SKIP" in *" $(basename "$f") "*) continue;; esac; SRC+=("$f"); done
compile_one() {
  local f="$1" o="$OBJ/$(basename "$(dirname "$f")")_$(basename "${f%.*}").o"
  "$CC" $TARGET "${CFLAGS[@]}" -c "$f" -o "$o" 2> "$OUT/log/$(basename "$o" .o).log" || { echo "FAIL $(basename "$f")"; return 1; }
}
asm_one() {   # Mach-O assembly syntax -> ELF, then assemble
  local f="$1" e="$OUT/asm/$(basename "$f")" o="$OBJ/asm_$(basename "${f%.*}").o"
  "$PY" "$ROOT/android-host/tools/macho_asm_to_elf.py" "$f" "$e" && \
  "$CC" $TARGET $PIC -c "$e" -o "$o" 2> "$OUT/log/asm_$(basename "${f%.*}").log" || { echo "FAIL $(basename "$f")"; return 1; }
}
do_compile() {
  gen_shaders || { echo "FAIL shaders"; return 1; }
  gen_profile || { echo "FAIL profile header"; return 1; }
  rm -f "$OBJ"/*.o; : > "$OUT/log/_compile.txt"
  for f in "$H/entry.S" "$G/nvn/nvn_entry.S" "$G/nvn/nvn_thunks.S"; do asm_one "$f" >> "$OUT/log/_compile.txt" & done
  for f in "${SRC[@]}"; do
    compile_one "$f" >> "$OUT/log/_compile.txt" &
    while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
  done; wait
  local fails; fails=$(grep -c FAIL "$OUT/log/_compile.txt"); sort "$OUT/log/_compile.txt"
  echo "compiled ${#SRC[@]} C files + 3 asm, failures: $fails"; [ "$fails" = 0 ]
}
VPX="$DM/third_party/libvpx"; VPXO="$OUT/vpx"   # -fPIE objects: also fine in the shared library (-fPIE code is position independent; it only assumes its own symbols are not preempted)
do_vpx() {   # libvpx VP9 decoder (hle_movie.c), built once into a static archive (its config/ headers are arm64, shared by Mac / iOS / here)
  [ -f "$VPXO/libvpx.a" ] && return 0
  mkdir -p "$VPXO"; local n=0 fails=0
  while IFS= read -r f; do
    local o="$VPXO/$(echo "${f#$VPX/}" | tr '/' '_').o"
    "$CC" $TARGET $PREFIX_MAP -O2 -fPIE -I"$VPX" -I"$VPX/config" -c "$f" -o "$o" 2>> "$OUT/log/_vpx.log" || fails=$((fails+1)) &
    n=$((n+1)); while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
  done < <(find "$VPX" -name '*.c'); wait
  "$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-ar.exe" rcs "$VPXO/libvpx.a" "$VPXO"/*.o && echo "libvpx: $n files"
}
do_link() {
  do_vpx || return 1
  local dst="$OUT/libdmhost.so"
  # -z defs: a symbol no library defines fails the link here; a shared library links without it and then fails to load on the phone
  # (ANativeWindow_setFrameRate is in libnativewindow, not libandroid: the APK closed at once, "dlopen failed: cannot locate symbol")
  "$CC" $TARGET -shared -Wl,-soname,libdmhost.so -Wl,-Bsymbolic -Wl,-u,ANativeActivity_onCreate -Wl,-z,max-page-size=16384 -Wl,-z,defs -Wl,--error-limit=0 -Wl,--wrap=mkdir -o "$dst" "$OBJ"/*.o "$VPXO/libvpx.a" -lm -ldl -llog -lz -laaudio -lvulkan -landroid -lnativewindow 2> "$OUT/log/_link.log"; local r=$?
  [ $r = 0 ] && echo "linked $dst" || { echo "link failed:"; grep -E "undefined symbol|error" "$OUT/log/_link.log" | sort | uniq -c | sort -rn | head -60; }
  return $r
}
case "${1:-all}" in
  compile) do_compile;; link) do_link;; all) do_compile && do_link;;
esac
