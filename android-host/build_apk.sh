#!/usr/bin/env bash
# Pack .cache/android/libdmhost.so (android-host/build.sh all) into a signed NativeActivity APK: aapt2 link -> add the library -> zipalign -> apksigner.
# Usage: android-host/build_apk.sh [--install]   Output: .cache/android/apk/pokopia.apk. The APK holds no game data, keys or shaders (all read from the phone).
# The signing key is KEYSTORE (android-host/build_android.py sets it, outside the checkout), made once with keytool
# when missing; an APK signed with another key cannot update the installed app. JDK from PATH (java / keytool).
# Env (build_android.py sets them): ANDROID_SDK, NDK, BUILD_TOOLS, ANDROID_JAR (default: what SDK_PACKAGES in build_android.py names), KEYSTORE,
#   APK_VERSION_CODE / APK_VERSION_NAME; DEBUG_APK=1 makes the APK debuggable (development only: any PC the phone trusts could then run-as the app).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_SDK:?set ANDROID_SDK; android-host/build_android.py does}"
BT="${BUILD_TOOLS:-$SDK/build-tools/35.0.0}"; JAR="${ANDROID_JAR:-$SDK/platforms/android-35/android.jar}"
NDK="${NDK:-$SDK/ndk/28.2.13676358}"; STRIP="$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip.exe"
OUT="$ROOT/.cache/android/apk"; mkdir -p "$OUT/stage/lib/arm64-v8a"
w() { cygpath -m "$1"; }
# Java reads its command line in the Windows ANSI code page, so a user name with other letters breaks a key path under %LOCALAPPDATA% (keytool could not make the key
# there): keytool and apksigner work on a copy in .cache, which is deleted at the end
KEY="${KEYSTORE:-$OUT/debug.keystore}"; KS="$OUT/sign.keystore"; mkdir -p "$(dirname "$KEY")"; rm -f "$KS"
trap 'rm -f "$KS"' EXIT   # a failed step (set -e) leaves no copy of the key behind
if [ -f "$KEY" ]; then cp "$KEY" "$KS"
else
  keytool -genkeypair -keystore "$(w "$KS")" -storepass android -keypass android -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US" >/dev/null     && cp "$KS" "$KEY.tmp" && mv "$KEY.tmp" "$KEY" || { echo "could not make the signing key $KEY" >&2; exit 1; }
fi
"$STRIP" --strip-debug -o "$(w "$OUT/stage/lib/arm64-v8a/libdmhost.so")" "$(w "$ROOT/.cache/android/libdmhost.so")"
# Backup rules (Android 12+): only the save, so Android's backup and phone-to-phone transfer carry the save and not the game files
rm -rf "$OUT/res" "$OUT/res.zip"; mkdir -p "$OUT/res/xml"
cat > "$OUT/res/xml/data_extraction_rules.xml" <<'EOF'
<?xml version="1.0" encoding="utf-8"?>
<data-extraction-rules>
    <cloud-backup><include domain="external" path="save"/></cloud-backup>
    <device-transfer><include domain="external" path="save"/></device-transfer>
</data-extraction-rules>
EOF
"$BT/aapt2.exe" compile -o "$(w "$OUT/res.zip")" --dir "$(w "$OUT/res")"   # (aapt2 on Windows rejects a single resource file given with / separators: --dir it is)
DBG=(); [ "${DEBUG_APK:-0}" = 1 ] && DBG=(--debug-mode)
"$BT/aapt2.exe" link -o "$(w "$OUT/base.apk")" -I "$(w "$JAR")" --manifest "$(w "$ROOT/android-host/app/AndroidManifest.xml")" --min-sdk-version 31 --target-sdk-version 34 \
  --version-code "${APK_VERSION_CODE:-1}" --version-name "${APK_VERSION_NAME:-dev}" ${DBG[@]+"${DBG[@]}"} "$(w "$OUT/res.zip")"
"${PY:-python}" -I -c "import sys, zipfile; z = zipfile.ZipFile(sys.argv[1], 'a', zipfile.ZIP_DEFLATED); z.write(sys.argv[2], 'lib/arm64-v8a/libdmhost.so'); z.close()" "$(w "$OUT/base.apk")" "$(w "$OUT/stage/lib/arm64-v8a/libdmhost.so")"
"$BT/zipalign.exe" -f -P 16 4 "$(w "$OUT/base.apk")" "$(w "$OUT/aligned.apk")"
# apksigner.jar straight on java, with apksigner.bat's own JVM options: through cmd //c, a path with a space
# split at the space; and with MSYS_NO_PATHCONV set, cmd got //c, signed nothing, exited 0, and the old APK was installed
java -Xmx1024M -Xss1m -jar "$(w "$BT/lib/apksigner.jar")" sign --ks "$(w "$KS")" --ks-pass pass:android --key-pass pass:android --out "$(w "$OUT/pokopia.apk")" "$(w "$OUT/aligned.apk")"
[ "$OUT/pokopia.apk" -nt "$OUT/aligned.apk" ] || { echo "apksigner did not write a new pokopia.apk" >&2; exit 1; }
rm -f "$KS"
echo "APK: $OUT/pokopia.apk ($(du -h "$OUT/pokopia.apk" | cut -f1))"
if [ "${1:-}" = --install ]; then MSYS_NO_PATHCONV=1 "$SDK/platform-tools/adb.exe" install -r "$(w "$OUT/pokopia.apk")"; fi
