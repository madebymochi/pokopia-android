#!/usr/bin/env bash
# Build the shader translator (dm/gfx/tools/dmsh_xlate and the Ryujinx sources in the repository) for Windows from a copy of the sources, so vendor/nx stays free of obj/ and bin/.
# NuGet packages and .NET's home stay in .cache. Env: DOTNET = dotnet.exe with the .NET 10 SDK. Output: .cache/tools/xlate/dmsh_xlate.dll
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NX="$ROOT/vendor/nx"
DOTNET="${DOTNET:?set DOTNET to a dotnet.exe with the .NET 10 SDK}"
COPY="$ROOT/.cache/tools/xlate-src"; OUT="$ROOT/.cache/tools/xlate"
rm -rf "$COPY" "$OUT"; mkdir -p "$COPY/dm/gfx/tools" "$COPY/dm/third_party" "$OUT"
cp -r "$NX/dm/gfx/tools/dmsh_xlate" "$COPY/dm/gfx/tools/"
cp -r "$NX/dm/third_party/ryujinx" "$COPY/dm/third_party/"
export NUGET_PACKAGES="$(cygpath -w "$ROOT/.cache/nuget")" DOTNET_CLI_HOME="$(cygpath -w "$ROOT/.cache/dotnet-home")" DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
"$DOTNET" publish "$(cygpath -w "$COPY/dm/gfx/tools/dmsh_xlate/dmsh_xlate.csproj")" -c Release -r win-x64 --self-contained false -o "$(cygpath -w "$OUT")" -v quiet -nologo
[ -f "$OUT/dmsh_xlate.dll" ] || { echo "no dmsh_xlate.dll" >&2; exit 1; }
rm -rf "$COPY"
echo "translator: $OUT/dmsh_xlate.dll"
