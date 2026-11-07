# Pokopia Android

Pokémon Pokopia (Nintendo Switch 2, version 2.0.1) running natively on Android. Not an emulator: the game's own ARM64 code runs on the phone. Your PC builds the app and the game files from your own dump. This repository has no game files or keys.

> [!WARNING]
> **This is a beta.** Tested on one phone (Galaxy S22, Android 16, 8 GB of RAM): a new game reaches the world and a saved game loads again, at 1080p, about 40 fps while the phone is cool and about 30 fps once it is warm. Long play is not tested yet, and neither are other phones. Expect bugs and back up your save. Offline, single player.

## You need

- **Game:** version 2.0.1, dumped from your own Switch 2, as files in this layout. NSP files are not read.
  ```
  <your dump folder>\updates\655360\exefs\     main, sdk, subsdk0, ...
  <your dump folder>\updates\655360\romfs\     the patched RomFS, the tree the game sees
  ```
- **PC:** 64-bit Windows 10 or 11, 60 GB free, with these on it:
  - [Git for Windows](https://git-scm.com/download/win)
  - [Python](https://www.python.org/downloads/windows/) 3.13, the version this is tested with. 3.12 and 3.14 have every package but are untested. 3.15 has no Windows package for `lz4` yet, so its install fails.
  - [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
  - a [JDK](https://adoptium.net/), with `java` and `keytool` on PATH
  - the [Android SDK command-line tools](https://developer.android.com/studio#command-line-tools-only), then:
    ```
    sdkmanager "ndk;28.2.13676358" "build-tools;35.0.0" "platforms;android-35" "platform-tools"
    ```
- **Phone:** arm64, Android 12 or newer, Vulkan 1.3 with `VK_EXT_external_memory_host` and `VK_EXT_extended_dynamic_state3`, 20 GB free.

## Build

In PowerShell:

```
git clone https://github.com/madebymochi/pokopia-android
cd pokopia-android
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python tools\setup_vendor.py
$env:ANDROID_SDK = "C:\path\to\android-sdk"
.venv\Scripts\python android-host\build_android.py --dump "C:\path\to\your dump folder"
```

The first build takes about 40 minutes on a fast PC, most of it translating about 99,000 shaders. Keep the PC awake. If it stops, run the last line again: finished steps are skipped. Everything made from the game stays in `.cache` and `output`, which Git ignores.

If `dotnet` is not on PATH, also set `$env:DOTNET` to your `dotnet.exe`.

## Install

1. On the phone, open **Settings**, **About phone** (on a Samsung, also **Software information**) and tap **Build number** 7 times.
2. Turn on **USB debugging** in **Developer options**, plug the phone in and tap **Allow**.
3. In the same window as **Build** (in a new one, `cd` to the folder and set `$env:ANDROID_SDK` again), run:
   ```
   .venv\Scripts\python android-host\build_android.py --dump "C:\path\to\your dump folder" --steps app --install --push
   ```
   It installs the app and copies about 17 GB to the phone. If the copy stops, run it again: it resumes. When you update, close the game on the phone first: the install stops if it is running.
4. Open the app and play. The first minutes stutter.

To update, run `git pull`, then the `setup_vendor.py` and `build_android.py` lines from **Build**, then step 3. Never uninstall the app on the phone: that deletes your save.

## Controls

On-screen touch controls, from nxnative. The button between − and + switches two layouts:

- **2D:** left stick, A B X Y, L R ZL ZR, − and +.
- **3D:** the same, plus a D-pad and L3 and R3. The right stick is not drawn: drag in the right third of the screen.

## Back up

- **Save:** close the game, then copy `Internal storage\Android\data\io.github.madebymochi.pokopia\files\save` to your PC. To restore, close the app fully and put the folder's contents back.
- **Signing key:** copy `%LOCALAPPDATA%\Pokopia Android\signing\debug.keystore` somewhere safe. An app signed with another key cannot update the one on your phone, and the only way out deletes the save. To use a key kept elsewhere, set `$env:KEYSTORE` to its path before building.

## Problems

- **The app closes at once:** look for `FATAL` in `log.txt`, in `Internal storage\Android\data\io.github.madebymochi.pokopia\files`.
- **A build step fails:** its log is in `.cache\logs`.
- **The game slows down after some minutes:** the phone is limiting its speed because of heat. It comes back after a rest.

Logs can show your folder paths, your local time and details of the game's files. Read them before you share them, and never share game files, keys or the built app.

## Credits and legal

Made by [madebymochi](https://github.com/madebymochi), on the Switch host of [nxnative](https://github.com/seedds/nxnative). Code under the [MIT License](LICENSE). Other projects used: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Not affiliated with Nintendo or The Pokémon Company. Pokémon and Nintendo Switch are their trademarks. Use it only with a game you dumped from your own console; whether you may do that depends on your local law. Never share what you build: it holds your copy of the game's code.
