# Third-party notices

This project builds on the following projects. Their notices follow.

- [seedds/nxnative](https://github.com/seedds/nxnative) (MIT), revision e5620b1a05: the Switch host this port starts from, and its Horizon services, code rewriter and shader translator (built on Ryujinx's shader library, MIT). The patches under `patches/` modify it, the tools call it, the Android host compiles its sources, `android-host/app/touch_controls.c` is a C port of its `DmGamepad.m`, and the shader job files in `android-host/tools/pokopia_spirv.py` follow its `dmsh_build.py`. Its license text below is copied from its `LICENSE` file.
- [xFractals/pokopia-native](https://github.com/xFractals/pokopia-native) (MIT), revision 87a3191bf5: the Switch 2 stubs in `patches/nxnative` (`nn::streamplay`, Nmouse, Jcms state, the vibration graph objects, `GetServerEnvironmentType`) and values in the game profile. Its GPL-3.0-or-later `ampere-lower/` folder is not part of this project: it was read for how instructions behave, no code copied.
- The Android app a build makes also compiles the third-party code nxnative ships in `vendor/nx/dm/third_party` (libvpx for the game's movies, stb_vorbis for its sound, the Vulkan headers). It is cloned at build time, not part of this repository; its license texts are in that folder.
- [BluetufieBugado/SuperMarioWonderRecomp](https://github.com/BluetufieBugado/SuperMarioWonderRecomp) (MIT): the BC7 decoder in `android-host/gfx/bc7.c` is ported to C from its `runtime/src/nvn/nvn_bc.cpp`. The BC6H decoder in that file is this project's own, written from the format's description.
- [Noto Sans CJK](https://github.com/notofonts/noto-cjk) (SIL Open Font License 1.1): not included in this repository. The build downloads it into `.cache/font/` and subsets it into your own build, with `tools/cjk_shared_font.py`.
- [d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile) by jhaizhou (MIT) and [lordran-native](https://github.com/Nishikori-Yui/lordran-native) by Nishikori-Yui (MIT): thanks for the native-execution idea.
- Read for how instructions behave, no code copied: [drakolordx7/pkp-recomp](https://github.com/drakolordx7/pkp-recomp) (GPL-3.0, revision 6980bb058a) and [moceure/eden-nx2-android](https://git.eden-emu.dev/moceure/eden-nx2-android) (GPL-3.0, revision 565253831a).

## xFractals/pokopia-native

```
MIT License

Copyright (c) 2026 the pokopia-native contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## seedds/nxnative

```
MIT License

Copyright (c) 2026 seedds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## BluetufieBugado/SuperMarioWonderRecomp

```
MIT License

Copyright (c) 2026 BelmanteGu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Ryujinx

The shader library nxnative ships in `dm/third_party/ryujinx`, which the patch in `patches/nxnative` changes.

```
MIT License

Copyright (c) Ryujinx Team and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
