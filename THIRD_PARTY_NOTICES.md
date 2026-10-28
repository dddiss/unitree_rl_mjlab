# Third-party notices

## P-BFM depth perception and G1 terrain

The files `src/tasks/supermimic/perception/camera.py`,
`src/tasks/supermimic/perception/sensor.py`,
`src/tasks/supermimic/perception/preprocessing.py`, and
`src/tasks/supermimic/rl/depth_encoder.py` adapt the G1 depth-camera convention,
`optical_z_maxcrop18_reset_v2` preprocessing, and CNN/GRU encoder from
[RoboParty/P-BFM](https://github.com/Roboparty/P-BFM), revision
`2aa985ac50162e446532d1e7231f81902c221e49`. P-BFM builds on
[RoboParty/UFO](https://github.com/Roboparty/UFO) and retains its
Creative Commons Attribution-NonCommercial 4.0 International license.
The complete source license is retained in
[`src/tasks/supermimic/perception/LICENSE-P-BFM`](src/tasks/supermimic/perception/LICENSE-P-BFM).
These adapted portions remain subject to CC BY-NC 4.0, including attribution
and the NonCommercial restriction; the project's other license notices do not
replace it. Source authorship and project attribution are retained by the links
above and the source file headers.

Changes: scoped the camera and preprocessing to fixed, synchronous G1 sensing,
including camera-housing-only ray exclusion;
used PyTorch adaptive pooling in the encoder; added activation checkpointing and
bounded encoding batches. The next-depth decoder, visual dynamics training, and
SuperMimic integration are additions here, not claims about P-BFM's algorithm.

The files in `src/tasks/supermimic/terrain/` and
`src/tasks/supermimic/config/g1/terrain_cfg.py` additionally adapt the same
revision's G1 six-family terrain primitives, `terrain_ufo_v0` generation defaults,
273-ray reset grid, generator-padding ray handling, and local-ground/adaptive-lift
reset rule. They carry the same CC BY-NC 4.0 attribution and license above.
Changes: retained only the G1 mixed-terrain families, used project-local Python
configuration without Hydra, and applied height placement to SuperMimic's existing
reset XY/pose/velocity/joint sampling. The ground probe starts above the compiled
map's highest collision bound to cover reference positions outside their initial
tile; the final height rule is unchanged. P-BFM's region sampling, random heading,
lying resets, terrain observations, and reward definitions are not imported.

## supermimic_vibed

The SuperMimic implementation adapts code from [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed).

MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

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
