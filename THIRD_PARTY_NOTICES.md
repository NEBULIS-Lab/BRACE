# Third-Party Notices

This repository includes a small set of third-party components for demos and integrations.

## AirSim (PythonClient)

Files:
- `experiments/airsim/vendor/airsim/`

Provenance (vendored):
- Upstream repo and commit are recorded in `experiments/airsim/vendor/airsim/upstream.md`.

License:
- MIT, see `experiments/airsim/vendor/airsim/LICENSE`.

## Bulma (CSS)

Files:
- `docs/static/css/bulma.min.css`

License:
- MIT (noted in the file header comment).

## Font Awesome Free

Files:
- `docs/static/js/fontawesome.all.min.js`
- `docs/static/webfonts/*`

License:
- See the license note embedded in the SVG font files under `docs/static/webfonts/`.

## E-RECAP (vendored module)

Files:
- `e-recap/`

License:
- MIT, see `assets/licenses/E_RECAP_MIT.txt` (upstream). This repository itself is MIT, see `LICENSE`.

## OpenMARL (vendored snapshot; integrated)

Files:
- `third_party/openmarl/robofactory/`

Notes:
- This is an integrated snapshot used by the RoboFactory VLA track; it is **not** fully original to BRACE.
- Checkpoints/weights/datasets are not included.

License:
- MIT (upstream notice embedded below).

```
MIT License

Copyright (c) 2025 MARS-EAI

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

## Habitat (external dependency; not distributed)

This repository includes a Meta AI Habitat runner/integration, but does **not** vendor or redistribute Habitat code/assets.

Upstream:
- Habitat-Sim: `https://github.com/facebookresearch/habitat-sim`
- Habitat-Lab: `https://github.com/facebookresearch/habitat-lab`
- Habitat portal/docs: `https://aihabitat.org/`

## RoboFactory (external dependency; not distributed)

This repository includes a RoboFactory runner/integration.

- It does **not** ship RoboFactory simulator assets/checkpoints.
- For the VLA track, an integrated OpenMARL snapshot is vendored under `third_party/openmarl/robofactory/` (see above).

Upstream:
- RoboFactory: `https://github.com/MARS-EAI/RoboFactory`
