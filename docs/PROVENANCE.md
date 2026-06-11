# Demo & Media Provenance (public)

This file tracks provenance metadata for the **small** media assets committed under `docs/static/`.
The goal is to make demo selection auditable without committing large raw artifacts.

Policy:
- Keep committed demo media **small** (GitHub Pages friendly).
- Record a stable checksum (`sha256`) for each committed asset.
- When available, record the **source run/config** used to produce the full-resolution artifact (kept outside this repo).

## `docs/static/videos/` (MP4)

| Asset | Size | sha256 | Source run/config (external; fill when known) |
|---|---:|---|---|
| `docs/static/videos/habitat_compare.mp4` | 582KB (595821B) | `84e631d60881744e1791f6d7db90570ad465cb97e6397823093062c321f9f9d1` | TBD |
| `docs/static/videos/robofactory_compare.mp4` | 544KB (556771B) | `b57991cf0341fb17acd1d0c04ffe6fd7cb6ff545ce8301d6c9019f201323d57b` | TBD |
| `docs/static/videos/airsim_compare.mp4` | 357KB (365360B) | `e14d8f1de694df976a5dded4b414522dea35e7e089e1b00220df63770eb301ef` | TBD |

## `docs/static/images/` (GIF/PNG)

| Asset | Size | sha256 | Source run/config (external; fill when known) |
|---|---:|---|---|
| `docs/static/images/habitat_compare.gif` | 3.0MB (3137852B) | `69c71945f7e0f16dda1c13b998e4c1d7077190dbd265606b2c0f29d981ca0a06` | TBD |
| `docs/static/images/robofactory_compare.gif` | 2.4MB (2418880B) | `2d2498ece26cca7bcec2355a402cd2d7c8d5cf6633589da9a32cffc4ed1fa7b7` | TBD |
| `docs/static/images/airsim_compare.gif` | 3.1MB (3194570B) | `b4d392ebe11452c6ff614c798f3f2b86e19bada1a28cb9d7ac383a43458f5f3a` | TBD |
| `docs/static/images/habitat_demo_frame.png` | 924KB (945737B) | `fd8a9b8954350d1993e0b1d11d6f415f38871f2435dc9f4dc1b521694ce3bcdb` | TBD |
| `docs/static/images/robofactory_demo_frame.png` | 1019KB (1042487B) | `54a5b7a34a68ed8ba54083484a91b63ab4eaeface573b81d059890ee7fc210d5` | TBD |
| `docs/static/images/airsim_demo_panel.png` | 2.3MB (2332739B) | `75f16178fad47fe0c0a7252484ca969feaec7bdc4614a699945bd3eaaa0973a5` | TBD |

## `docs/static/images/paper/` (paper-facing still images)

| Asset | Size | sha256 | Source run/config (external; fill when known) |
|---|---:|---|---|
| `docs/static/images/paper/brace_workflow_clean.png` | 360KB (368322B) | `ceb3bb2b0defd2b3fe7023074a0167fb0666e3d5aa5a57c8fcd438d42955bd76` | workflow schematic; TBD |
| `docs/static/images/paper/overview.png` | 760K (775231B) | `4567b0b5ceedfe1f93ed549f8e2c1c75b8c01d3920f6d8eb9a3ab4db2095dfab` | converted from `BRACE-icml2026-camera-ready/fig/overview.pdf` |
| `docs/static/images/paper/erecap_method_architecture.png` | 487KB (498873B) | `89e7eaea09bcefc19e291c7a210c88f4d75be29310b8423a48688eaa67266721` | method schematic; TBD |
| `docs/static/images/paper/meta_habitat_motivation_t2.png` | 951KB (973498B) | `3556f330cea099b32d4b887df8735d4384db6ea096a5d749f9742387148728b4` | Habitat public figure; TBD |
| `docs/static/images/paper/habitat_spnoise_nobrace_contact.png` | 1.8M (1821420B) | `170a37863f431a68246feec0e706dbf07d5fe74ed691050f2dfea9a2a4615d05` | copied from `BRACE-icml2026-camera-ready/fig/habitat_spnoise_nobrace_contact.png` |
| `docs/static/images/paper/latency_cdf_meta_habitat.png` | 40KB (40500B) | `a3b7fa9e7b8e578c9e65318cce3136ba38af3ec2bff520aa7d567004fe6e482b` | Habitat postprocess plot; TBD |
| `docs/static/images/paper/slo_violation_bar.png` | 25KB (25806B) | `0d8477bda4d77d6b03b13a1aacaf9e383135220cd0b959730ce9f0078e4567ca` | cross-domain postprocess plot; TBD |
| `docs/static/images/paper/microsoft_airsim_example_compare_frame.png` | 4.3MB (4466209B) | `462d3a1e54580e535c0d3e42a21c421efa231183f8e49d09dfc73919b40be738` | AirSim comparison still; TBD |
| `docs/static/images/paper/airsimnh_retained_collage_main.jpg` | 504K (515694B) | `f3c2d7c817e8b3cb5ceedefc5c7438abfe0d3caf242e4c683211af69aa033f98` | copied from `BRACE-icml2026-camera-ready/fig/airsimnh_retained_collage_main.jpg` |
| `docs/static/images/paper/robofactory_example_baseline.jpg` | 104K (104941B) | `9d947698475183ac049610765ed234712318094f9b2be7889bf848ed6ace73b3` | copied from `BRACE-icml2026-camera-ready/fig/robofactory_example_baseline.jpg` |
| `docs/static/images/paper/robofactory_example_method.jpg` | 104K (103169B) | `74760f0570cf08210818a48705c3b6bb012e8ab863a9331c08be986d2497eddb` | copied from `BRACE-icml2026-camera-ready/fig/robofactory_example_method.jpg` |
| `docs/static/images/paper/robofactory_example_method.png` | 493KB (504451B) | `b2735f20175b97ffb4311db24316a451217501d6802694232fb61d75ea3881c1` | RoboFactory method still; TBD |
| `docs/static/images/paper/robofactory_example_baseline.png` | 512KB (523951B) | `4bca96a5be700b0fa9e7c792b01b58c2c1bad01d80b732d887dbe21fa953cd7d` | RoboFactory baseline still; TBD |
| `docs/static/images/paper/take_fruit_real.png` | 708K (723752B) | `2902ae1e1cbbddce57e3daedc50d647f0a0beb61dbacc521aec053681fb84b2f` | converted from `BRACE-icml2026-camera-ready/fig/take_fruit_real.pdf` |
| `docs/static/images/paper/diagnostics_coordination_wait_robofactory.png` | 96K (96566B) | `bbc0e33a74fd11023e7d242bd8e06e009589a51d97957bb7db8f1662c0fdb481` | copied from `BRACE-icml2026-camera-ready/fig/diagnostics_coordination_wait_robofactory.png` |
| `docs/static/images/paper/diagnostics_slo_violation_meta_habitat.png` | 108K (106851B) | `15e4ceca87e70366c5781e6bfd16ecb27e6da728587e83a41b7f6488312031b7` | copied from `BRACE-icml2026-camera-ready/fig/diagnostics_slo_violation_meta_habitat.png` |
| `docs/static/images/paper/diagnostics_token_latency_robofactory.png` | 148K (149782B) | `52367beb46c3ae3d16a3c22fca4277a2f44af6c2b15852a089be1c4c8894b268` | copied from `BRACE-icml2026-camera-ready/fig/diagnostics_token_latency_robofactory.png` |
| `docs/static/images/paper/diagnostics_token_latency_meta_habitat.png` | 152K (153436B) | `4a4b6fe69ac2c080c5891009b91e78dc6c4bf74ef5ca1731926a096720f05c35` | copied from `BRACE-icml2026-camera-ready/fig/diagnostics_token_latency_meta_habitat.png` |

## `docs/static/images/paper/*_paper.*` (current website paper-source figures)

These assets are the current website-facing versions generated from the exact figure files used by `BRACE-icml2026-camera-ready/*.tex`. If the paper used a PDF, the web asset was converted from that PDF with its crop box; if the paper used JPG, the web asset was copied from the JPG.

| Asset | Size | sha256 | Source |
|---|---:|---|---|
| `docs/static/images/paper/overview_paper.png` | 752K (769002B) | `f811b7101be5ce8aa2d50b71a9899af2f1fbbe6f1723a1133f58e534f2c14007` | converted from `BRACE-icml2026-camera-ready/fig/overview.pdf` |
| `docs/static/images/paper/habitat_spnoise_nobrace_contact_paper.png` | 2.6M (2713619B) | `5082bba5081f51872103aea30b835c6199fe2ab454e25111408cc1f93e942f3b` | converted from `BRACE-icml2026-camera-ready/fig/habitat_spnoise_nobrace_contact.pdf` |
| `docs/static/images/paper/erecap_method_architecture_paper.png` | 456K (464137B) | `45a219a75970b47af33407080a4e9e7dcda82de143e58104c240215c8d972e60` | converted from `BRACE-icml2026-camera-ready/fig/erecap_method_architecture.pdf` |
| `docs/static/images/paper/latency_cdf_meta_habitat_paper.png` | 44K (42251B) | `41811bf8b887b7404e09ac8efd86c6979321b0182706468df12407466b4599b1` | converted from `BRACE-icml2026-camera-ready/fig/latency_cdf_meta_habitat.pdf` |
| `docs/static/images/paper/slo_violation_bar_paper.png` | 48K (45930B) | `ca66c88ba88963852a3fb6187a272e398033528b873990047df31c27d3378697` | converted from `BRACE-icml2026-camera-ready/fig/slo_violation_bar.pdf` |
| `docs/static/images/paper/robofactory_example_baseline_paper.jpg` | 104K (104941B) | `9d947698475183ac049610765ed234712318094f9b2be7889bf848ed6ace73b3` | copied from `BRACE-icml2026-camera-ready/fig/robofactory_example_baseline.jpg` |
| `docs/static/images/paper/robofactory_example_method_paper.jpg` | 104K (103169B) | `74760f0570cf08210818a48705c3b6bb012e8ab863a9331c08be986d2497eddb` | copied from `BRACE-icml2026-camera-ready/fig/robofactory_example_method.jpg` |
| `docs/static/images/paper/airsimnh_retained_collage_main_paper.png` | 1.1M (1143918B) | `dbb3e9bc1e2545677c9bf9f3f0351de79a309e3e98c79cabf95cb5de5c96d604` | converted from `BRACE-icml2026-camera-ready/fig/airsimnh_retained_collage_main.pdf` |
| `docs/static/images/paper/take_fruit_real_paper.png` | 832K (851531B) | `a9dafb9ac58c8ca479f4462448310207885a2f168c559cde11753e2a080605a2` | converted from `BRACE-icml2026-camera-ready/fig/take_fruit_real.pdf` |
| `docs/static/images/paper/diagnostics_coordination_wait_robofactory_paper.png` | 28K (24648B) | `628f8a9fa1bb3874a2cd55e924b091f4d55a7339e200b05bbe7a9b756f54b549` | converted from `BRACE-icml2026-camera-ready/fig/diagnostics_coordination_wait_robofactory.pdf` |
| `docs/static/images/paper/diagnostics_slo_violation_meta_habitat_paper.png` | 32K (28890B) | `fcea573527fb3712dde25f5ba87bc70f2e1e6814e8c07999ff767464a3f615d9` | converted from `BRACE-icml2026-camera-ready/fig/diagnostics_slo_violation_meta_habitat.pdf` |
| `docs/static/images/paper/diagnostics_token_latency_robofactory_paper.png` | 44K (43648B) | `a1250e33082e10232a240b13a67fde0168329ebe2d747dba0cc5d921873aa2f4` | converted from `BRACE-icml2026-camera-ready/fig/diagnostics_token_latency_robofactory.pdf` |
| `docs/static/images/paper/diagnostics_token_latency_meta_habitat_paper.png` | 44K (43829B) | `ed0fd7f4f987232b4b596101b356f7bdc06a039d9beaa904eef13d98f1d0efac` | converted from `BRACE-icml2026-camera-ready/fig/diagnostics_token_latency_meta_habitat.pdf` |
| `docs/static/images/paper/quantitative_summary_paper.png` | 319K (326692B) | `a5eeb15e1716e5679e64ab72c4bffef92ed3131c46b7cdf9a8c0aece8dceb470` | composed from `latency_cdf_meta_habitat_paper.png`, `slo_violation_bar_paper.png`, `diagnostics_coordination_wait_robofactory_paper.png`, and `diagnostics_token_latency_meta_habitat_paper.png` |

## How to update this file

When replacing any committed media:
1) Keep filenames stable when possible (avoid breaking `docs/index.html` references).
2) Update the `Size` + `sha256` fields above.
3) Fill `Source run/config` if you have the generating run id, config JSON, and any postprocess table path.
