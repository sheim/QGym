# Report move manifest

Pre-move hashes were recorded before any file was moved from `/tmp/opencode`.
All destinations are relative to this directory. Every file listed below was
moved (not copied) to this docs tree, and a compatibility symlink now points
from the original report path to the durable destination.

| Original report | Original sha256 | Bytes | Durable destination | Current sha256 | Bytes | Intentional changes |
|---|---|---:|---|---|---:|---|
| `/tmp/opencode/height-q-fixed-variance-20261110-v1/REPORT.md` | `86815efa98c48a18ca5e816eddf77c2ca9909991173c2a6ae9005e8ce9ec3e2b` | 6630 | `fixed-variance-20261110-v1/REPORT.md` | `a61547f17538cd968136643fd2363c37a4541899a07d64021c154172c9e2c400` | 7240 | Provenance note |
| `/tmp/opencode/height-q-fixed-variance-64-20261110-v1/REPORT.md` | `dca51a58671226f19d220e889c1a9b0ba800490422e5b27695668ae29d7ba8d3` | 6003 | `fixed-variance-64-20261110-v1/REPORT.md` | `54e79099017751547412f6ff45ee3cbba59debce0e7ef11674b707052c506836` | 6960 | Provenance note; relative artifact links made absolute to `/tmp/opencode`; cross-report link retargeted; link lines reflowed |
| `/tmp/opencode/height-q-fixed-variance-16x64-20261110-v1/REPORT.md` | `44da6f7c6938a00c186f7307799bb99eb9531c2a42f40218ff0272ef47ffeaf9` | 7492 | `fixed-variance-16x64-20261110-v1/REPORT.md` | `8c766dd7ca73123c48c558b241a9d3cb0f0db24f88a4d4187e12563f1ae277c9` | 8126 | Provenance note |
| `/tmp/opencode/height-q-persistent-intervention-20261110-v1/REPORT.md` | `bf7501425e817f06f0ec05c8e2fa702c2c78df675c897c6296b8827916cb227c` | 13982 | `persistent-intervention-20261110-v1/REPORT.md` | `b74ae40cd2b3783a476d83c06325cb49fb38b237322dc124365b8cffe9b6794f` | 14628 | Provenance note |
| `/tmp/opencode/height-q-paired16-comparison-20261111-v1/REPORT.md` | `f6bc04e2a09e34196a0e27e3d7fd621ba2be11ff857e9cafda0f47917dcb9006` | 17377 | `paired16-comparison-20261111-v1/REPORT.md` | `aeee3d9c4fb8777723111b52fea85b8db80d5d69672c81bdb2aa93319b73f8c9` | 18007 | Provenance note |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/REPORT.md` | `8ddcecfec8175b45ce86ad6f453abf80f85ff945deefa855277047ceb4c121ec` | 16683 | `two-critic-datasets-20261111-v1/REPORT.md` | `d55b2f64db6ecf1674a1b0507fdb8cbda207071f024520453e7fe65c2b87dcc1` | 17313 | Provenance note |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/REPORT.md` | `e6dd2477db458bd0d3f72e376e6eced53c911f8cc59a3d32e540dafbaf0977c6` | 10612 | `two-critic-datasets-20261111-v1/analysis/REPORT.md` | `6732552e1c75e4153e4636340a17530183190a3454b0a4dec44e0e3a591dae87` | 11263 | Provenance note |
| `/tmp/opencode/height-q-two-critic-datasets-20261111-v1/analysis/return_stats.md` | `a63c07d99a62209722b0bd1c83ac1170715f7ae85c5e43e949f294cc0d0c8fff` | 2570 | `two-critic-datasets-20261111-v1/analysis/return_stats.md` | `d688aef07f9e3935074aed9c1faf40625cae7e36e505786be6ac6a0c9216f357` | 3233 | Provenance note |
| `/tmp/opencode/height-q-rejection-threshold-20261111-v1/REPORT.md` | `5cc8e9f024bda94b527e1d19c38e97b46cde548e6f5b8e290b188e3b54140d6d` | 6783 | `rejection-threshold-20261111-v1/REPORT.md` | `3a5c1409826b86094c32bbc0ec06975d796be1ff4dc720c04cdf14d5327022dd` | 8457 | Provenance note; consolidated threshold-attainment table added 2026-11-11 at user request |
| `/tmp/opencode/height-q-original-reward-tradeoff-20261111-v1/REPORT.md` | `01f64239ea4767a4cb42c2fcffe1ccfec599c23cfd33a8c58cd1ace1fd766712` | 9988 | `original-reward-tradeoff-20261111-v1/REPORT.md` | `b9cd723ca965fb55cc9ab5a7f4e4ad1926efa7adc07140cfe52d1d81b199574e` | 10622 | Provenance note |

## Preservation check

For every row, removing the added provenance note and reversing exactly the
listed link/table changes reconstructed bytes whose sha256 equaled the recorded
pre-move original hash. The rejection report's added table is explicitly marked
inside the report as not part of the original sha256. Raw data, checkpoints,
logs, CSVs, and all other experiment artifacts were not moved and remain in the
original `/tmp/opencode/height-q-*` directories.

`README.md`, this manifest, and the top-of-report provenance notes are new
documentation, not moved report content.
