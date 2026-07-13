# Environment Audit

Audit date: 2026-07-14.

## WSL diagnosis

Ubuntu was not lost and its virtual disk was not unregistered. Codex ordinary shell commands run under
a restricted Windows sandbox SID, while `Ubuntu-22.04` is registered under the interactive Windows user
SID. WSL consults the current user's `HKCU` registry, so an ordinary sandbox command saw an empty distro
list even though the real user registration was healthy.

An approved command running under the interactive user sees `Ubuntu-22.04` normally. The existing
33.7 GB `ext4.vhdx`, default Linux user `beta`, `/home/beta/LNS2-RL`, Conda environments, and other home
data are intact. No `wsl --import-in-place`, reinstall, duplicate distro, or registry edit was performed.

## Existing dependencies

| Component | Existing version/status |
| --- | --- |
| Ubuntu | 22.04 on WSL2 |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| CMake | 3.22.1 |
| G++ | 11.4.0 |
| GNU Make | 4.3 |
| Git | 2.34.1 |
| Python | 3.10.12 |
| Boost development packages | 1.74 |
| Eigen development package | 3.4.0 |
| Python development headers | installed |
| pybind11 development headers | 2.9.1 |

Ninja and system `pip3` were absent, but neither is required by the selected Make/CMake and system
pybind11 workflow. No packages were installed or upgraded.

The existing `/home/beta/LNS2-RL` checkout has user modifications and generated model/build files. It
was inspected read-only and was not reused as the official baseline.
