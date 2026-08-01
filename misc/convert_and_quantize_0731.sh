#!/bin/bash
# Download and setup DeepSeek-V4-Flash-0731 (July 31 Official Release) for Strix Halo.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${ROOT}/misc/download_0731_aria2.sh"
