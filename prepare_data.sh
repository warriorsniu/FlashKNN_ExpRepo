#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=()
if [[ "${ACCEPT_S3DIS_LICENSE:-0}" == "1" ]]; then
  ARGS+=(--accept-s3dis-license)
fi

# A separately transferred licensed pack can simply be placed here.  This is
# the zero-configuration path used for the H20 bundle.
INCOMING_PACK="$REPO_DIR/data/incoming/semantickitti_pack"
if [[ -d "$INCOMING_PACK" ]]; then
  ARGS+=(--semantickitti-pack "$INCOMING_PACK")
elif [[ -n "${SEMANTICKITTI_PACK:-}" ]]; then
  ARGS+=(--semantickitti-pack "$SEMANTICKITTI_PACK")
elif [[ -n "${SEMANTICKITTI_PACK_URL:-}" ]]; then
  python "$REPO_DIR/scripts/download_semantickitti.py" \
    --pack-url "$SEMANTICKITTI_PACK_URL" \
    --work-dir "$REPO_DIR/data/downloads/semantickitti" \
    --output-pack "$REPO_DIR/data/downloads/semantickitti/pack"
  DOWNLOADED_PACK="$(cat "$REPO_DIR/data/downloads/semantickitti/resolved_pack.txt")"
  ARGS+=(--semantickitti-pack "$DOWNLOADED_PACK")
elif [[ -n "${KITTI_VELODYNE_URL:-}" ]]; then
  python "$REPO_DIR/scripts/download_semantickitti.py" \
    --kitti-velodyne-url "$KITTI_VELODYNE_URL" \
    --work-dir "$REPO_DIR/data/downloads/semantickitti" \
    --output-pack "$REPO_DIR/data/downloads/semantickitti/pack"
  DOWNLOADED_PACK="$(cat "$REPO_DIR/data/downloads/semantickitti/resolved_pack.txt")"
  ARGS+=(--semantickitti-pack "$DOWNLOADED_PACK")
elif [[ -n "${SEMANTICKITTI_ROOT:-}" ]]; then
  ARGS+=(--semantickitti-root "$SEMANTICKITTI_ROOT")
else
  echo "SemanticKITTI input was not found." >&2
  echo "Place the supplied pack at data/incoming/semantickitti_pack," >&2
  echo "or set SEMANTICKITTI_PACK_URL, KITTI_VELODYNE_URL," >&2
  echo "SEMANTICKITTI_PACK, or SEMANTICKITTI_ROOT." >&2
  exit 2
fi

if [[ -n "${S3DIS_ROOT:-}" ]]; then
  ARGS+=(--s3dis-existing "$S3DIS_ROOT")
elif [[ -d "$REPO_DIR/data/s3dis" ]]; then
  ARGS+=(--s3dis-existing "$REPO_DIR/data/s3dis")
elif [[ "${ACCEPT_S3DIS_LICENSE:-0}" != "1" ]]; then
  echo "S3DIS requires license acceptance." >&2
  echo "After reading https://cvg-data.inf.ethz.ch/s3dis/, rerun as:" >&2
  echo "  ACCEPT_S3DIS_LICENSE=1 bash prepare_data.sh" >&2
  exit 2
fi

python "$REPO_DIR/scripts/prepare_data.py" "${ARGS[@]}"
bash "$REPO_DIR/scripts/preflight.sh" --data-only
