#!/usr/bin/env bash

set -euo pipefail

# This script wraps the official CVAT Docker Compose workflow and injects
# a project-specific shared directory mount. The shared directory lets CVAT
# read the existing dataset tree directly, so annotation tasks can import
# `durian/images/train` from local storage instead of re-uploading files.

ROOT_DIR="/Users/liujiaqi/code/JS/durian-helper-mini-program"
PY_CV_DIR="${ROOT_DIR}/server/python-cv-service"
CVAT_DIR="${PY_CV_DIR}/.tools/cvat"
DATASETS_DIR="${PY_CV_DIR}/datasets"
OVERRIDE_FILE="${CVAT_DIR}/docker-compose.override.yml"
CVAT_REPO_URL="https://github.com/cvat-ai/cvat.git"

print_usage() {
  cat <<'EOF'
Usage:
  bash server/python-cv-service/scripts/cvat_local.sh prepare
  bash server/python-cv-service/scripts/cvat_local.sh start
  bash server/python-cv-service/scripts/cvat_local.sh stop
  bash server/python-cv-service/scripts/cvat_local.sh status

Commands:
  prepare  Clone CVAT if needed and write docker-compose.override.yml
  start    Start CVAT with the local dataset share mounted in /home/django/share
  stop     Stop CVAT containers
  status   Show current container status
EOF
}

ensure_prerequisites() {
  command -v git >/dev/null 2>&1 || {
    echo "git is required but not found." >&2
    exit 1
  }

  command -v docker >/dev/null 2>&1 || {
    echo "docker is required but not found." >&2
    exit 1
  }

  docker compose version >/dev/null 2>&1 || {
    echo "docker compose is required but not found." >&2
    exit 1
  }
}

clone_cvat_if_missing() {
  if [[ -d "${CVAT_DIR}/.git" ]]; then
    echo "CVAT repo already exists: ${CVAT_DIR}"
    return
  fi

  mkdir -p "${PY_CV_DIR}/.tools"
  git clone --depth 1 "${CVAT_REPO_URL}" "${CVAT_DIR}"
}

write_override_file() {
  mkdir -p "${CVAT_DIR}"

  # The override mounts the project dataset root as CVAT's shared storage.
  # CVAT scans `/home/django/share`, so mounting `datasets/` there keeps the
  # task source aligned with the training folder layout already used here.
  cat > "${OVERRIDE_FILE}" <<EOF
services:
  cvat_server:
    volumes:
      - ${DATASETS_DIR}:/home/django/share:ro
  cvat_worker_import:
    volumes:
      - ${DATASETS_DIR}:/home/django/share:ro
  cvat_worker_export:
    volumes:
      - ${DATASETS_DIR}:/home/django/share:ro
  cvat_worker_annotation:
    volumes:
      - ${DATASETS_DIR}:/home/django/share:ro
EOF

  echo "Wrote ${OVERRIDE_FILE}"
}

prepare() {
  ensure_prerequisites
  clone_cvat_if_missing
  write_override_file
}

start() {
  prepare
  (
    cd "${CVAT_DIR}"
    docker compose up -d
  )
}

stop() {
  if [[ ! -d "${CVAT_DIR}" ]]; then
    echo "CVAT repo not found: ${CVAT_DIR}" >&2
    exit 1
  fi

  (
    cd "${CVAT_DIR}"
    docker compose down
  )
}

status() {
  if [[ ! -d "${CVAT_DIR}" ]]; then
    echo "CVAT repo not found: ${CVAT_DIR}" >&2
    exit 1
  fi

  (
    cd "${CVAT_DIR}"
    docker compose ps
  )
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    prepare)
      prepare
      ;;
    start)
      start
      ;;
    stop)
      stop
      ;;
    status)
      status
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
