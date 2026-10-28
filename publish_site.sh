#!/usr/bin/env bash
# Build the docs (Jupyter Book) and the 3D viewer, and publish both to GitHub Pages
# (gh-pages branch). The viewer is served at /viewer/ and reads its data from the bucket.
#
#   ./publish_site.sh           build and publish
#   ./publish_site.sh --no-push build only, to check book/_build/html locally
set -euo pipefail
cd "$(dirname "$0")"

VIEWER_DATA_URL="https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/web"

jupyter-book build book/

(
  cd web
  npm ci
  VITE_DATA_URL="$VIEWER_DATA_URL" npm run build -- --base=/anta_database/viewer/
  # The local data (web/public/data) is copied by the build but read from the bucket online
  rm -rf dist/data
)
rm -rf book/_build/html/viewer
cp -r web/dist book/_build/html/viewer

if [[ "${1:-}" != "--no-push" ]]; then
  ghp-import -n -p -f book/_build/html/
fi
