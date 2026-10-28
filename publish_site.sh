#!/usr/bin/env bash
# Build the docs (Jupyter Book) and the 3D viewer, and publish both to GitHub Pages
# (gh-pages branch). The viewer is served at /viewer/ and reads its data from the bucket.
#
# If the git remote "viewer-site" exists, the viewer is also published on its own, at the
# root of its custom domain (VIEWER_DOMAIN), to the gh-pages branch of that repository.
#
#   ./publish_site.sh           build and publish
#   ./publish_site.sh --no-push build only, to check book/_build/html locally
set -euo pipefail
cd "$(dirname "$0")"

VIEWER_DATA_URL="https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/web"
VIEWER_DOMAIN="3d-antadatabase.com"

export JUPYTER_BOOK_BUILD=True
jupyter-book build book/

(
	cd web
	npm ci
	VITE_DATA_URL="$VIEWER_DATA_URL" npm run build -- --base=/anta_database/viewer/
	VITE_DATA_URL="$VIEWER_DATA_URL" npx vite build --base=/ --outDir dist-site --emptyOutDir
	# The local data (web/public/data) is copied by the build but read from the bucket online
	rm -rf dist/data dist-site/data
	# Custom domain of the standalone viewer, and no Jekyll processing on GitHub Pages
	echo "$VIEWER_DOMAIN" > dist-site/CNAME
	touch dist-site/.nojekyll
)
rm -rf book/_build/html/viewer
cp -r web/dist book/_build/html/viewer

if [[ "${1:-}" != "--no-push" ]]; then
	ghp-import -n -p -f book/_build/html/

	if git remote get-url viewer-site >/dev/null 2>&1; then
		# A fresh one-commit branch each time, like ghp-import, without touching this repository
		url="$(git remote get-url viewer-site)"
		site="$(mktemp -d)"
		cp -r web/dist-site/. "$site"
		git -C "$site" init -q -b gh-pages
		git -C "$site" add -A
		git -C "$site" commit -q -m "Publish the 3D viewer"
		git -C "$site" push -q -f "$url" gh-pages
		rm -rf "$site"
		echo "Viewer published to $url (https://$VIEWER_DOMAIN)"
	fi
fi
