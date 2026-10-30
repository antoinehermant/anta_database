#!/usr/bin/env bash
# Build and publish the AntADatabase websites, all on GitHub Pages:
#
#   https://antadatabase.com        front page, 3D viewer (/viewer/), download (/download/) and
#                                   submit (/submit/): web/, to the gh-pages branch of the
#                                   repository of the git remote "site"
#   https://docs.antadatabase.com   documentation (Jupyter Book): book/, to the gh-pages branch of
#                                   this repository. Its old copy of the viewer (/viewer/) now
#                                   redirects to the new site.
#   https://3d-antadatabase.com     old address of the viewer: redirects to the new site, from the
#                                   gh-pages branch of the repository of the git remote "viewer-site"
#   https://api.antadatabase.com    submission portal (portal/, on its own server: portal/deploy.sh),
#                                   used by /contact/ and the development version of /submit/
#                                   (/submit/?dev) once it resolves. /submit/ itself opens with
#                                   PORTAL_PUBLIC=true
#
# The viewer and the download page read their data from the bucket.
#
# Nothing is pushed before antadatabase.com resolves (DNS set up, see web/README.md): the old
# addresses would otherwise redirect to nowhere. The docs only get their custom domain once
# docs.antadatabase.com resolves too (until then they stay on github.io).
#
#   ./publish_site.sh           build and publish
#   ./publish_site.sh --no-push build only, to check book/_build/html, web/dist and web/dist-redirect
set -euo pipefail
cd "$(dirname "$0")"

VIEWER_DATA_URL="https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/web"
SITE_DOMAIN="antadatabase.com"
DOCS_DOMAIN="docs.antadatabase.com"
OLD_VIEWER_DOMAIN="3d-antadatabase.com"
PORTAL_URL="https://api.antadatabase.com"

push=true
[[ "${1:-}" == "--no-push" ]] && push=false

# A name in the DNS: locally, or at a public resolver (a router may remember for a while that a
# new name did not exist)
resolves() { getent hosts "$1" >/dev/null || [[ -n "$(dig +short "$1" @1.1.1.1 2>/dev/null)" ]]; }

if $push; then
	if ! resolves "$SITE_DOMAIN"; then
		echo "$SITE_DOMAIN does not resolve yet: set up its DNS first (see web/README.md)," >&2
		echo "or run with --no-push to only build." >&2
		exit 1
	fi
	if ! git remote get-url site >/dev/null 2>&1; then
		echo "No git remote \"site\" (the GitHub repository of $SITE_DOMAIN, see web/README.md)." >&2
		exit 1
	fi
fi

# A redirect page (web/redirect.html) to the new site, for the old addresses of the viewer
redirect_page() { # <output file> <path of the old viewer to strip>
	sed -e "s#__SITE__#https://$SITE_DOMAIN#g" -e "s#__STRIP__#$2#g" web/redirect.html >"$1"
}

# Publish a directory as a fresh one-commit gh-pages branch of a remote, like ghp-import,
# without touching this repository
push_pages() { # <directory> <remote> <message>
	local url site
	url="$(git remote get-url "$2")"
	site="$(mktemp -d)"
	cp -r "$1"/. "$site"
	git -C "$site" init -q -b gh-pages
	git -C "$site" add -A
	git -C "$site" commit -q -m "$3"
	git -C "$site" push -q -f "$url" gh-pages
	rm -rf "$site"
	echo "Published $1 to $url"
}

# Documentation
export JUPYTER_BOOK_BUILD=True
jupyter-book build book/
html=book/_build/html
rm -rf "$html/viewer"
mkdir -p "$html/viewer/download"
redirect_page "$html/viewer/index.html" /viewer
redirect_page "$html/viewer/download/index.html" /viewer
touch "$html/.nojekyll"
rm -f "$html/CNAME"
if resolves "$DOCS_DOMAIN"; then
	echo "$DOCS_DOMAIN" >"$html/CNAME"
else
	echo "Note: $DOCS_DOMAIN does not resolve yet, the docs stay on github.io." >&2
fi

# Website: the forms use the portal once its address resolves (otherwise the contact page writes
# an email and the submit page stays closed)
portal_url=""
if resolves "${PORTAL_URL#https://}"; then
	portal_url="$PORTAL_URL"
else
	echo "Note: ${PORTAL_URL#https://} does not resolve yet, the forms do not use the portal." >&2
fi
(
	cd web
	npm ci
	VITE_DATA_URL="$VIEWER_DATA_URL" VITE_PORTAL_URL="$portal_url" VITE_PORTAL_PUBLIC="${PORTAL_PUBLIC:-false}" npm run build
	# The local data (web/public/data) is copied by the build but read from the bucket online
	rm -rf dist/data
	# Custom domain, and no Jekyll processing on GitHub Pages
	echo "$SITE_DOMAIN" >dist/CNAME
	touch dist/.nojekyll
	# Front page of the site repository on GitHub (shows og-image.jpg)
	cp site-README.md dist/README.md

	# Old address of the viewer: every path (index.html and 404.html) redirects
	rm -rf dist-redirect dist-site
	mkdir dist-redirect
	cd ..
	redirect_page web/dist-redirect/index.html ""
	cp web/dist-redirect/index.html web/dist-redirect/404.html
	echo "$OLD_VIEWER_DOMAIN" >web/dist-redirect/CNAME
	touch web/dist-redirect/.nojekyll
	printf '# %s\n\nThe AntADatabase 3D viewer has moved to https://%s/viewer/. This repository only\nredirects the old address there.\n' \
		"$OLD_VIEWER_DOMAIN" "$SITE_DOMAIN" >web/dist-redirect/README.md
)

if $push; then
	ghp-import -n -p -f "$html"
	push_pages web/dist site "Publish antadatabase.com"
	if git remote get-url viewer-site >/dev/null 2>&1; then
		push_pages web/dist-redirect viewer-site "Redirect to $SITE_DOMAIN"
	fi
fi
