#!/usr/bin/env bash
# Deploy the portal on its server (see README.md): copy the code, the anta_database package and
# the settings, then (re)build and start the containers.
#
#   portal/deploy.sh [user@host]      (default: ubuntu@api.antadatabase.com)
#
# The settings on the server (/opt/portal/portal.env) are made of portal/.env (email) and
# portal/.env.server (keys of the buckets), not in git. The worker also needs usurf and thk of
# Bedmap3 (BEDMAP_SURFACE, see README.md), copied once into /opt/portal/data.
set -euo pipefail
cd "$(dirname "$0")"
HOST="${1:-ubuntu@api.antadatabase.com}"
DIR=/opt/portal
BEDMAP_SURFACE="${BEDMAP_SURFACE:-$HOME/data/bedmap/bedmap3_surface.nc}"

rsync -az --delete --exclude .venv --exclude data --exclude dev --exclude package --exclude '.env*' --exclude portal.env \
	--exclude __pycache__ ./ "$HOST:$DIR/"
ssh "$HOST" "mkdir -p $DIR/package"
rsync -az --delete --exclude __pycache__ --exclude 'test_*.py' ../anta_database/ "$HOST:$DIR/package/anta_database/"
ssh "$HOST" "mkdir -p $DIR/data"
if [ -f "$BEDMAP_SURFACE" ]; then
	rsync -az "$BEDMAP_SURFACE" "$HOST:$DIR/data/bedmap3_surface.nc"
else
	echo "No $BEDMAP_SURFACE: the previews and checks will go without Bedmap3 (see README.md)" >&2
fi
{
	cat .env
	echo
	cat .env.server
	echo "PORTAL_SUBMISSIONS_BUCKET=antadatabase-submissions"
	echo "PORTAL_CONTACT_BUCKET=antadatabase-contact"
	echo "PORTAL_COMPILED_BUCKET=antadatabase-compiled"
	echo "PORTAL_REVIEW_BUCKET=antadatabase-review"
	echo "PORTAL_ORIGINS=https://antadatabase.com"
} | ssh "$HOST" "umask 077 && cat > $DIR/portal.env"
ssh "$HOST" "cd $DIR && docker compose up -d --build --remove-orphans && docker compose ps"
