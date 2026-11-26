#!/usr/bin/env bash
# Deploy the portal on its server (see README.md): copy the code, the anta_database package and
# the settings, then (re)build and start the containers.
#
#   portal/deploy.sh [user@host]      (default: ubuntu@<PORTAL_DOMAIN>, i.e. ubuntu@api.antadatabase.com)
#
# The settings on the server (/opt/portal/portal.env) are made of portal/.env (email) and
# portal/.env.server (keys of the buckets), not in git. Another copy of the portal (e.g. the test
# one, README.md "A test copy") is deployed with its own settings files:
#
#   PORTAL_ENV_FILE=.env.test PORTAL_SERVER_ENV_FILE=.env.server.test portal/deploy.sh
#
# The names below (domain, buckets, origins of the site) have the values of the official portal;
# another copy must set them all in its own files. PORTAL_DOMAIN goes to Caddy (caddy.env), for
# its HTTPS certificate.
#
# The worker also needs usurf and thk of Bedmap3 (BEDMAP_SURFACE, see README.md), copied once into
# /opt/portal/data.
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE="${PORTAL_ENV_FILE:-.env}"
SERVER_ENV_FILE="${PORTAL_SERVER_ENV_FILE:-.env.server}"
DIR=/opt/portal
BEDMAP_SURFACE="${BEDMAP_SURFACE:-$HOME/data/bedmap/bedmap3_surface.nc}"
DEFAULTS=(
	PORTAL_DOMAIN=api.antadatabase.com
	PORTAL_SUBMISSIONS_BUCKET=antadatabase-submissions
	PORTAL_CONTACT_BUCKET=antadatabase-contact
	PORTAL_COMPILED_BUCKET=antadatabase-compiled
	PORTAL_REVIEW_BUCKET=antadatabase-review
	PORTAL_ORIGINS=https://antadatabase.com
)

for f in "$ENV_FILE" "$SERVER_ENV_FILE"; do
	[ -f "$f" ] || { echo "No settings file portal/$f" >&2; exit 1; }
done
# Another copy of the portal sets all the names itself: none of the official ones by default
if [ "$ENV_FILE" != .env ] || [ "$SERVER_ENV_FILE" != .env.server ]; then
	for pair in "${DEFAULTS[@]}"; do
		grep -q "^${pair%%=*}=" "$ENV_FILE" "$SERVER_ENV_FILE" ||
			{ echo "${pair%%=*} is not set in portal/$ENV_FILE or portal/$SERVER_ENV_FILE" >&2; exit 1; }
	done
fi
# A setting of the settings files (the last one wins), or its default
setting() { # <name> <default>
	local value
	value="$(grep -h "^$1=" "$ENV_FILE" "$SERVER_ENV_FILE" | tail -n 1 | cut -d= -f2-)" || true
	echo "${value:-$2}"
}
DOMAIN="$(setting PORTAL_DOMAIN api.antadatabase.com)"
HOST="${1:-ubuntu@$DOMAIN}"
echo "Deploying to $HOST, domain $DOMAIN, settings portal/$ENV_FILE and portal/$SERVER_ENV_FILE"

rsync -az --delete --exclude .venv --exclude data --exclude dev --exclude package --exclude '.env*' \
	--exclude portal.env --exclude caddy.env --exclude __pycache__ ./ "$HOST:$DIR/"
ssh "$HOST" "mkdir -p $DIR/package"
rsync -az --delete --exclude __pycache__ --exclude 'test_*.py' ../anta_database/ "$HOST:$DIR/package/anta_database/"
ssh "$HOST" "mkdir -p $DIR/data"
if [ -f "$BEDMAP_SURFACE" ]; then
	rsync -az "$BEDMAP_SURFACE" "$HOST:$DIR/data/bedmap3_surface.nc"
else
	echo "No $BEDMAP_SURFACE: the previews and checks will go without Bedmap3 (see README.md)" >&2
fi
{
	cat "$ENV_FILE"
	echo
	cat "$SERVER_ENV_FILE"
	echo
	for pair in "${DEFAULTS[@]}"; do
		grep -q "^${pair%%=*}=" "$ENV_FILE" "$SERVER_ENV_FILE" || echo "$pair"
	done
} | ssh "$HOST" "umask 077 && cat > $DIR/portal.env"
echo "PORTAL_DOMAIN=$DOMAIN" | ssh "$HOST" "cat > $DIR/caddy.env"
# The worker runs the same image as the portal: built once (two builds at once collide on its name)
ssh "$HOST" "cd $DIR && docker compose build --pull portal && docker compose up -d --remove-orphans && docker compose ps"
