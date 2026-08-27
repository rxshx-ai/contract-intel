#!/usr/bin/env bash
# A project-local Postgres + pgvector. Data never leaves this machine, and
# nothing touches whatever Postgres you already run.
#
#   ./deploy/local-db.sh start     initialise if needed, then start
#   ./deploy/local-db.sh stop
#   ./deploy/local-db.sh status
#   ./deploy/local-db.sh psql
#   ./deploy/local-db.sh reset     destroy the data and start clean
#   ./deploy/local-db.sh url       print the DATABASE_URL
#
# The cluster lives in .pgdata/ inside the project and listens on 55432, so it
# cannot collide with a system Postgres on 5432. Delete the folder and it is
# gone -- no services registered, nothing left behind.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGDATA="$ROOT/.pgdata"
PORT="${PGPORT:-55432}"
DB=contract_intel
# The socket lives OUTSIDE the data directory: initdb refuses to
# initialise into a folder that already contains anything.
SOCKET="$ROOT/.pgsock"

# pgvector is built against a specific major version; prefer one that has it.
find_bin() {
  for v in 18 17 16; do
    if [ -x "/opt/homebrew/opt/postgresql@$v/bin/pg_ctl" ] \
       && [ -f "/opt/homebrew/lib/postgresql@$v/vector.dylib" ]; then
      echo "/opt/homebrew/opt/postgresql@$v/bin"; return
    fi
  done
  command -v pg_ctl >/dev/null && dirname "$(command -v pg_ctl)" && return
  echo ""
}

BIN="$(find_bin)"
[ -n "$BIN" ] || {
  echo "No Postgres with pgvector found. Install:"
  echo "  brew install postgresql@17 pgvector"; exit 1; }

url() { echo "postgresql://postgres@/$DB?host=$SOCKET&port=$PORT"; }

start() {
  if [ ! -d "$PGDATA/base" ]; then
    echo "==> Initialising cluster in .pgdata (first run)"
    rm -rf "$PGDATA"
    "$BIN/initdb" -D "$PGDATA" -U postgres --auth=trust >/dev/null
  fi
  mkdir -p "$SOCKET"
  if "$BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    echo "==> Already running on port $PORT"
  else
    echo "==> Starting on port $PORT"
    "$BIN/pg_ctl" -D "$PGDATA" -o "-p $PORT -k $SOCKET -c listen_addresses=''" \
      -l "$PGDATA/server.log" start >/dev/null
    sleep 2
  fi
  "$BIN/psql" -h "$SOCKET" -p "$PORT" -U postgres -tc \
    "SELECT 1 FROM pg_database WHERE datname='$DB'" | grep -q 1 \
    || "$BIN/psql" -h "$SOCKET" -p "$PORT" -U postgres -c "CREATE DATABASE $DB" >/dev/null
  "$BIN/psql" -h "$SOCKET" -p "$PORT" -U postgres -d "$DB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null
  echo "==> Ready. pgvector $("$BIN/psql" -h "$SOCKET" -p "$PORT" -U postgres -d "$DB" \
    -tAc "SELECT extversion FROM pg_extension WHERE extname='vector'")"
  echo
  echo "  export DATABASE_URL=\"$(url)\""
}

case "${1:-start}" in
  start) start ;;
  stop) "$BIN/pg_ctl" -D "$PGDATA" stop >/dev/null 2>&1 && echo "stopped" || echo "not running" ;;
  status) "$BIN/pg_ctl" -D "$PGDATA" status || true ;;
  psql) shift; "$BIN/psql" -h "$SOCKET" -p "$PORT" -U postgres -d "$DB" "$@" ;;
  url) url ;;
  reset)
    "$BIN/pg_ctl" -D "$PGDATA" stop >/dev/null 2>&1 || true
    rm -rf "$PGDATA"; echo "destroyed .pgdata"; start ;;
  *) echo "usage: $0 {start|stop|status|psql|url|reset}"; exit 1 ;;
esac
