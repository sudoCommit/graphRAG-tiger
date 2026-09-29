#!/usr/bin/env bash

set -u

cd /home/sidharth/imp_repos/graphRAG-tiger/pre_defined/graphrag_deploy

SERVICE="graphrag-ecc"
STATUS_URL="http://localhost:8001/WikipediaGraph/graphrag/rebuild_status"
STOP_FILE="$(mktemp)"

: "${TG_API_KEY:?Export TG_API_KEY before running this script}"

cleanup() {
  rm -f "$STOP_FILE"
}
trap cleanup EXIT

# Watch new ECC logs for rate limiting.
(
  while IFS= read -r line; do
    if [[ "$line" =~ HTTP/[0-9.]+[[:space:]]+429([[:space:]]|$) ]] ||
    [[ "$line" =~ (status_code|status)[=:][[:space:]]*429([^0-9]|$) ]] ||
    [[ "$line" =~ RateLimitError|rate_limit_exceeded|Too[[:space:]]Many[[:space:]]Requests ]]; then
      echo "429 rate limit detected. Killing $SERVICE..."
      docker compose kill "$SERVICE"
      touch "$STOP_FILE"
      exit 0
    fi
  done < <(docker compose logs --since=1s -f --no-color "$SERVICE")
) &

while true; do
  if [[ -f "$STOP_FILE" ]]; then
    echo "Refresh stopped because of a 429 rate limit."
    exit 1
  fi

  body="$(curl -fsS --max-time 10 \
    -H "Authorization: Bearer $TG_API_KEY" \
    "$STATUS_URL" 2>/dev/null || true)"

  if [[ -n "$body" ]]; then
    running="$(jq -r '.is_running // false' <<< "$body")"
    current="$(jq -r '.progress_current // empty' <<< "$body")"
    total="$(jq -r '.progress_total // empty' <<< "$body")"
    stage="$(jq -r '.stage // empty' <<< "$body")"

    if [[ "$running" == "true" && -n "$current" && -n "$total" ]]; then
      echo "Refresh is currently still running at $current/$total documents. Stage: $stage"
    else
      echo "Refresh status: $stage"
    fi
  else
    echo "Unable to query refresh status."
  fi

  sleep 30
done
