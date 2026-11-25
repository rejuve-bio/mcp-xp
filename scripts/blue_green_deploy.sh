#!/usr/bin/env bash
set -euo pipefail

ACTION="${ACTION:-prepare}"
NAMESPACE="${NAMESPACE:?NAMESPACE is required}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-mcp-app}"
CONTAINER_NAME="${CONTAINER_NAME:-mcp-app}"
LIVE_SERVICE="${LIVE_SERVICE:-mcp-app}"
PREVIEW_SERVICE="${PREVIEW_SERVICE:-mcp-app-preview}"
CONTAINER_PORT="${CONTAINER_PORT:-8895}"
HEALTH_PATH="${HEALTH_PATH:-/}"
LOCAL_PORT="${LOCAL_PORT:-18895}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDERER="${SCRIPT_DIR}/blue_green_resource.py"

for command in kubectl python3 curl; do
  command -v "$command" >/dev/null || {
    echo "Required command is unavailable: $command" >&2
    exit 1
  }
done

deployment_exists() {
  kubectl get deployment "$1" -n "$NAMESPACE" >/dev/null 2>&1
}

active_slot() {
  kubectl get service "$LIVE_SERVICE" -n "$NAMESPACE" \
    -o jsonpath='{.spec.selector.app\.kubernetes\.io/slot}' 2>/dev/null || true
}

other_slot() {
  if [[ "$1" == "blue" ]]; then
    echo "green"
  else
    echo "blue"
  fi
}

wait_for_deployment() {
  kubectl rollout status "deployment/$1" -n "$NAMESPACE" --timeout=180s
}

ensure_preview_service() {
  local slot="$1"
  if ! kubectl get service "$PREVIEW_SERVICE" -n "$NAMESPACE" >/dev/null 2>&1; then
    kubectl get service "$LIVE_SERVICE" -n "$NAMESPACE" -o json \
      | python3 "$RENDERER" service --name "$PREVIEW_SERVICE" --slot "$slot" \
      | kubectl apply -f -
  fi
}

patch_service_slot() {
  local service="$1"
  local slot="$2"
  kubectl patch service "$service" -n "$NAMESPACE" --type merge \
    -p "{\"spec\":{\"selector\":{\"app.kubernetes.io/slot\":\"${slot}\"}}}"
}

smoke_test() {
  local deployment="$1"
  local log_file="${RUNNER_TEMP:-/tmp}/mcp-xp-port-forward-${deployment}.log"
  local port_forward_pid

  kubectl port-forward "deployment/$deployment" \
    "${LOCAL_PORT}:${CONTAINER_PORT}" -n "$NAMESPACE" >"$log_file" 2>&1 &
  port_forward_pid=$!
  trap 'kill "${port_forward_pid}" >/dev/null 2>&1 || true' RETURN

  for _ in {1..30}; do
    if curl --fail --silent --show-error --max-time 5 \
      "http://127.0.0.1:${LOCAL_PORT}${HEALTH_PATH}" >/dev/null; then
      kill "$port_forward_pid" >/dev/null 2>&1 || true
      wait "$port_forward_pid" 2>/dev/null || true
      trap - RETURN
      echo "Smoke test passed for deployment/$deployment"
      return 0
    fi
    sleep 2
  done

  echo "Smoke test failed for deployment/$deployment" >&2
  sed -n '1,120p' "$log_file" >&2 || true
  return 1
}

bootstrap_active_color() {
  local current_image
  local blue_deployment="${DEPLOYMENT_NAME}-blue"

  kubectl get service "$LIVE_SERVICE" -n "$NAMESPACE" >/dev/null

  if ! deployment_exists "$blue_deployment"; then
    if ! deployment_exists "$DEPLOYMENT_NAME"; then
      echo "Neither deployment/$blue_deployment nor legacy deployment/$DEPLOYMENT_NAME exists" >&2
      exit 1
    fi
    current_image="$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" \
      -o jsonpath="{.spec.template.spec.containers[?(@.name=='${CONTAINER_NAME}')].image}")"
    if [[ -z "$current_image" ]]; then
      echo "Could not read the current image from deployment/$DEPLOYMENT_NAME" >&2
      exit 1
    fi
    kubectl get deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" -o json \
      | python3 "$RENDERER" deployment --name "$blue_deployment" \
          --slot blue --image "$current_image" --container "$CONTAINER_NAME" \
      | kubectl apply -f -
  fi

  wait_for_deployment "$blue_deployment"
  ensure_preview_service blue
  patch_service_slot "$LIVE_SERVICE" blue
  echo "Migrated live traffic to deployment/$blue_deployment"
}

case "$ACTION" in
  prepare)
    IMAGE="${IMAGE:?IMAGE is required for ACTION=prepare}"
    current_slot="$(active_slot)"
    if [[ "$current_slot" != "blue" && "$current_slot" != "green" ]]; then
      bootstrap_active_color
      current_slot="blue"
    fi

    target_slot="$(other_slot "$current_slot")"
    source_deployment="${DEPLOYMENT_NAME}-${current_slot}"
    target_deployment="${DEPLOYMENT_NAME}-${target_slot}"
    deployment_exists "$source_deployment" || {
      echo "Active deployment/$source_deployment does not exist" >&2
      exit 1
    }

    kubectl get deployment "$source_deployment" -n "$NAMESPACE" -o json \
      | python3 "$RENDERER" deployment --name "$target_deployment" \
          --slot "$target_slot" --image "$IMAGE" --container "$CONTAINER_NAME" \
      | kubectl apply -f -

    wait_for_deployment "$target_deployment"
    ensure_preview_service "$target_slot"
    patch_service_slot "$PREVIEW_SERVICE" "$target_slot"
    smoke_test "$target_deployment"

    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
      {
        echo "previous-slot=$current_slot"
        echo "target-slot=$target_slot"
        echo "target-deployment=$target_deployment"
      } >>"$GITHUB_OUTPUT"
    fi
    echo "Candidate deployment/$target_deployment is healthy and ready for promotion"
    ;;

  switch)
    TARGET_SLOT="${TARGET_SLOT:?TARGET_SLOT is required for ACTION=switch}"
    if [[ "$TARGET_SLOT" != "blue" && "$TARGET_SLOT" != "green" ]]; then
      echo "TARGET_SLOT must be blue or green" >&2
      exit 1
    fi
    target_deployment="${DEPLOYMENT_NAME}-${TARGET_SLOT}"
    deployment_exists "$target_deployment" || {
      echo "Target deployment/$target_deployment does not exist" >&2
      exit 1
    }
    wait_for_deployment "$target_deployment"
    smoke_test "$target_deployment"
    patch_service_slot "$LIVE_SERVICE" "$TARGET_SLOT"
    echo "Live traffic now targets deployment/$target_deployment"
    ;;

  rollback)
    current_slot="$(active_slot)"
    if [[ "$current_slot" != "blue" && "$current_slot" != "green" ]]; then
      echo "The live service does not have a valid blue/green slot selector" >&2
      exit 1
    fi
    target_slot="$(other_slot "$current_slot")"
    target_deployment="${DEPLOYMENT_NAME}-${target_slot}"
    deployment_exists "$target_deployment" || {
      echo "Rollback deployment/$target_deployment does not exist" >&2
      exit 1
    }
    wait_for_deployment "$target_deployment"
    ensure_preview_service "$target_slot"
    patch_service_slot "$PREVIEW_SERVICE" "$target_slot"
    smoke_test "$target_deployment"
    patch_service_slot "$LIVE_SERVICE" "$target_slot"
    echo "Rolled live traffic back from $current_slot to $target_slot"
    ;;

  *)
    echo "ACTION must be prepare, switch, or rollback" >&2
    exit 1
    ;;
esac
