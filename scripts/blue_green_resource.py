#!/usr/bin/env python3
"""Render a color-specific Kubernetes resource from a live resource."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from typing import Any


SLOT_LABEL = "app.kubernetes.io/slot"
NAME_LABEL = "app.kubernetes.io/name"
SERVER_METADATA = {
    "creationTimestamp",
    "generation",
    "managedFields",
    "resourceVersion",
    "selfLink",
    "uid",
    "ownerReferences",
}


def clean_metadata(metadata: dict[str, Any]) -> None:
    for key in SERVER_METADATA:
        metadata.pop(key, None)

    annotations = metadata.get("annotations", {})
    annotations.pop("deployment.kubernetes.io/revision", None)
    annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
    if not annotations:
        metadata.pop("annotations", None)


def slot_labels(labels: dict[str, str] | None, slot: str) -> dict[str, str]:
    result = dict(labels or {})
    result[NAME_LABEL] = "mcp-app"
    result[SLOT_LABEL] = slot
    return result


def render_deployment(
    source: dict[str, Any], name: str, slot: str, image: str, container_name: str
) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result.pop("status", None)
    clean_metadata(result["metadata"])
    result["metadata"]["name"] = name
    result["metadata"]["labels"] = slot_labels(
        result["metadata"].get("labels"), slot
    )

    spec = result["spec"]
    spec["selector"]["matchLabels"] = slot_labels(
        spec["selector"].get("matchLabels"), slot
    )
    template_metadata = spec["template"].setdefault("metadata", {})
    template_metadata["labels"] = slot_labels(template_metadata.get("labels"), slot)
    template_annotations = template_metadata.setdefault("annotations", {})
    template_annotations["mcp-xp.rejuve.bio/release"] = image
    template_annotations["mcp-xp.rejuve.bio/deployed-at"] = datetime.now(
        timezone.utc
    ).isoformat()

    containers = spec["template"]["spec"]["containers"]
    container = next(
        (item for item in containers if item.get("name") == container_name), None
    )
    if container is None:
        available = ", ".join(item.get("name", "<unnamed>") for item in containers)
        raise ValueError(
            f"container {container_name!r} was not found; available containers: {available}"
        )
    container["image"] = image
    container["imagePullPolicy"] = "Always"
    return result


def render_service(source: dict[str, Any], name: str, slot: str) -> dict[str, Any]:
    result = copy.deepcopy(source)
    result.pop("status", None)
    clean_metadata(result["metadata"])
    result["metadata"]["name"] = name
    result["metadata"]["labels"] = slot_labels(
        result["metadata"].get("labels"), slot
    )

    spec = result["spec"]
    for key in (
        "allocateLoadBalancerNodePorts",
        "clusterIP",
        "clusterIPs",
        "externalIPs",
        "externalTrafficPolicy",
        "healthCheckNodePort",
        "ipFamilies",
        "ipFamilyPolicy",
        "loadBalancerClass",
        "loadBalancerIP",
        "loadBalancerSourceRanges",
        "publishNotReadyAddresses",
    ):
        spec.pop(key, None)
    spec["type"] = "ClusterIP"
    spec["selector"] = slot_labels(spec.get("selector"), slot)
    for port in spec.get("ports", []):
        port.pop("nodePort", None)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="resource", required=True)

    deployment = subparsers.add_parser("deployment")
    deployment.add_argument("--name", required=True)
    deployment.add_argument("--slot", choices=("blue", "green"), required=True)
    deployment.add_argument("--image", required=True)
    deployment.add_argument("--container", default="mcp-app")

    service = subparsers.add_parser("service")
    service.add_argument("--name", required=True)
    service.add_argument("--slot", choices=("blue", "green"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.load(sys.stdin)
    if source.get("kind") != args.resource.title():
        raise ValueError(
            f"expected a {args.resource.title()}, got {source.get('kind', '<unknown>')}"
        )

    if args.resource == "deployment":
        result = render_deployment(
            source, args.name, args.slot, args.image, args.container
        )
    else:
        result = render_service(source, args.name, args.slot)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
