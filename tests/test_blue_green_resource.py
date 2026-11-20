import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "blue_green_resource.py"
SPEC = importlib.util.spec_from_file_location("blue_green_resource", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
render_deployment = MODULE.render_deployment
render_service = MODULE.render_service


def test_render_deployment_changes_only_color_and_release_fields():
    source = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "mcp-app-blue",
            "namespace": "mcp-xp-staging",
            "resourceVersion": "123",
            "labels": {"app": "mcp-app", "app.kubernetes.io/slot": "blue"},
        },
        "spec": {
            "replicas": 2,
            "selector": {
                "matchLabels": {
                    "app": "mcp-app",
                    "app.kubernetes.io/slot": "blue",
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "mcp-app",
                        "app.kubernetes.io/slot": "blue",
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "mcp-app",
                            "image": "rejuvebio/mcp-xp:old",
                            "envFrom": [{"secretRef": {"name": "mcp-app-env"}}],
                        }
                    ]
                },
            },
        },
        "status": {"readyReplicas": 2},
    }

    result = render_deployment(
        source,
        "mcp-app-green",
        "green",
        "rejuvebio/mcp-xp:sha-abc",
        "mcp-app",
    )

    assert result["metadata"]["name"] == "mcp-app-green"
    assert "resourceVersion" not in result["metadata"]
    assert "status" not in result
    assert result["spec"]["selector"]["matchLabels"]["app.kubernetes.io/slot"] == "green"
    assert result["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/slot"] == "green"
    container = result["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "rejuvebio/mcp-xp:sha-abc"
    assert container["imagePullPolicy"] == "Always"
    assert container["envFrom"] == [{"secretRef": {"name": "mcp-app-env"}}]


def test_render_preview_service_is_internal_and_selects_candidate():
    source = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "mcp-app",
            "namespace": "mcp-xp-staging",
            "uid": "server-value",
        },
        "spec": {
            "type": "NodePort",
            "clusterIP": "10.0.0.10",
            "selector": {"app": "mcp-app", "app.kubernetes.io/slot": "blue"},
            "ports": [{"name": "http", "port": 8895, "nodePort": 30895}],
        },
    }

    result = render_service(source, "mcp-app-preview", "green")

    assert result["metadata"]["name"] == "mcp-app-preview"
    assert "uid" not in result["metadata"]
    assert result["spec"]["type"] == "ClusterIP"
    assert "clusterIP" not in result["spec"]
    assert "nodePort" not in result["spec"]["ports"][0]
    assert result["spec"]["selector"]["app.kubernetes.io/slot"] == "green"
