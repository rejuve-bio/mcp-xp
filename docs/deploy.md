# Blue-green deployment

The repository deploys `mcp-xp` with two Kubernetes Deployments per
environment:

- `mcp-app-blue`
- `mcp-app-green`

The existing `mcp-app` Service is the stable entry point. Its
`app.kubernetes.io/slot` selector determines which color receives live traffic.
`mcp-app-preview` selects the inactive color for inspection before promotion.

## Delivery flow

1. A push to `staging` builds and pushes two images:
   `rejuvebio/mcp-xp:sha-<commit>` and `rejuvebio/mcp-xp:staging`.
2. The build workflow dispatches the immutable SHA tag to the staging workflow.
3. The workflow reads the live Service selector to find the active color.
4. It clones the active Deployment configuration to the inactive color, changing
   only the name, slot labels, image, and deployment annotations. This preserves
   the cluster's existing environment variables, secrets, volumes, probes, and
   resource settings.
5. It waits for the inactive Deployment and smoke-tests its `/` endpoint through
   a direct `kubectl port-forward`.
6. A successful candidate is retagged as
   `rejuvebio/mcp-xp:staging-stable`.
7. The live Service selector is changed to the validated color. The previous
   color remains running for an immediate rollback.
8. Production deployment is an explicit manual or `deploy-production` dispatch.
   It resolves `rejuvebio/mcp-xp:staging-stable` to an immutable image digest,
   deploys that digest to the inactive production color, validates it, and then
   switches the production Service.

The Service selector change is the traffic switch; this is a true blue-green
deployment rather than a normal rolling update with blue/green naming.

## First-run migration

The first deployment bootstraps from the existing cluster resources. Each
namespace must already contain:

- a Deployment named `mcp-app` with a container named `mcp-app`;
- a Service named `mcp-app` that routes to that Deployment;
- working application dependencies and image-pull credentials.

The workflow clones `mcp-app` to `mcp-app-blue`, waits until blue is healthy,
and adds `app.kubernetes.io/slot=blue` to the live Service selector. It then
creates the green candidate and a ClusterIP preview Service. The legacy
`mcp-app` Deployment is intentionally left unchanged during migration. It can
be scaled down or removed after the first successful deployment is verified.

If HPAs, PodDisruptionBudgets, NetworkPolicies, or monitoring rules target the
legacy Deployment by name, update them for both colored Deployments before
removing the legacy Deployment.

## Required GitHub secrets

| Secret | Purpose |
|---|---|
| `KUBECONFIG_STAGING` | kubeconfig for `mcp-xp-staging` |
| `KUBECONFIG_PRODUCTION` | kubeconfig for `mcp-xp-production` |
| `DOCKER_HUB_USERNAME` | Docker Hub account allowed to push `rejuvebio/mcp-xp` |
| `DOCKER_HUB_TOKEN` | Docker Hub access token |
| `EMAIL_USERNAME`, `EMAIL_PASSWORD` | Gmail SMTP credentials for deployment notifications |
| `RECIEVER_EMAIL` | Deployment-notification recipient |

The self-hosted deployment runner needs `kubectl`, Python 3, `curl`, Docker, and
network access to the clusters and Docker Hub.

## Manual operations

Both deployment workflows support **Run workflow** in GitHub Actions.

- Staging `deploy`: deploy the supplied image tag to the inactive color.
- Staging `rollback`: health-check the previous color, then switch traffic back.
- Production `deploy`: deploy the current `staging-stable` tag.
- Production `rollback`: health-check the previous color, then switch traffic
  back.

Production remains a separate approval boundary: passing staging does not
automatically change production.

## Failure behavior

- A rollout or smoke-test failure leaves the live Service on its current color.
- A Docker tag-promotion failure also prevents the Service switch.
- A failed production candidate leaves production traffic unchanged.
- Rollback validates the previous color before changing the Service selector.
