# Deploy workflows

Two Kubernetes deploy workflows, mirroring the pattern used by
`hypothesis-generation-demo`:

| Workflow | Trigger event | Target namespace |
|---|---|---|
| `cd-staging-deployment.yml` | `repository_dispatch` type `deploy-staging` | `mcp-xp-staging` (hetzner cluster) |
| `cd-production-deployment.yml` | `repository_dispatch` type `deploy-production` | `mcp-xp-production` (app.rejuve.bio cluster) |

Both run on a **self-hosted runner** that has cluster network access
and the `kubectl` binary available. Neither auto-deploys on push or
merge — each requires an explicit `repository_dispatch` event.

## How each deploy fires

The workflow expects a `repository_dispatch` payload:

```json
{
  "event_type": "deploy-staging",
  "client_payload": { "image_tag": "sha-abc1234" }
}
```

`image_tag` defaults to `"staging"` (or `"production"`) if the payload
is empty.

Typical trigger paths:

**From a CI image-build workflow** (recommended once one exists):
```yaml
- uses: peter-evans/repository-dispatch@v3
  with:
    token: ${{ secrets.REPO_DISPATCH_PAT }}
    repository: rejuve-bio/mcp-xp
    event-type: deploy-staging
    client-payload: '{"image_tag": "${{ needs.build.outputs.tag }}"}'
```

**Manually via `gh`**:
```bash
gh api /repos/rejuve-bio/mcp-xp/dispatches \
  -f event_type=deploy-staging \
  -f client_payload='{"image_tag":"sha-abc1234"}'
```

**Manually via curl**:
```bash
curl -X POST -H "Authorization: token <GITHUB_PAT>" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/rejuve-bio/mcp-xp/dispatches \
  -d '{"event_type":"deploy-staging","client_payload":{"image_tag":"sha-abc1234"}}'
```

## What each workflow does

1. Debug — echoes event type, image tag, and actor.
2. Checkout the repo.
3. Setup kubectl — writes the kubeconfig from the environment's secret
   into `~/.kube/config`.
4. `kubectl set image deployment/mcp-app mcp-app=rejuvebio/mcp-xp:$TAG
   -n <namespace>` — updates the single mcp-xp deployment.
5. `kubectl rollout status deployment/mcp-app -n <namespace>
   --timeout=180s` — waits for the rollout to complete or fails at 3
   minutes.
6. `kubectl get pods -n <namespace>` — logs the current pod state.
7. Success / failure email via `dawidd6/action-send-mail@v3`.

## One-time setup

### Required secrets (repo-level)

| Secret | Purpose |
|---|---|
| `KUBECONFIG_STAGING` | full kubeconfig text pointing at the hetzner k8s cluster |
| `KUBECONFIG_PRODUCTION` | full kubeconfig text pointing at the app.rejuve.bio k8s cluster |
| `EMAIL_USERNAME`, `EMAIL_PASSWORD` | Gmail SMTP creds used by `action-send-mail` |
| `RECIEVER_EMAIL` | recipient address for deploy notifications |

### Required cluster state

- A namespace `mcp-xp-staging` on the hetzner cluster with a Deployment
  named `mcp-app` and a container `mcp-app` running the image
  `rejuvebio/mcp-xp:staging`. `kubectl set image` only updates the
  image reference on an existing Deployment; it doesn't create one.
- Same shape on the production cluster in namespace
  `mcp-xp-production` with the image tag `production`.
- Each Deployment's environment includes `JWT_SECRET`: the
  `JWT_SECRET_KEY` of the auth service whose login tokens that
  environment accepts (comma-separate several). Without it the app
  and the MCP server refuse to start. Roll a new environment out with
  `JWT_VERIFY=report` first, check the logs for "would be rejected",
  then switch to `enforce` (the default). See `.env.example`.
- Whatever registry hosts `rejuvebio/mcp-xp` needs to be accessible
  from both clusters (either public Docker Hub or a
  `imagePullSecrets`-configured private registry).
- A separate CI workflow that **builds and pushes** the docker image
  before firing the dispatch (this repo currently has no such
  workflow — you'll need to add one, or push images manually).

### Self-hosted runner

The workflows use `runs-on: self-hosted`. There needs to be at least
one registered runner in this repo (or an org-level runner group the
repo has access to) that has:

- Network reachability to the two k8s clusters
- `kubectl` installed
- No secrets exposed via other jobs on the same runner

Register a runner from Settings → Actions → Runners.

## Rollback

Re-fire the dispatch with the previous good tag:

```bash
gh api /repos/rejuve-bio/mcp-xp/dispatches \
  -f event_type=deploy-staging \
  -f client_payload='{"image_tag":"sha-<previous-good>"}'
```

State (Redis, Qdrant, the timestamp state file) is on persistent
volumes and unaffected by the pod image change.
