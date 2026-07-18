# deploy_lab_remote.sh

## Purpose

Deploy the two-GPU laboratory without accidentally interrupting a persistent
organism that shares the supervisor's systemd cgroup.

## Modes

### `frontend-only`

- Runs the TypeScript check and production build locally.
- Copies static assets into the deploy worktree.
- Never checks out backend code or restarts the service, so active trainers continue.

### `full`

- Refuses to proceed when any deploy-owned `petridish.train_shakespeare` process is
  alive.
- Fetches the configured branch, checks out its exact commit, copies the verified
  frontend, and restarts the monitor service only after the guard passes.

## Configuration

The remote, deploy root, service, and branch may be overridden with
`PETRIDISH_LAB_REMOTE`, `PETRIDISH_LAB_DEPLOY_ROOT`, `PETRIDISH_LAB_SERVICE`, and
`PETRIDISH_LAB_BRANCH`.

## Contract

A full deployment must never rely on trainer SIGTERM recovery as an ordinary update
mechanism. Same-organism continuation remains checkpoint-safe, but service deployment
is required to avoid creating that interruption in the first place.

