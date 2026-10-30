# TrashTheTest frontend (pnpm workspace aware; dev server on $PORT).
#
# Build from the REPO ROOT so the pnpm workspace (catalog: deps such as
# @workspace/api-client-react) resolves:
#
#   docker build -f artifacts/trash-the-test/Dockerfile .
#   # or via compose (context already set to the root): docker compose up --build frontend
#
# npm install does NOT work here: package.json uses the `catalog:` protocol,
# which only pnpm understands. PORT defaults to 5173 to match the compose
# 5173:5173 port mapping (vite.config.ts reads $PORT, fallback 3000).
# NOTE: glibc-based slim, NOT alpine. pnpm-workspace.yaml `overrides` strip
# the *-musl native binaries (lightningcss, tailwind oxide) and keep only
# linux-x64-gnu, so musl-based alpine can never load them. slim matches the
# lockfile's kept linux-x64-gnu artifacts with zero workspace changes.
FROM node:24-slim
RUN corepack enable
WORKDIR /repo

# Workspace manifests first for a cached dependency layer.
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml .npmrc ./
COPY artifacts/trash-the-test/package.json ./artifacts/trash-the-test/package.json
COPY lib/api-client-react/package.json ./lib/api-client-react/package.json
RUN pnpm install --filter @workspace/trash-the-test... --frozen-lockfile

# Sources + TS project references needed by `pnpm dev` / `pnpm build`.
COPY tsconfig.base.json tsconfig.json ./
COPY lib/api-client-react ./lib/api-client-react
COPY artifacts/trash-the-test ./artifacts/trash-the-test

# Re-link workspace bins AFTER the source copy. The copy above must never
# clobber node_modules (see root .dockerignore), but this offline re-run is a
# cheap no-op when links are clean and repairs them otherwise, so the runtime
# `vite` binary is always present. devDependencies (vite lives there) are
# intentionally installed: the container runs the vite dev server, not prod.
RUN pnpm install --filter @workspace/trash-the-test... --frozen-lockfile --prefer-offline

WORKDIR /repo/artifacts/trash-the-test

EXPOSE 5173
ENV PORT=5173
CMD ["pnpm", "run", "dev"]
