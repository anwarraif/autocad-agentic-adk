# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# web — Next.js viewer.
#
# Three stages so the runtime image carries neither the toolchain nor the full
# node_modules tree: `next build` with output:"standalone" emits a server with
# only the packages actually imported.
# ---------------------------------------------------------------------------

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
# `npm ci` needs a lockfile; fall back to `install` on first build when one
# has not been generated yet.
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi


FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* is inlined at build time, not read at runtime, so the browser
# needs the host-visible API URL baked in here.
ARG NEXT_PUBLIC_API_URL=http://localhost:4311
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build


FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup -g 10001 -S nodejs && adduser -u 10001 -S nextjs -G nodejs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
