FROM node:20-alpine AS base

# Install OpenSSL for Prisma
RUN apk add --no-cache openssl

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

# Generate Prisma client
COPY prisma ./prisma/
COPY prisma.config.ts ./
RUN npx prisma generate

# Copy source
COPY . .

# Build Next.js — env vars must be set at build time for App Router
# (Next.js App Router route handlers only see build-time env values)
ARG NEXTAUTH_URL=http://localhost:3180/social/api/auth
ENV NEXTAUTH_URL=${NEXTAUTH_URL}
ENV AUTH_CASDOOR_ISSUER=http://localhost:8380
ENV AUTH_CASDOOR_ID=social-app
ENV AUTH_CASDOOR_SECRET=social-casdoor-secret-2026
ENV AUTH_SECRET=street-voices-social-secret-2026
ENV DATABASE_URL=postgresql://lobehub:lobehub_password@localhost:5433/social
RUN npm run build

# Production
FROM node:20-alpine AS runner
RUN apk add --no-cache openssl
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3182
# These must also be set in the runner stage for Prisma and other runtime code
ENV DATABASE_URL=postgresql://lobehub:lobehub_password@localhost:5433/social
ENV NEXTAUTH_URL=http://localhost:3180/social/api/auth
ENV AUTH_CASDOOR_ISSUER=http://localhost:8380
ENV AUTH_CASDOOR_ID=social-app
ENV AUTH_CASDOOR_SECRET=social-casdoor-secret-2026
ENV AUTH_SECRET=street-voices-social-secret-2026

COPY --from=base /app/package.json ./
COPY --from=base /app/node_modules ./node_modules
COPY --from=base /app/.next ./.next
COPY --from=base /app/next.config.mjs ./next.config.mjs
COPY --from=base /app/.env.production ./.env.production
COPY --from=base /app/public ./public
COPY --from=base /app/prisma ./prisma
COPY --from=base /app/prisma.config.ts ./prisma.config.ts
COPY --from=base /app/server.mjs ./server.mjs
COPY --from=base /app/src/generated ./src/generated

CMD ["node", "server.mjs"]

EXPOSE 3182
