FROM node:20-alpine AS base

# Install OpenSSL for Prisma
RUN apk add --no-cache openssl

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts

# Generate Prisma client
COPY prisma ./prisma/
RUN npx prisma generate

# Copy source
COPY . .

# Build Next.js
RUN npm run build

# Production
FROM node:20-alpine AS runner
RUN apk add --no-cache openssl
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3182

COPY --from=base /app/package.json ./
COPY --from=base /app/node_modules ./node_modules
COPY --from=base /app/.next ./.next
COPY --from=base /app/public ./public
COPY --from=base /app/prisma ./prisma
COPY --from=base /app/server.mjs ./server.mjs

# Run migrations then start
CMD ["sh", "-c", "npx prisma db push --skip-generate && node server.mjs"]

EXPOSE 3182
