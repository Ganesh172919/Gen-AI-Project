# FaithForge Frontend — Dockerfile

FROM node:20-alpine AS base
WORKDIR /app

# Accept API URL as build arg
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Build
COPY . .
RUN npm run build

# Production
EXPOSE 3000
CMD ["npm", "start"]
