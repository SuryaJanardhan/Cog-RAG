FROM node:18-alpine

WORKDIR /app

# Install dependencies first (leverage Docker layers caching)
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

EXPOSE 3000

# Start dev or prod server
CMD ["npm", "start"]
