FROM node:20-alpine

WORKDIR /usr/src/app

COPY packages/db-migrate/package*.json ./
RUN npm install

COPY packages/db-migrate ./

CMD ["npm", "run", "start:dev"]
