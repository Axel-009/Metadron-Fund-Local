# Real Time & Query API for Financial News

- Covers more than 10,000 news sources (Reuters, Bloomberg, The Wall
  Street Journal, Seeking Alpha, SEC, and many more). [See the full list of
  sources here.](https://newsfilter.io)
- Real-time financial news stream using websockets. The API returns a new article
  as soon as it is published on one of the supported news platforms.
- Query API to search the [newsfilter.io](https://newsfilter.io) article database
- JSON formatted
- Articles mapped to company ticker
- Supports Python, C++, JavaScript (Node.js, React, jQuery, Angular, Vue), Java
  and Excel plugins using websocket or socket.io clients

![https://i.imgur.com/Rd7x9Mi.png](https://i.imgur.com/Rd7x9Mi.png)

# Real Time Streaming API

You can use the streaming API in your command line, or develop your own application
using the API as imported package. Both options are explained below.

**Before you start**:

- Install Node.js if you haven't already. On Mac in the command line type `brew install node`.

## Command Line

Type in your command line:

1. `npm install realtime-newsapi -g` to install the package
2. `realtime-newsapi` to connect to the stream
3. Done! You will see new articles printed in your command line
   as soon as they are published on one of the supported news platforms.

## Node.js

Type in your command line:

1. `mkdir my-project && cd my-project` to create a new folder for your project.
2. `npm init -y` to set up Node.js boilerplate.
3. `npm install realtime-newsapi` to install the package.
4. `touch index.js` to create a new file. Copy/paste the example code below
   into the file index.js.

```js
const api = require('realtime-newsapi')();

api.on('articles', (articles) => {
  console.log(articles);
});
```

5. `node index.js` to start listening for new articles.

## Live Demo

[https://codesandbox.io/s/k5q6nwqkrr](https://codesandbox.io/s/k5q6nwqkrr)

# Query API

Use `curl` or Postman to send requests to the query API.

- API endpoint: `https://api.newsfilter.io/public/actions`
- Supported HTTP Method: `POST`
- Supported content type: `JSON`

# Consulting

This service is already used around the world by startups, top news organizations, graduate school researchers,
and, of course, hackers like you :)
