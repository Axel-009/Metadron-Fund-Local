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

# Real Time Streaming API

You can use the streaming API in your command line, or develop your own application
using the API as imported package.

## Command Line

1. `npm install realtime-newsapi -g` to install the package
2. `realtime-newsapi` to connect to the stream

## Node.js

1. `mkdir my-project && cd my-project`
2. `npm init -y`
3. `npm install realtime-newsapi`
4. Create index.js with the example code
5. `node index.js` to start listening

# Query API

- API endpoint: `https://api.newsfilter.io/public/actions`
- Supported HTTP Method: `POST`
- Supported content type: `JSON`
