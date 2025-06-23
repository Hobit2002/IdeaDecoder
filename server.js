const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const { error } = require('console');
const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });
const domain = "jasmiapp.onrender.com"
const httpProtocol = "https"

const PORT = process.env.PORT || 3000;

// Serve the HTML file at root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'client_patient.html'));
});

// Serve listener page only if client_id is present in the speakers map
app.get('/listener_page', (req, res) => {
  const clientId = req.query.client_id;

  if (!clientId) {
    return res.status(400).send('Missing client_id parameter.');
  }

  if (speakers.has(clientId)) {
    res.sendFile(path.join(__dirname, 'client_listener.html'));
  } else {
    res.status(404).send('Client ID not found.');
  }
});

// Route to download audio and save as "episode.mp3"
app.get('/download', (req, res) => {
  const downloadPairs = [
    ['http://pytlik.pruzor.cz/Stachovi_hoste.mp3', path.join(__dirname, 'episode2.mp3')],
    ['http://pytlik.pruzor.cz/vocabularies.json', path.join(__dirname, 'vocabularies.json')]
  ];

  const downloadFile = (url, destination) => {
    return new Promise((resolve, reject) => {
      const file = fs.createWriteStream(destination);
      http.get(url, (response) => {
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download ${url}: Status code ${response.statusCode}`));
          return;
        }
        response.pipe(file);
        file.on('finish', () => {
          file.close(() => resolve()); // close the stream, then resolve
        });
      }).on('error', (err) => {
        fs.unlink(destination, () => {}); // delete partial file
        reject(err);
      });
    });
  };

  Promise.all(downloadPairs.map(([url, dest]) => downloadFile(url, dest)))
    .then(() => res.send('Download complete'))
    .catch(err => {
      console.error('Download failed:', err.message);
      res.status(500).send('Download failed: ' + err.message);
    });
});

// Optional: Serve static assets if needed
app.use(express.static(path.join(__dirname)));

// ---- WebSocket Logic ----

const speakers = new Map();       // key: client ID, value: ws
const transcribers = new Set();   // ws connections
const listeners = new Set();   // ws connections
const session_id = uuidv4();

console.log(`Server running on wss://localhost:${PORT}`);

wss.on('connection', (ws) => {
  ws.id = uuidv4();
  ws.role = null;

  ws.on('message', (data, isBinary) => {
    try {
      if (!isBinary) {
        const msg = JSON.parse(data.toString());

        if (msg.action === 'register_client') {
          ws.role = msg.role;
          if (ws.role === "speaker") {
            console.log("Speaker connected");
            speakers.set(ws.id, ws);
            
            // Load vocabularies.json and send to speaker
            const vocabPath = path.join(__dirname, 'vocabularies.json');
            fs.readFile(vocabPath, 'utf8', (err, data) => {
              if (err) {
                console.error("Error reading vocabularies.json:", err);
                return;
              }
              try {
                // Try to parse and re-serialize to ensure it's valid JSON
                const vocabs = JSON.parse(data);
                ws.send(JSON.stringify({
                  type: 'vocabularies',
                  vocabs: JSON.stringify(vocabs)
                }));
              } catch (e) {
                console.error("Invalid JSON in vocabularies.json:", e);
              }
            });

          } else if (ws.role === "transcriber") {
            console.log("Transcriber connected");
            transcribers.add(ws);
          } else if (ws.role === "listener") {
            console.log("Listeners connected");
            listeners.add(ws);
          }
        }

        if (msg.action === 'request_address') {
          const address = `${httpProtocol}://${domain}/listener_page?client_id=${ws.id}`;
          ws.send(JSON.stringify({ type: 'address', address }));
        }

        if (msg.action === 'transcription') {
          const { text, speaker_id, counter } = msg;
          console.log("Received transcription:", text);
          const target = speakers.get(speaker_id);
          // Send transcriptions to speaker
          if (target && target.readyState === WebSocket.OPEN) {
            target.send(JSON.stringify({ type: 'transcription', counter: counter, text }));
          }
          // Send transcriptions to listener
          Array.from(listeners).forEach( listener => {
            if (listener.readyState === WebSocket.OPEN) {
              listener.send(JSON.stringify({ type: 'transcription', counter: counter, text }))
            }
          })
        }

        if (msg.action === 'log'){
          const available = Array.from(transcribers).find(t => t.readyState === WebSocket.OPEN);
          if (available) {
            console.log("Received log message:", msg.msg)
            available.send(JSON.stringify({ action: "log", msg: msg.msg }));
          } else {
            console.log("No transcriber available.");
          }
        }
        if (msg.action == 'select_vocabulary'){
          const available = Array.from(transcribers).find(t => t.readyState === WebSocket.OPEN);
          if (available) {
            console.log("Received log message:", msg.argument)
            available.send(JSON.stringify({ action: "select_vocabulary", msg: msg.argument }));
          }
        }
      } else {
        if (ws.role === 'speaker') {
          const available = Array.from(transcribers).find(t => t.readyState === WebSocket.OPEN);
          if (available) {
            available.send(JSON.stringify({ action: "speaker_id", speaker_id: ws.id }));
            available.send(data);
          } else {
            console.log("No transcriber available.");
          }
        }
      }
    } catch (err) {
      console.error("Error processing message:", err);
    }
  });

  ws.on('close', () => {
    if (ws.role === 'speaker') speakers.delete(ws.id);
    if (ws.role === 'transcriber') transcribers.delete(ws);
  });

  setTimeout(() => {
    if (!ws.role) {
      ws.close();
      console.log("Unregistered connection closed");
    }
  }, 1000);
});

// Start server
server.listen(PORT, () => {
  console.log(`HTTP/WebSocket server listening on port ${PORT}`);
});
