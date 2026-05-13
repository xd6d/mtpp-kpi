/**
 * Лабораторна робота №3 - допоміжний процес (Node.js).
 * Логи пишуться у worker.log (поряд із цим скриптом), щоб не переповнити
 * OS-pipe для stderr — це причина TimeoutExpired при великій кількості кругів.
 */
"use strict";

const fs   = require("fs");
const net  = require("net");
const path = require("path");
const readline = require("readline");

const mode = process.argv[2];
const LOG_PATH = path.join(__dirname, "worker.log");
const logFd = fs.openSync(LOG_PATH, "w");

function log(msg) {
  fs.writeSync(logFd, `[${mode} pid=${process.pid}] ${msg}\n`);
}
function stderrOnce(msg) {
  process.stderr.write(`[worker:${mode} pid=${process.pid}] ${msg}\n`);
}

stderrOnce(`старт (логи у ${LOG_PATH})`);
log(`старт у режимі ${mode}`);

// ----------------------------- PIPE -----------------------------------------
function runPipe() {
  const rl = readline.createInterface({ input: process.stdin });
  rl.on("line", (line) => {
    const cleaned = line.replace(/\r$/, "");   // на Windows може лишатися \r
    const value = Number(cleaned);
    log(`pipe got ${value}`);
    process.stdout.write(`${value}\n`);
  });
  rl.on("close", () => {
    log("stdin закрито, завершуємось");
    try { fs.closeSync(logFd); } catch (_) {}
    process.exit(0);
  });
}

// ----------------------------- SOCKET ---------------------------------------
function runSocket() {
  const port = Number(process.argv[3]);
  const client = net.createConnection({ host: "127.0.0.1", port });
  let buffer = "";
  client.on("data", (chunk) => {
    buffer += chunk.toString();
    let idx;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx).replace(/\r$/, "");
      buffer = buffer.slice(idx + 1);
      const value = Number(line);
      log(`socket got ${value}`);
      client.write(`${value}\n`);
    }
  });
  client.on("end", () => { try { fs.closeSync(logFd); } catch (_) {} process.exit(0); });
  client.on("error", (e) => { log("socket error: " + e.message); process.exit(1); });
}

// ----------------------------- SHARED MEMORY (mmap) -------------------------
function runShmem() {
  const filePath = process.argv[3];
  log(`mmap файл ${filePath}`);
  const fd = fs.openSync(filePath, "r+");
  const buf = Buffer.alloc(16);
  let lastReqId = 0;
  while (true) {
    fs.readSync(fd, buf, 0, 16, 0);
    const reqId = buf.readUInt32LE(0);
    if (reqId === 0xFFFFFFFF) {
      log("отримано сигнал завершення");
      fs.closeSync(fd);
      try { fs.closeSync(logFd); } catch (_) {}
      process.exit(0);
    }
    if (reqId !== lastReqId && reqId !== 0) {
      const value = buf.readUInt32LE(4);
      log(`shmem got id=${reqId} value=${value}`);
      const valBuf = Buffer.alloc(4); valBuf.writeUInt32LE(value, 0);
      fs.writeSync(fd, valBuf, 0, 4, 12);
      const idBuf = Buffer.alloc(4);  idBuf.writeUInt32LE(reqId, 0);
      fs.writeSync(fd, idBuf, 0, 4, 8);
      lastReqId = reqId;
    }
  }
}

switch (mode) {
  case "pipe":   runPipe();   break;
  case "socket": runSocket(); break;
  case "shmem":  runShmem();  break;
  default: stderrOnce(`невідомий режим: ${mode}`); process.exit(2);
}