"use strict";

/**
 * Node-only fireye runner (jsdom, no real browser).
 * stdin JSON lines -> stdout JSON lines
 */

const fs = require("fs");
const path = require("path");
const readline = require("readline");
const { JSDOM } = require("jsdom");
const { installCanvas, installMatchMedia } = require("./dom_polyfill");

const ROOT = __dirname;
const AWSC_CODE = fs.readFileSync(path.join(ROOT, "awsc.js"), "utf8");
const FY_CODE = fs.readFileSync(path.join(ROOT, "fireyejs.js"), "utf8");
const ORIGIN = process.env.QWEN_FIREYE_ORIGIN || "https://chat.qwen.ai/";
const INIT_TIMEOUT_MS = Number(process.env.QWEN_FIREYE_TIMEOUT_MS || 20000);
const LOCATION = process.env.QWEN_FIREYE_LOCATION || "cn";

let domRef = null;
let fyBridge = null;
let initPromise = null;

function wait(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function isRealUa(v) {
  const s = String(v || "");
  return s.startsWith("231!") && s.length > 100;
}

function hookLocalScripts(window) {
  const document = window.document;
  const rawCreate = document.createElement.bind(document);
  document.createElement = function (tag) {
    const el = rawCreate(tag);
    if (String(tag).toLowerCase() !== "script") return el;
    let pending = "";
    Object.defineProperty(el, "src", {
      configurable: true,
      enumerable: true,
      get() {
        return pending;
      },
      set(v) {
        pending = String(v || "");
        queueMicrotask(() => {
          try {
            if (pending.includes("fireyejs")) {
              window.eval(FY_CODE);
              if (typeof el.onload === "function") el.onload();
            } else if (pending.includes("awsc.js")) {
              window.eval(AWSC_CODE);
              if (typeof el.onload === "function") el.onload();
            } else if (typeof el.onerror === "function") {
              el.onerror(new Error("unmapped script " + pending));
            }
          } catch (err) {
            if (typeof el.onerror === "function") el.onerror(err);
          }
        });
      },
    });
    return el;
  };
}

function buildDom() {
  const dom = new JSDOM(
    "<!doctype html><html><head></head><body></body></html>",
    {
      url: ORIGIN,
      referrer: ORIGIN,
      pretendToBeVisual: true,
      runScripts: "dangerously",
    }
  );
  const { window } = dom;
  try {
    Object.defineProperty(window.navigator, "javaEnabled", {
      value: () => false,
      configurable: true,
    });
  } catch (_e) {
    window.navigator.javaEnabled = () => false;
  }
  try {
    Object.defineProperty(window.navigator, "userAgent", {
      get() {
        return (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " +
          "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
        );
      },
      configurable: true,
    });
  } catch (_e) {}
  window.chrome = window.chrome || { runtime: {} };
  // fireye 会往 console 打调试对象，避免污染 stdout JSON 协议。
  window.console.log = function () {};
  window.console.info = function () {};
  window.console.debug = function () {};
  window.console.warn = function () {};
  installMatchMedia(window);
  installCanvas(window);
  hookLocalScripts(window);
  window.eval(AWSC_CODE);
  if (!window.AWSC || typeof window.AWSC.configFYEx !== "function") {
    throw new Error("AWSC.configFYEx missing");
  }
  return dom;
}

async function ensureInit() {
  if (fyBridge) {
    try {
      const sample = fyBridge.getFYToken({
        location: LOCATION,
        reqUrl: ORIGIN + "api/v2/chat/completions",
      });
      if (isRealUa(sample)) return fyBridge;
    } catch (_e) {}
  }
  if (initPromise) return initPromise;

  initPromise = (async () => {
    if (domRef) {
      try {
        domRef.window.close();
      } catch (_e) {}
      domRef = null;
    }
    domRef = buildDom();
    const { window } = domRef;
    const opts = {
      location: LOCATION,
      MaxMTLog: 20,
      MaxNGPLog: 10,
      MaxKSLog: 5,
      MaxFocusLog: 3,
    };
    const bridge = await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error("configFYEx timeout")),
        INIT_TIMEOUT_MS
      );
      try {
        window.AWSC.configFYEx(
          (mod) => {
            clearTimeout(timer);
            resolve(mod);
          },
          opts,
          INIT_TIMEOUT_MS
        );
      } catch (err) {
        clearTimeout(timer);
        reject(err);
      }
    });

    const deadline = Date.now() + INIT_TIMEOUT_MS;
    while (Date.now() < deadline) {
      try {
        const ua = bridge.getFYToken({
          location: LOCATION,
          reqUrl: ORIGIN + "api/v2/chat/completions",
        });
        if (isRealUa(ua)) {
          fyBridge = bridge;
          return fyBridge;
        }
      } catch (_e) {}
      await wait(50);
    }
    fyBridge = bridge;
    return fyBridge;
  })().finally(() => {
    initPromise = null;
  });

  return initPromise;
}

function reset() {
  fyBridge = null;
  initPromise = null;
  if (domRef) {
    try {
      domRef.window.close();
    } catch (_e) {}
    domRef = null;
  }
}

async function handle(msg) {
  const cmd = String((msg && msg.cmd) || "");
  if (cmd === "ping") return { ok: true, pong: true };
  if (cmd === "exit") return { ok: true, exit: true };
  if (cmd === "reset") {
    reset();
    return { ok: true, reset: true };
  }
  if (cmd === "token") {
    const mod = await ensureInit();
    const url =
      (msg && msg.url) || ORIGIN + "api/v2/chat/completions";
    const opt = { location: LOCATION, reqUrl: url };
    let bxUa = "";
    let bxUmidToken = "";
    try {
      bxUa = String(mod.getFYToken(opt) || "");
    } catch (err) {
      return {
        ok: false,
        error: "getFYToken",
        detail: String((err && err.message) || err),
      };
    }
    try {
      bxUmidToken = String(mod.getUidToken(opt) || "");
    } catch (_e) {
      bxUmidToken = "";
    }
    return {
      ok: true,
      bxUa,
      bxUmidToken,
      bxV: "2.5.37",
      uaLen: bxUa.length,
      umidLen: bxUmidToken.length,
      source: "fireyejs",
    };
  }
  return { ok: false, error: "unknown_cmd" };
}

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    crlfDelay: Infinity,
  });
  for await (const line of rl) {
    const raw = String(line || "").trim();
    if (!raw) continue;
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (err) {
      process.stdout.write(
        JSON.stringify({ ok: false, error: "bad_json", detail: String(err) }) +
          "\n"
      );
      continue;
    }
    try {
      const out = await handle(msg);
      process.stdout.write(JSON.stringify(out) + "\n");
      if (out.exit) process.exit(0);
    } catch (err) {
      process.stdout.write(
        JSON.stringify({
          ok: false,
          error: "runtime",
          detail: String((err && err.stack) || err),
        }) + "\n"
      );
    }
  }
}

main().catch((err) => {
  process.stderr.write(String((err && err.stack) || err) + "\n");
  process.exit(1);
});
