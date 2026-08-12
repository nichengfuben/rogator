"use strict";
/** 一次性逆向：jsdom 跑 fireyejs，poll init，导出 bx-ua 样本（不参与运行时）。 */

const fs = require("fs");
const path = require("path");
const { webcrypto } = require("crypto");
const { JSDOM } = require("jsdom");

const ROOT = __dirname;
const AWSC = fs.readFileSync(path.join(ROOT, "awsc.js"), "utf8");
const FY = fs.readFileSync(path.join(ROOT, "fireyejs.js"), "utf8");
const ORIGIN = "https://chat.qwen.ai/";

function installCanvas(window) {
  const noop = () => {};
  const HCE = window.HTMLCanvasElement;
  HCE.prototype.getContext = function (type) {
    if (type !== "2d" && !String(type).includes("webgl")) return null;
    const gl = {
      canvas: this,
      drawingBufferWidth: 300,
      drawingBufferHeight: 150,
      getParameter: (p) => {
        if (p === 37445) return "Google Inc. (NVIDIA)";
        if (p === 37446)
          return "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0)";
        return 0;
      },
      getExtension: (n) =>
        n === "WEBGL_debug_renderer_info"
          ? { UNMASKED_VENDOR_WEBGL: 37445, UNMASKED_RENDERER_WEBGL: 37446 }
          : {},
      getSupportedExtensions: () => ["WEBGL_debug_renderer_info"],
      getAttribLocation: () => 0,
      createBuffer: () => ({}),
      bindBuffer: noop,
      bufferData: noop,
      createShader: () => ({}),
      shaderSource: noop,
      compileShader: noop,
      createProgram: () => ({}),
      attachShader: noop,
      linkProgram: noop,
      useProgram: noop,
      getShaderParameter: () => 1,
      getProgramParameter: () => 1,
      enableVertexAttribArray: noop,
      vertexAttribPointer: noop,
      drawArrays: noop,
      viewport: noop,
      clear: noop,
    };
    return gl;
  };
  if (typeof window.matchMedia !== "function") {
    window.matchMedia = () => ({ matches: false, addListener: noop, removeListener: noop });
  }
}

function hookCrypto(window, captures) {
  if (!window.crypto) {
    window.crypto = webcrypto;
  } else if (!window.crypto.subtle) {
    window.crypto.subtle = webcrypto.subtle;
  }
  const subtle = window.crypto.subtle;
  if (!subtle || typeof subtle.encrypt !== "function") {
    return;
  }
  const origEncrypt = subtle.encrypt.bind(subtle);
  subtle.encrypt = async function (algo, key, data) {
    captures.push({
      algo: algo && algo.name,
      ivLen: algo && algo.iv ? algo.iv.byteLength : 0,
      dataLen: data.byteLength,
    });
    return origEncrypt(algo, key, data);
  };
}

function buildDom(captures) {
  const dom = new JSDOM("<!doctype html><html><head></head><body></body></html>", {
    url: ORIGIN,
    referrer: ORIGIN,
    pretendToBeVisual: true,
    runScripts: "dangerously",
  });
  const { window } = dom;
  installCanvas(window);
  hookCrypto(window, captures);
  window.console.log = () => {};
  const rawCreate = window.document.createElement.bind(window.document);
  window.document.createElement = function (tag) {
    const el = rawCreate(tag);
    if (String(tag).toLowerCase() !== "script") return el;
    let pending = "";
    Object.defineProperty(el, "src", {
      configurable: true,
      get() {
        return pending;
      },
      set(v) {
        pending = String(v || "");
        queueMicrotask(() => {
          if (pending.includes("fireyejs")) window.eval(FY);
          else if (pending.includes("awsc")) window.eval(AWSC);
          if (typeof el.onload === "function") el.onload();
        });
      },
    });
    return el;
  };
  window.eval(AWSC);
  return dom;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitReady(mod, ms = 20000) {
  const start = Date.now();
  while (Date.now() - start < ms) {
    const umid = String(mod.getUidToken({ location: "cn" }) || "");
    if (umid.length > 20) return umid;
    await sleep(100);
  }
  return "";
}

async function main() {
  const captures = [];
  const dom = buildDom(captures);
  const { window } = dom;
  const mod = await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("configFYEx timeout")), 25000);
    window.AWSC.configFYEx(
      (m) => {
        clearTimeout(t);
        resolve(m);
      },
      { location: "cn", MaxMTLog: 20, MaxNGPLog: 10, MaxKSLog: 5, MaxFocusLog: 3 },
      25000
    );
  });
  const umid = await waitReady(mod);
  const urls = [
    "https://chat.qwen.ai/api/v2/chats/new",
    "https://chat.qwen.ai/api/v2/chat/completions?chat_id=f07fc0a2-f718-4076-8f7d-56834a8013bb",
  ];
  const probe = String(mod.getFYToken({ location: "cn", reqUrl: urls[0] }) || "");
  const out = { tokens: [], umid, probe, cryptoCaptures: captures.slice(0, 20) };
  for (const reqUrl of urls) {
    const ua = String(mod.getFYToken({ location: "cn", reqUrl }) || "");
    out.tokens.push({ reqUrl, ua, len: ua.length });
  }
  fs.writeFileSync(path.join(ROOT, "dump.json"), JSON.stringify(out, null, 2));
  console.log(JSON.stringify(out, null, 2));
  dom.window.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
