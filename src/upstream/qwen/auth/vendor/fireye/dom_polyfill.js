"use strict";

/** Minimal WebGLRenderingContext stub for fireyejs fingerprinting. */

function makeWebGL(canvas) {
  const noop = function () {};
  const gl = {
    canvas,
    drawingBufferWidth: 300,
    drawingBufferHeight: 150,
    VERTEX_SHADER: 35633,
    FRAGMENT_SHADER: 35632,
    ARRAY_BUFFER: 34962,
    ELEMENT_ARRAY_BUFFER: 34963,
    STATIC_DRAW: 35044,
    TRIANGLES: 4,
    COLOR_BUFFER_BIT: 16384,
    DEPTH_BUFFER_BIT: 256,
    FLOAT: 5126,
    UNSIGNED_SHORT: 5123,
    UNSIGNED_BYTE: 5121,
    DEPTH_TEST: 2929,
    LINK_STATUS: 35714,
    COMPILE_STATUS: 35713,
    RENDERER: 7937,
    VENDOR: 7936,
    VERSION: 7938,
    SHADING_LANGUAGE_VERSION: 35724,
    MAX_TEXTURE_SIZE: 3379,
    MAX_VERTEX_ATTRIBS: 34921,
    getParameter(p) {
      if (p === 37445) return "Google Inc. (NVIDIA)";
      if (p === 37446) return "ANGLE (NVIDIA, NVIDIA GeForce GTX 1080 Direct3D11 vs_5_0 ps_5_0)";
      if (p === gl.RENDERER) return "WebKit WebGL";
      if (p === gl.VENDOR) return "WebKit";
      if (p === gl.VERSION) return "WebGL 1.0 (OpenGL ES 2.0 Chromium)";
      if (p === gl.SHADING_LANGUAGE_VERSION) return "WebGL GLSL ES 1.0 (OpenGL ES GLSL ES 1.0 Chromium)";
      if (p === gl.MAX_TEXTURE_SIZE) return 16384;
      if (p === gl.MAX_VERTEX_ATTRIBS) return 16;
      return 0;
    },
    getExtension(name) {
      if (name === "WEBGL_debug_renderer_info") {
        return {
          UNMASKED_VENDOR_WEBGL: 37445,
          UNMASKED_RENDERER_WEBGL: 37446,
        };
      }
      return {};
    },
    getSupportedExtensions() {
      return [
        "WEBGL_debug_renderer_info",
        "EXT_texture_filter_anisotropic",
        "OES_texture_float",
      ];
    },
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
    getShaderInfoLog: () => "",
    getProgramInfoLog: () => "",
    getAttribLocation: () => 0,
    getUniformLocation: () => ({}),
    enableVertexAttribArray: noop,
    vertexAttribPointer: noop,
    uniform1f: noop,
    uniform1i: noop,
    uniform2f: noop,
    uniform3f: noop,
    uniform4f: noop,
    uniformMatrix4fv: noop,
    drawArrays: noop,
    drawElements: noop,
    viewport: noop,
    clear: noop,
    clearColor: noop,
    enable: noop,
    disable: noop,
    blendFunc: noop,
    depthFunc: noop,
    createTexture: () => ({}),
    bindTexture: noop,
    texImage2D: noop,
    texParameteri: noop,
    activeTexture: noop,
    pixelStorei: noop,
    readPixels: noop,
    flush: noop,
    finish: noop,
    isContextLost: () => false,
    getContextAttributes: () => ({
      alpha: true,
      antialias: true,
      depth: true,
      failIfMajorPerformanceCaveat: false,
      powerPreference: "default",
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      stencil: false,
      desynchronized: false,
      xrCompatible: false,
    }),
  };
  return gl;
}

function make2d(canvas) {
  return {
    canvas,
    fillStyle: "#000",
    strokeStyle: "#000",
    font: "14px Arial",
    textBaseline: "alphabetic",
    globalCompositeOperation: "source-over",
    fillRect() {},
    clearRect() {},
    strokeRect() {},
    getImageData(x, y, w, h) {
      const width = w || 1;
      const height = h || 1;
      return {
        data: Uint8ClampedArray.from(
          { length: width * height * 4 },
          (_, i) => (i * 17 + 3) % 255
        ),
        width,
        height,
      };
    },
    putImageData() {},
    createImageData(w, h) {
      return {
        data: new Uint8ClampedArray((w || 1) * (h || 1) * 4),
        width: w || 1,
        height: h || 1,
      };
    },
    measureText(t) {
      return { width: String(t || "").length * 7 };
    },
    fillText() {},
    strokeText() {},
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    stroke() {},
    fill() {},
    arc() {},
    rect() {},
    save() {},
    restore() {},
    setTransform() {},
    translate() {},
    scale() {},
    rotate() {},
    transform() {},
    drawImage() {},
    createLinearGradient() {
      return { addColorStop() {} };
    },
    createRadialGradient() {
      return { addColorStop() {} };
    },
    createPattern() {
      return {};
    },
  };
}

function installCanvas(window) {
  const proto = window.HTMLCanvasElement && window.HTMLCanvasElement.prototype;
  if (!proto) return;
  proto.getContext = function (type) {
    const kind = String(type || "");
    if (kind.includes("webgl") || kind === "experimental-webgl") {
      return makeWebGL(this);
    }
    return make2d(this);
  };
  proto.toDataURL = function () {
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
  };
}

function installMatchMedia(window) {
  if (typeof window.matchMedia === "function") return;
  window.matchMedia = function (query) {
    return {
      matches: false,
      media: String(query || ""),
      onchange: null,
      addListener() {},
      removeListener() {},
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {
        return false;
      },
    };
  };
}

module.exports = { installCanvas, installMatchMedia, makeWebGL, make2d };
