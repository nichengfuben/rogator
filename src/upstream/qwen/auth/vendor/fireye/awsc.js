!(function (e, t) {
  var n = 1e4,
    g_moduleConfig = {
      uabModule: {
        grey: ["AWSC/uab/1.140.0/collina.js"],
        stable: ["AWSC/uab/1.140.0/collina.js"],
        greyBr: ["AWSC-br/uab/1.140.0/collina.js"],
        stableBr: ["AWSC-br/uab/1.140.0/collina.js"],
        ratio: 1e4,
        greyConfig: {},
        stableConfig: {},
      },
      fyModule: {
        grey: ["AWSC/fireyejs/1.231.69/fireyejs.js"],
        stable: ["AWSC/fireyejs/1.231.69/fireyejs.js"],
        greyBr: ["AWSC-br/fireyejs/1.227.0/fireyejs.js"],
        stableBr: ["AWSC-br/fireyejs/1.227.0/fireyejs.js"],
        ratio: 1e4,
        greyConfig: {},
        stableConfig: {},
      },
      nsModule: {
        grey: ["js/nc/60.js"],
        stable: ["js/nc/60.js"],
        ratio: 1e4,
        greyConfig: {},
        stableConfig: {},
      },
      umidPCModule: {
        grey: ["AWSC/WebUMID/1.93.0/um.js"],
        stable: ["AWSC/WebUMID/1.93.0/um.js"],
        greyBr: ["AWSC-br/WebUMID/1.93.0/um.js"],
        stableBr: ["AWSC-br/WebUMID/1.93.0/um.js"],
        ratio: 1e4,
        greyConfig: {},
        stableConfig: {},
      },
      etModule: {
        grey: ["AWSC/et/1.83.41/et_f.js", "AWSC/et/1.83.41/et_f.js"],
        stable: ["AWSC/et/1.83.41/et_f.js", "AWSC/et/1.83.41/et_f.js"],
        greyBr: ["AWSC-br/et/1.80.0/et_f.js", "AWSC-br/et/1.80.1/et_n.js"],
        stableBr: ["AWSC-br/et/1.80.0/et_f.js", "AWSC-br/et/1.80.1/et_n.js"],
        ratio: 1e4,
        greyConfig: { whitelist: ["*"] },
        stableConfig: { whitelist: ["*"] },
      },
      ncModule: {
        grey: ["AWSC/nc/1.97.0/nc.js"],
        stable: ["AWSC/nc/1.97.0/nc.js"],
        ratio: 1e4,
        greyConfig: {},
        stableConfig: {},
      },
    },
    o = [
      {
        name: "umidPCModule",
        features: ["umpc", "um", "umh5"],
        depends: [],
        sync: !1,
      },
      { name: "uabModule", features: ["uab"], depends: [], sync: !1 },
      { name: "fyModule", features: ["fy"], depends: [], sync: !1 },
      { name: "nsModule", features: ["ns"], depends: [], sync: !1 },
      { name: "etModule", features: ["et"], depends: [], sync: !1 },
      {
        name: "ncModule",
        features: ["nc", "nvc", "ic"],
        depends: ["fy"],
        sync: !1,
      },
    ];
  e.location.hostname &&
    "" != e.location.hostname &&
    (/^(?:[\w-]+\.)*(cainiao|dingtalk|aliexpress|1688|youku|amap|goofish|fliggy)\.com$/.test(
      e.location.hostname
    ) ||
      e.location.hostname.indexOf("lazlogistics.in.th") > -1 ||
      e.location.hostname.indexOf("aliexpress.us") > -1 ||
      e.location.hostname.indexOf("ele.me") > -1 ||
      e.location.hostname.indexOf("quark.cn") > -1 ||
      e.location.hostname.indexOf("alimama.com") > -1 ||
      e.location.hostname.indexOf("chat.qwen.ai") > -1 ||
      e.location.hostname.indexOf("damai.cn") > -1) &&
    ((g_moduleConfig.fyModule.grey = ["AWSC/fireyejs/1.231.67/fireyejs.js"]),
    (g_moduleConfig.fyModule.stable = ["AWSC/fireyejs/1.231.67/fireyejs.js"])),
    e.location.hostname &&
      "" != e.location.hostname &&
      /^(?:[\w-]+\.)*(taobao|tmall)\.com$/.test(e.location.hostname) &&
      (/WindVane/i.test(window.navigator.userAgent)
        ? ((g_moduleConfig.fyModule.grey = [
            "AWSC/fireyejs/1.234.20/fireyejs.js",
          ]),
          (g_moduleConfig.fyModule.stable = [
            "AWSC/fireyejs/1.234.20/fireyejs.js",
          ]))
        : ((g_moduleConfig.fyModule.grey = [
            "AWSC/fireyejs/1.234.24/fireyejs.js",
          ]),
          (g_moduleConfig.fyModule.stable = [
            "AWSC/fireyejs/1.234.24/fireyejs.js",
          ])),
      (g_moduleConfig.etModule.grey = [
        "AWSC/et/1.84.2/et_f.js",
        "AWSC/et/1.84.2/et_f.js",
      ])),
    window.location &&
      window.location.href &&
      window.location.href.indexOf("detail.1688.com/offer") > -1 &&
      ((g_moduleConfig.fyModule.grey = ["AWSC/fireyejs/1.234.29/fireyejs.js"]),
      (g_moduleConfig.fyModule.stable = ["AWSC/fireyejs/1.231.67/fireyejs.js"]),
      (g_moduleConfig.fyModule.ratio = 1e4),
      (g_moduleConfig.etModule.grey = [
        "AWSC/et/1.84.10/et_f.js",
        "AWSC/et/1.84.10/et_f.js",
      ]),
      (g_moduleConfig.etModule.ratio = 1e4)),
    e.location &&
      e.location.hostname &&
      "" != e.location.hostname &&
      e.location.hostname.indexOf("qianwen.com") > -1 &&
      ((g_moduleConfig.fyModule.grey = ["AWSC/fireyejs/1.234.29/fireyejs.js"]),
      (g_moduleConfig.fyModule.stable = ["AWSC/fireyejs/1.231.67/fireyejs.js"]),
      (g_moduleConfig.fyModule.ratio = 10));
  var a = navigator.userAgent,
    r = a.match(/Chrome\/(\d*)/);
  r && (r = +r[1]);
  var i = a.match(/Edge\/([\d]*)/),
    s = a.match(/Safari\/([\d]*)/),
    l = a.match(/Firefox\/([\d]*)/),
    f = a.match(/MSIE|Trident/);
  function c() {
    var e = "function%20javaEnabled%28%29%20%7B%20%5Bnative%20code%5D%20%7D";
    return "WebkitAppearance" in document.documentElement.style &&
      escape(navigator.javaEnabled.toString()) === e
      ? !0
      : !1;
  }
  function d(t, o) {
    var a = "AWSC_SPECIFY_" + t.toUpperCase() + "_ADDRESSES";
    if (e[a]) return e[a];
    var d = { normalAddresses: [], brAddresses: [] };
    for (var u in g_moduleConfig)
      if (g_moduleConfig.hasOwnProperty(u) && u === t) {
        var m = g_moduleConfig[u],
          g = Math.ceil(Math.random() * n) <= m.ratio;
        (d.normalAddresses = g ? m.grey.slice() : m.stable.slice()),
          m.stableBr &&
            m.greyBr &&
            (d.brAddresses = g ? m.greyBr.slice() : m.stableBr.slice()),
          "etModule" === t &&
            (i
              ? (d.normalAddresses.pop(), d.brAddresses.pop())
              : r
              ? (d.normalAddresses.pop(), d.brAddresses.pop())
              : s || l || f
              ? (d.normalAddresses.shift(), d.brAddresses.shift())
              : c()
              ? (d.normalAddresses.pop(), d.brAddresses.pop())
              : (d.normalAddresses.shift(), d.brAddresses.shift()));
        for (var h = 0; h < d.normalAddresses.length; h++) {
          var y = o ? "https://" + o + "/" : j;
          ("https://assets.alicdn.com/" === y ||
            "https://lzd-g.slatic.net/" === y ||
            "https://g.lazcdn.com/" === y ||
            "https://assets.aliexpress-media.com/" === y) &&
            (y += "g/"),
            (d.normalAddresses[h] = y + d.normalAddresses[h]),
            d.brAddresses[h] && (d.brAddresses[h] = y + d.brAddresses[h]);
        }
        return d;
      }
  }
  var u = [],
    m = "loading",
    g = "loaded",
    h = "timeout",
    y = "unexpected",
    p = "no such feature",
    b = new RegExp(
      "^([\\w\\d+.-]+:)?(?://(?:([^/?#@]*)@)?([\\w\\d\\-\\u0100-\\uffff.+%]*|\\[[^\\]]+\\])(?::([0-9]+))?)?([^?#]+)?(\\?[^#]*)?(#.*)?$"
    ),
    j = C(v());
  function A(e, t, n) {
    try {
      var o = e + "=";
      t && (o += "; domain=" + t + "; path= /;"),
        (o += "; expires=Thu, 01 Jan 1970 00:00:00 GMT"),
        n && ((o += "; SameSite=None; Secure"), (o += "; Partitioned")),
        (document.cookie = o);
    } catch (a) {}
  }
  function C(e) {
    var t = "https://g.alicdn.com/";
    if (!e) return t;
    if (
      location.href &&
      location.href.indexOf &&
      (location.href.indexOf("aliexpress.com") > -1 ||
        location.href.indexOf("aliexpress.us") > -1 ||
        location.href.indexOf("aliexpress.ru") > -1)
    )
      return "https://assets.aliexpress-media.com/";
    if (/aliexpress/.test(location.href)) return "https://aeis.alicdn.com/";
    var n = b.exec(e);
    return n ? "https://" + n[3] + (n[4] ? ":" + n[4] : "") + "/" : t;
  }
  function v() {
    for (
      var e = document.getElementsByTagName("script"), t = 0;
      t < e.length;
      t++
    ) {
      var n = e[t],
        o = n.hasAttribute ? n.src : n.getAttribute("src", 4);
      if (/\/awsc\.js/.test(o)) return o;
    }
  }
  function S(e) {
    for (var t = void 0, n = 0; n < o.length; n++) {
      for (var a = o[n], r = 0; r < a.features.length; r++)
        if (a.features[r] === e) {
          t = a;
          break;
        }
      if (t) break;
    }
    return t;
  }
  function w(e) {
    for (var t = 0; t < u.length; t++) {
      var n = u[t];
      if (n.name === e) return n;
    }
  }
  function W(e) {
    for (var t = void 0, n = 0; n < o.length; n++) {
      var a = o[n];
      if (a.name === e) {
        t = a;
        break;
      }
      if (t) break;
    }
    return t;
  }
  function _(e, n, o) {
    if (o)
      for (var a = 0; a < e.normalAddresses.length; a++) {
        var r = e.normalAddresses[a];
        t.write("<script src=" + r + "></script>");
      }
    else {
      function i(e, o, a) {
        for (var r = 0; r < e.length; r++) {
          var i = e[r],
            s = t.createElement("script");
          (s.async = !1), (s.src = i), (s.id = n);
          var l = new Date().getTime();
          s.onerror =
            o ||
            function () {
              location.href &&
                location.href.indexOf &&
                (location.href.indexOf("aliexpress.com") > -1 ||
                  location.href.indexOf("aliexpress.us") > -1 ||
                  location.href.indexOf("aliexpress.ru") > -1) &&
                M(
                  "loadjserr",
                  "loadid=" + n + "&time=" + (new Date().getTime() - l),
                  "awsc_load"
                );
            };
          var f = window.awscrprtrt || 0.001;
          Math.random() < f
            ? (s.onload =
                a ||
                function () {
                  M(
                    "loadtime",
                    "loadid=" + n + "&time=" + (new Date().getTime() - l),
                    "awsc_load"
                  );
                })
            : (s.onload = a || function () {});
          var c = t.getElementsByTagName("script")[0];
          c && c.parentNode
            ? c.parentNode.insertBefore(s, c)
            : ((c = t.body || t.head), c && c.appendChild(s));
        }
      }
      i(e.normalAddresses);
    }
  }
  function M(t, n, o) {
    var a =
        "https://acjs.aliyun.com/error?v=" +
        t +
        "&e=" +
        encodeURIComponent(n) +
        "&stack=" +
        encodeURIComponent(o) +
        "&version=1.0.0&log=awsc&href=" +
        encodeURIComponent(location.href),
      r = new Image(),
      i = "_awsc_img_" + Math.floor(1e6 * Math.random());
    (e[i] = r),
      (r.onload = r.onerror =
        function () {
          try {
            delete e[i];
          } catch (t) {
            e[i] = null;
          }
        }),
      (r.src = a);
  }
  function T(e, t) {
    Math.random() < 1e-4 &&
      M(
        "awsc_state",
        "feature=" +
          e.name +
          "&state=" +
          e.state +
          "&href=" +
          encodeURIComponent(location.href)
      );
    for (var n = void 0; (n = e.callbacks.shift()); )
      try {
        if ("function" == typeof n) n(e.state, e.exportObj);
        else if ("object" == typeof n) {
          var o = e.exportObj;
          o && "function" == typeof o.init && o.init(n);
        }
      } catch (a) {
        if (t) throw a;
        M(e.name, a.message, a.stack);
      }
  }
  function x(e, t, n, o) {
    var a = S(e);
    if (!a) return "function" == typeof t && t(p), void 0;
    var r = n && n.cdn,
      i = n && n.sync,
      s = (n && n.timeout) || 5e3;
    if (0 !== a.depends.length)
      for (var l = 0; l < a.depends.length; l++) {
        var f = a.depends[l];
        n && (delete n.sync, delete n.timeout, delete n.cdn), k(f, void 0, n);
      }
    var c = o || {};
    (c.module = a),
      (c.name = e),
      (c.state = m),
      (c.callbacks = c.callbacks || []),
      (c.options = n),
      t && c.callbacks.push(t),
      (c.timeoutTimer = setTimeout(function () {
        (c.state = h), T(c, n && n.throwExceptionInCallback);
      }, s)),
      o || u.push(c);
    var g = a.sync;
    i && (g = i);
    var y = d(a.name, r);
    _(y, "AWSC_" + a.name, g);
  }
  function k(e, t, n) {
    var o = w(e);
    if (o)
      if (o.state === h) x(e, t, n, o);
      else if (o.state === g) {
        if ("function" == typeof t) t(o.state, o.exportObj);
        else if ("object" == typeof t) {
          var a = o.exportObj;
          a && "function" == typeof a.init && a.init(t);
        }
      } else o.state === m ? t && o.callbacks.push(t) : void 0;
    else x(e, t, n);
  }
  function O(e, t, n) {
    var o = !1;
    try {
      var a = W(e);
      a || void 0, (a.moduleLoadStatus = g);
      for (var r = void 0, i = 0; i < u.length; i++) {
        var s = u[i];
        s.module === a &&
          s.name === t &&
          ((o = s.options && s.options.throwExceptionInCallback),
          (r = s),
          clearTimeout(r.timeoutTimer),
          delete r.timeoutTimer,
          (r.exportObj = n),
          s.state === m || s.state === h
            ? ((r.state = g),
              setTimeout(function () {
                T(r, o);
              }, 0))
            : void 0);
      }
      r ||
        ((r = {}),
        (r.module = a),
        (r.name = t),
        (r.state = g),
        (r.exportObj = n),
        (r.callbacks = []),
        u.push(r));
    } catch (l) {
      if (o) throw l;
      M("awsc_error", l.message, l.stack);
    }
  }
  function U(e) {
    (e.AWSCFY = function () {}),
      (e.AWSC.configFY = function (e, n, o, a) {
        return t(e, n, o, a);
      }),
      (e.AWSC.configFYSync = function (e, n) {
        return t(null, e, n);
      });
    function t(t, n, o, a) {
      var r = location.protocol + "//" + location.host + location.pathname,
        i = new e.AWSCFY();
      e._umopt_npfp = 0;
      var s = !1;
      (i.umidToken =
        "defaultToken1_um_not_loaded@@" + r + "@@" + new Date().getTime()),
        e.AWSC.use("um", function (e, t) {
          "loaded" === e
            ? ((i.umidToken =
                "defaultToken3_init_callback_not_called@@" +
                r +
                "@@" +
                new Date().getTime()),
              t.init(n, function (e, t) {
                "success" === e
                  ? (i.umidToken = t.tn)
                  : (i.umidToken =
                      "defaultToken4_init_failed with " +
                      e +
                      "@@" +
                      r +
                      "@@" +
                      new Date().getTime()),
                  (s = !0),
                  c();
              }))
            : ((i.umidToken =
                "defaultToken2_load_failed with " +
                e +
                "@@" +
                r +
                "@@" +
                new Date().getTime()),
              (s = !0),
              c());
        });
      var l = !1;
      if (
        ((i.getUA = function () {
          return (
            "defaultUA1_uab_not_loaded@@" + r + "@@" + new Date().getTime()
          );
        }),
        e.AWSC.use("uab", function (e, t) {
          (l = !0),
            "loaded" === e
              ? ((i.uabModule = t),
                (i.uabConfig = o),
                (i.getUA = function () {
                  return this.uabModule.getUA(this.uabConfig);
                }))
              : (i.getUA = function () {
                  return (
                    "defaultUA2_load_failed with " +
                    e +
                    "@@" +
                    r +
                    "@@" +
                    new Date().getTime()
                  );
                }),
            c();
        }),
        null != t)
      )
        var f = e.setTimeout(
          function () {
            c(!0);
          },
          a ? a : 2e3
        );
      function c(n) {
        null != t && ((l && s) || n) && (t(i), e.clearTimeout(f));
      }
      return null == t ? i : void 0;
    }
  }
  function D(e) {
    var t = function () {};
    (e.AWSC.configFYEx = function (e, t, o) {
      return n(e, t, o);
    }),
      (e.AWSC.configFYSyncEx = function (e) {
        return n(null, e);
      });
    function n(n, o, a) {
      var r = (
          location.protocol +
          "//" +
          location.host +
          location.pathname
        ).substr(0, 128),
        i = new t();
      if (((a = a || 5e3), "function" == typeof n))
        var s = e.setTimeout(function () {
          l();
        }, a);
      function l() {
        "function" == typeof n && (n(i), e.clearTimeout(s));
      }
      return (
        (i.getFYToken = i.getUidToken =
          function () {
            return (
              "defaultFY1_fyjs_not_loaded@@" + r + "@@" + new Date().getTime()
            );
          }),
        (e.fyglobalopt = o),
        e.AWSC.use(
          "fy",
          function (e, t) {
            "loaded" === e
              ? ((i.getFYToken = i.getUidToken =
                  function () {
                    return (
                      "defaultFY3_fyjs_not_initialized@@" +
                      r +
                      "@@" +
                      new Date().getTime()
                    );
                  }),
                (i.fyObj = t),
                t.init(o, function (e) {
                  (i.getFYToken = function (e) {
                    return (
                      "object" == typeof e &&
                      e.reqUrl &&
                      "string" == typeof e.reqUrl
                        ? (o.reqUrl = e.reqUrl)
                        : "string" == typeof e && (o.reqUrl = e),
                      this.fyObj.getFYToken(o)
                    );
                  }),
                    (i.getUidToken = function () {
                      return this.fyObj.getUidToken(o);
                    }),
                    l();
                }))
              : ((i.getFYToken = i.getUidToken =
                  function () {
                    return (
                      "defaultFY2_load_failed with " +
                      e +
                      "@@" +
                      r +
                      "@@" +
                      new Date().getTime()
                    );
                  }),
                l());
          },
          { timeout: a }
        ),
        "function" == typeof n ? void 0 : i
      );
    }
  }
  function B(e) {
    var t = g_moduleConfig.etModule,
      o = Math.ceil(Math.random() * n) <= t.ratio,
      a = o ? t.greyConfig.whitelist : t.stableConfig.whitelist,
      r = new RegExp(("^" + a.join("$|^") + "$").replace(/\*/g, ".*"));
    r.test(location.host + location.pathname) &&
      ((window.etrprtrt = 0.01), e.__etModule || e.AWSC.use("et"));
  }
  var E = window.awscrprtrt || 0.01;
  Math.random() < E && M("init", "report", "awsc");
  function F(e) {
    e.AWSC ||
      ((e.AWSC = {}),
      (e.AWSC.use = k),
      (e.AWSCInner = {}),
      (e.AWSCInner.register = O),
      U(e),
      D(e),
      B(e));
  }
  F(e);
})(window, document);
