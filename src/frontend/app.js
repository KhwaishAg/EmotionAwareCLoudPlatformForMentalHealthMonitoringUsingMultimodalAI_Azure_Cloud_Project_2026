(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var qContainer = $("q-container"), bContainer = $("b-container");
  var submit = $("submit"), next = $("next"), prev = $("prev"), consent = $("consent");
  var errorBox = $("error"), loading = $("loading"), fileInput = $("voice-file");
  var recStart = $("rec-start"), recStop = $("rec-stop"), recClear = $("rec-clear"), recStatus = $("rec-status");
  var schema = null;
  var cloud = {};            // which optional Azure features this deployment has switched on (from /status)
  var current = 1, TOTAL_STEPS = 4;
  var recordedBase64 = null; // voice kept only in memory, never persisted
  var IDLE_VOICE_TEXT = "No recording yet.";

  var BEHAVIOR_LABELS = {
    B1_sleep_hours: "Average hours of sleep per night",
    B2_study_consistency: "How consistent is your daily study routine? (1 = not at all, 5 = very)",
    B3_social_activity_freq: "How often do you socialise? (1 = rarely, 5 = very often)",
    B4_screen_time_hours: "Non-academic screen time per day (hours)",
    B5_routine_irregularity: "How irregular is your daily routine? (1 = regular, 5 = very irregular)"
  };
  var STATUS_TEXT = {
    available: "Used", not_provided: "Not provided",
    model_not_trained: "Unavailable (no trained voice model yet)",
    model_unavailable: "Unavailable (model files missing or unusable)",
    invalid_input: "Could not be used (input problem)", prediction_failed: "Unavailable (analysis failed)",
    module_unavailable: "Unavailable",
    no_speech_detected: "Could not be used (not enough clear speech was recognised)",
    speech_service_unavailable: "Unavailable (Azure speech analysis could not be reached)"
  };
  var SOURCE_NAMES = { questionnaire: "Questionnaire", behavioral: "Daily habits", voice: "Voice" };
  var EFFECT_TEXT = { increases_risk: "tends to raise the estimate", decreases_risk: "tends to lower the estimate", model_associated: "influences the estimate" };
  var LEVEL_TITLE = { low: "Low stress indicators", moderate: "Moderate stress indicators", high: "High stress indicators" };
  var CONSISTENCY_TEXT = {
    consistent: "Your sources point the same way.",
    moderate_difference: "Your sources differ somewhat.",
    conflict: "Your sources point in clearly different directions."
  };
  var PATTERN_TEXT = {
    masked_stress_possible: "Possible masked stress: your own answers give a lower result than your daily habits do. This is only a prompt to reflect (and the daily-habits model was trained on synthetic data), not a finding.",
    self_report_higher: "You describe more stress than your daily habits show. Your own experience still counts."
  };

  function el(tag, props, children) {
    var node = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) { node[k] = props[k]; });
    (children || []).forEach(function (c) { node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function showError(message) { errorBox.textContent = message; errorBox.hidden = !message; }
  function failAt(step, message) { var e = new Error(message); e.step = step; return e; }
  // The questionnaire's own answer labels contain a long dash ("1 - Not at all"); show them with a colon.
  // Only the displayed text changes: the value sent to the model is untouched.
  function plainOption(text) { return String(text).split(" " + String.fromCharCode(8212) + " ").join(": "); }

  function shorten(text, n) { return text.length > n ? text.slice(0, n - 1) + "\u2026" : text; }
  function storage() { try { return window.localStorage; } catch (e) { return null; } }
  function anonId() {
    var store = storage(), id = null;
    try { id = store && store.getItem("checkin-anon-id"); } catch (e) { id = null; }
    if (!id) {
      id = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : null;
      try { if (id && store) { store.setItem("checkin-anon-id", id); } } catch (e) { /* private mode: the ID lasts this page only */ }
    }
    return id;
  }

  /* ---------- theme (light / dark), remembered in this browser only ---------- */
  var root = document.documentElement, themeButton = $("theme");
  function isDark() {
    var set = root.getAttribute("data-theme");
    return set ? set === "dark" : !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function syncThemeLabel() { themeButton.textContent = isDark() ? "Light theme" : "Dark theme"; }
  try { var saved = localStorage.getItem("theme"); if (saved === "light" || saved === "dark") { root.setAttribute("data-theme", saved); } } catch (e) { /* storage may be blocked */ }
  syncThemeLabel();
  themeButton.addEventListener("click", function () {
    var nextTheme = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", nextTheme); syncThemeLabel();
    try { localStorage.setItem("theme", nextTheme); } catch (e) { /* ignore */ }
  });

  /* ---------- step-by-step flow ---------- */
  var stepPanels = document.querySelectorAll(".step"), stepperItems = document.querySelectorAll("#stepper li");
  function updateButtons() {
    next.disabled = !(consent.checked && schema);
    submit.disabled = !(consent.checked && schema);
  }
  function goTo(n, scroll) {
    current = Math.max(1, Math.min(TOTAL_STEPS, n));
    Array.prototype.forEach.call(stepPanels, function (panel) { panel.classList.toggle("active", Number(panel.dataset.step) === current); });
    Array.prototype.forEach.call(stepperItems, function (item) {
      var k = Number(item.dataset.step);
      item.classList.toggle("active", k === current); item.classList.toggle("done", k < current);
    });
    prev.hidden = current === 1; next.hidden = current === TOTAL_STEPS; submit.hidden = current !== TOTAL_STEPS;
    showError("");
    if (scroll) { $("assess").scrollIntoView({ block: "start" }); }
  }
  goTo(1, false);
  consent.addEventListener("change", updateButtons);
  next.addEventListener("click", function () {
    if (current === 1 && !consent.checked) { showError("Please give consent to continue."); return; }
    goTo(current + 1, true);
  });
  prev.addEventListener("click", function () { goTo(current - 1, true); });
  Array.prototype.forEach.call(stepperItems, function (item) {
    item.addEventListener("click", function () { if (item.classList.contains("done")) { goTo(Number(item.dataset.step), true); } });
  });
  $("use-q").addEventListener("change", function () { $("q-wrap").classList.toggle("off", !this.checked); });
  $("use-b").addEventListener("change", function () { bContainer.classList.toggle("off", !this.checked); });

  /* ---------- form construction ---------- */
  function refreshAnswered() {
    if (!schema) { return; }
    var done = schema.questionnaire_items.filter(function (item) {
      return qContainer.querySelector('input[name="' + item.id + '"]:checked');
    }).length, total = schema.questionnaire_items.length;
    $("q-count").textContent = done + " of " + total + " answered";
    $("q-progress").style.width = (done / total * 100) + "%";
  }
  function buildQuestionnaire(items) {
    qContainer.textContent = ""; qContainer.classList.remove("muted");
    items.forEach(function (item) {
      var group = el("fieldset", { className: "q-card", id: "card-" + item.id }, [
        el("legend", {}, [el("span", { className: "q-num" }, [item.id + "."]), item.text])
      ]);
      var options = el("div", { className: "options" });
      item.options.forEach(function (option) {
        options.appendChild(el("label", { className: "option" }, [
          el("input", { type: item.type === "multi" ? "checkbox" : "radio", value: option, name: item.id }),
          el("span", {}, [plainOption(option)])
        ]));
      });
      group.appendChild(options);
      if (item.type === "multi") { group.appendChild(el("p", { className: "hint" }, ["Select all that apply."])); }
      group.addEventListener("change", function () { group.classList.remove("missing"); refreshAnswered(); });
      qContainer.appendChild(group);
    });
    refreshAnswered();
  }
  function buildBehavioral(fields) {
    bContainer.textContent = "";
    Object.keys(fields).forEach(function (name) {
      var range = fields[name], inputId = "habit-" + name;
      bContainer.appendChild(el("div", { className: "habit" }, [
        el("label", { htmlFor: inputId }, [BEHAVIOR_LABELS[name] || name]),
        el("input", { id: inputId, type: "number", name: name, min: range.min, max: range.max, step: "0.5", inputMode: "decimal", placeholder: range.min + " to " + range.max }),
        el("p", { className: "hint" }, ["Allowed range: " + range.min + " to " + range.max])
      ]));
    });
  }

  fetch("/schema").then(function (r) { if (!r.ok) { throw new Error("schema"); } return r.json(); })
    .then(function (data) {
      schema = data; buildQuestionnaire(data.questionnaire_items); buildBehavioral(data.behavioral_fields); updateButtons();
    })
    .catch(function () { qContainer.textContent = "Could not load the questionnaire. Is the assessment service running?"; });

  function collectQuestionnaire() {
    var responses = {}, missing = [];
    schema.questionnaire_items.forEach(function (item) {
      var picked = Array.prototype.map.call(
        qContainer.querySelectorAll('input[name="' + item.id + '"]:checked'), function (n) { return n.value; });
      if (!picked.length) { missing.push(item.id); $("card-" + item.id).classList.add("missing"); }
      responses[item.id] = item.type === "multi" ? picked.join("; ") : (picked[0] || "");
    });
    if (missing.length) {
      throw failAt(2, "Please answer every questionnaire item (missing: " + missing.join(", ") + "), or switch the questionnaire off.");
    }
    return { responses: responses };
  }
  function collectBehavioral() {
    var responses = {};
    Object.keys(schema.behavioral_fields).forEach(function (name) {
      var range = schema.behavioral_fields[name];
      var raw = bContainer.querySelector('input[name="' + name + '"]').value;
      if (raw === "") { throw failAt(3, "Please fill every daily-habits field, or switch daily habits off."); }
      var value = Number(raw);
      if (!isFinite(value) || value < range.min || value > range.max) {
        throw failAt(3, (BEHAVIOR_LABELS[name] || name) + " must be between " + range.min + " and " + range.max + ".");
      }
      responses[name] = value;
    });
    return { responses: responses };
  }

  /* ---------- voice: record in the browser and encode 16 kHz mono WAV ---------- */
  var recorder = null;

  function encodeWav(chunks, inputRate) {
    var length = chunks.reduce(function (n, c) { return n + c.length; }, 0);
    var merged = new Float32Array(length), offset = 0;
    chunks.forEach(function (c) { merged.set(c, offset); offset += c.length; });
    var targetRate = 16000, ratio = inputRate / targetRate, outLength = Math.floor(merged.length / ratio);
    var pcm = new Int16Array(outLength);
    for (var i = 0; i < outLength; i++) {
      var start = Math.floor(i * ratio), end = Math.max(start + 1, Math.floor((i + 1) * ratio)), sum = 0;
      for (var j = start; j < end && j < merged.length; j++) { sum += merged[j]; }
      var sample = Math.max(-1, Math.min(1, sum / (end - start)));
      pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    var buffer = new ArrayBuffer(44 + pcm.length * 2), view = new DataView(buffer);
    function text(pos, s) { for (var k = 0; k < s.length; k++) { view.setUint8(pos + k, s.charCodeAt(k)); } }
    text(0, "RIFF"); view.setUint32(4, 36 + pcm.length * 2, true); text(8, "WAVE"); text(12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true); view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, pcm.length * 2, true);
    for (var n = 0; n < pcm.length; n++) { view.setInt16(44 + n * 2, pcm[n], true); }
    return { bytes: new Uint8Array(buffer), seconds: outLength / targetRate };
  }
  function toBase64(bytes) {
    var binary = "", step = 0x8000;
    for (var i = 0; i < bytes.length; i += step) { binary += String.fromCharCode.apply(null, bytes.subarray(i, i + step)); }
    return btoa(binary);
  }
  function setVoiceState(text, hasVoice) { recStatus.textContent = text; recStatus.className = "muted small"; recClear.hidden = !hasVoice; }
  function showRecording(seconds) {
    recStatus.className = "recording small"; recStatus.textContent = "";
    recStatus.appendChild(el("span", { className: "rec-dot" })); recStatus.appendChild(document.createTextNode("Recording, " + seconds + " s"));
  }

  recStart.addEventListener("click", function () {
    showError("");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Recording is not supported in this browser; please upload a WAV file instead."); return;
    }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      var Context = window.AudioContext || window.webkitAudioContext, context = new Context();
      var source = context.createMediaStreamSource(stream), processor = context.createScriptProcessor(4096, 1, 1);
      var chunks = [], started = Date.now(), timer;
      processor.onaudioprocess = function (e) { chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      source.connect(processor); processor.connect(context.destination);
      recorder = {
        stop: function () {
          clearInterval(timer); processor.disconnect(); source.disconnect();
          stream.getTracks().forEach(function (t) { t.stop(); });
          var rate = context.sampleRate; context.close();
          var wav = encodeWav(chunks, rate); chunks = [];
          recorder = null; recStart.hidden = false; recStop.hidden = true;
          if (wav.seconds < 5) { recordedBase64 = null; setVoiceState("Recording too short (" + wav.seconds.toFixed(1) + " s). Please record at least 5 seconds.", false); return; }
          recordedBase64 = toBase64(wav.bytes); fileInput.value = "";
          setVoiceState("Recorded " + wav.seconds.toFixed(0) + " s (kept in memory only).", true);
        }
      };
      fileInput.value = ""; recordedBase64 = null; recStart.hidden = true; recStop.hidden = false; recClear.hidden = true;
      showRecording(0);
      timer = setInterval(function () { showRecording(Math.round((Date.now() - started) / 1000)); }, 500);
    }).catch(function () { showError("Microphone access was denied or unavailable. You can upload a WAV file instead."); });
  });
  recStop.addEventListener("click", function () { if (recorder) { recorder.stop(); } });
  recClear.addEventListener("click", function () { recordedBase64 = null; fileInput.value = ""; setVoiceState(IDLE_VOICE_TEXT, false); });
  fileInput.addEventListener("change", function () { if (fileInput.files[0]) { recordedBase64 = null; setVoiceState("File selected.", true); } });

  function readVoice() {
    if (recordedBase64) { return Promise.resolve(recordedBase64); }
    var file = fileInput.files[0];
    if (!file) { return Promise.resolve(null); }
    return new Promise(function (resolve, reject) {
      if (file.size > schema.max_audio_bytes) { reject(failAt(4, "Audio file is too large.")); return; }
      var reader = new FileReader();
      reader.onload = function () { resolve(String(reader.result).split(",")[1] || ""); };
      reader.onerror = function () { reject(failAt(4, "Could not read the audio file.")); };
      reader.readAsDataURL(file);
    });
  }

  /* ---------- result rendering (server text goes in via textContent only) ---------- */
  function sourceCard(name, data) {
    var detail = data.modality_details[name];
    var card = el("div", { className: "source" + (detail ? "" : " off") });
    var top = el("div", { className: "source-top" }, [el("h4", {}, [SOURCE_NAMES[name]])]);
    card.appendChild(top);
    if (detail) {
      top.appendChild(el("span", { className: "tag " + detail.risk_level }, [detail.risk_level]));
      var bar = el("div", { className: "bar " + detail.risk_level }, [el("span")]);
      bar.firstChild.style.width = pct(detail.risk_score);
      card.appendChild(bar);
      card.appendChild(el("p", { className: "muted small" }, ["Score " + detail.risk_score.toFixed(2) + ", confidence " + pct(detail.confidence) + ", completeness " + pct(detail.completeness)]));
      if (detail.source === "azure_speech_language") {
        card.appendChild(el("p", { className: "muted small" }, ["From the tone of what you said (" + detail.sentiment + ")."]));
        if (detail.transcript) { card.appendChild(el("p", { className: "quote" }, ["\u201c" + shorten(detail.transcript, 220) + "\u201d"])); }
        if (detail.key_phrases && detail.key_phrases.length) { card.appendChild(el("p", { className: "muted small" }, ["Key phrases: " + detail.key_phrases.join(", ")])); }
      }
    } else {
      card.appendChild(el("p", { className: "muted small" }, [STATUS_TEXT[data.modality_status[name]] || data.modality_status[name]]));
    }
    return card;
  }

  function render(data) {
    var a = data.assessment, available = a.status === "available";
    var result = $("result"), summary = $("r-summary-card"), level = $("r-level");
    result.hidden = false;

    // headline: level, project-defined score, position on a three-band scale
    summary.className = "card summary" + (available ? " " + a.risk_level : "");
    if (available) {
      $("r-title").textContent = LEVEL_TITLE[a.risk_level];
      level.textContent = a.risk_level; level.className = "tag " + a.risk_level;
      $("r-score-text").textContent = "Score " + a.risk_score.toFixed(2) + " on a 0 to 1 scale. This is a project-defined index, not a probability.";
      $("r-scale").hidden = false; $("r-marker").style.left = pct(a.risk_score);
      $("r-confidence").textContent = pct(a.confidence);
      $("r-agreement").textContent = a.agreement.level === "not_applicable" ? "not applicable (one source)" : a.agreement.level;
      $("r-completeness").textContent = pct(a.data_quality.completeness);
    } else {
      $("r-title").textContent = "No result";
      level.textContent = ""; level.className = "tag";
      $("r-score-text").textContent = "None of the provided inputs could be analysed. The source cards below say why.";
      $("r-scale").hidden = true;
      $("r-confidence").textContent = "n/a"; $("r-agreement").textContent = "n/a"; $("r-completeness").textContent = "n/a";
    }

    // per-source cards
    var cards = $("r-cards"); cards.textContent = "";
    ["questionnaire", "behavioral", "voice"].forEach(function (name) { cards.appendChild(sourceCard(name, data)); });

    // cross-source consistency
    var cbox = $("r-consistency"), cbody = $("r-consistency-body"), cons = a.consistency;
    cbox.hidden = !available; cbody.textContent = "";
    cbox.classList.toggle("flag", !!(cons && cons.applicable && cons.pattern !== "none"));
    if (cons && cons.applicable) {
      cbody.textContent = (CONSISTENCY_TEXT[cons.status] || "") + " " + (PATTERN_TEXT[cons.pattern] || "");
    } else if (cons) {
      cbody.textContent = "Only one source was used, so there is nothing to compare. Adding another source (for example the questionnaire plus daily habits) lets the app check whether they agree, including a possible masked-stress pattern.";
    }

    // separate, experimental vocal emotion
    var emotionBox = $("r-emotion"), emotionBody = $("r-emotion-body"), ve = data.voice_emotion;
    emotionBox.hidden = !ve; emotionBody.textContent = "";
    if (ve) {
      if (ve.status === "available" && ve.result) {
        var ranked = Object.keys(ve.result.probabilities).sort(function (x, y) { return ve.result.probabilities[y] - ve.result.probabilities[x]; });
        emotionBody.textContent = "Most likely acted-emotion category: " + ve.result.predicted_emotion + " (" + pct(ve.result.confidence) +
          "). Next: " + ranked.slice(1, 3).map(function (k) { return k + " " + pct(ve.result.probabilities[k]); }).join(", ") + ".";
      } else {
        emotionBody.textContent = STATUS_TEXT[ve.status] ? "Not available: " + STATUS_TEXT[ve.status].replace(/^Unavailable \(|\)$/g, "") : "Not available.";
      }
    }

    // personal insights
    var insightList = $("r-insights"); insightList.textContent = "";
    (data.insights || []).forEach(function (item) { insightList.appendChild(el("li", {}, [item.evidence.join(" ")])); });
    $("r-insights-empty").hidden = !available || (data.insights || []).length > 0;

    // personalized recommendations (produced on the server from the person's own answers)
    var advice = $("r-advice"); advice.textContent = "";
    (data.recommendations || []).forEach(function (rec) {
      var box = el("div", { className: "rec" + (rec.theme === "support" ? " support" : "") + (/^(masked_stress|self_report_higher)$/.test(rec.theme) ? " priority" : "") }, [
        el("h4", {}, [rec.title]), el("p", {}, [rec.advice])
      ]);
      if (rec.tips && rec.tips.length) {
        var tipList = el("ul", { className: "tips" });
        rec.tips.forEach(function (tip) { tipList.appendChild(el("li", {}, [tip])); });
        box.appendChild(tipList);
      }
      if (rec.because && rec.because.length) { box.appendChild(el("p", { className: "why" }, ["Based on: " + rec.because.join(" ")])); }
      advice.appendChild(box);
    });

    // model-wide factors
    var tbody = $("r-factors"); tbody.textContent = "";
    a.key_factors.forEach(function (f) {
      var from = f.modality.split(",").map(function (m) { return SOURCE_NAMES[m] || m; }).join(" + ");
      tbody.appendChild(el("tr", {}, [el("td", {}, [f.label || f.feature]), el("td", {}, [from]),
        el("td", {}, [EFFECT_TEXT[f.direction] || "influences the estimate"]), el("td", {}, [pct(f.contribution)])]));
    });
    $("r-disclaimer").textContent = data.disclaimer;
    var saved = { queued: "Summary sent to your history.", saved: "Summary saved to your history.",
                  failed: "Your history could not be updated this time.", unavailable: "History is not enabled here." }[data.history_status];
    if (saved) { $("h-status").textContent = saved; loadHistory(1500); }
    result.scrollIntoView({ block: "start" });
  }

  /* ---------- submit ---------- */
  $("form").addEventListener("submit", function (event) {
    event.preventDefault(); showError("");
    if (!consent.checked) { goTo(1, true); showError("Please give consent to continue."); return; }
    var body = {};
    try {
      if ($("use-q").checked) { body.questionnaire = collectQuestionnaire(); }
      if ($("use-b").checked) { body.behavioral = collectBehavioral(); }
    } catch (e) { goTo(e.step || current, true); showError(e.message); return; }
    submit.disabled = true; loading.hidden = false;
    readVoice().then(function (audio) {
      if (audio) {
        body.voice = { audio_base64: audio };
        if (cloud.speech_content && $("use-cloud-speech").checked) { body.voice.cloud_speech = true; }
      }
      if (cloud.history && $("save-history").checked) { body.history = { anon_id: anonId() }; }
      if (!body.questionnaire && !body.behavioral && !body.voice) { throw failAt(2, "Please include at least one input."); }
      return fetch("/assessment", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    }).then(function (r) {
      if (r.status === 422) { throw new Error("Some answers were not valid. Please check the form and try again."); }
      if (!r.ok) { throw new Error("The service could not complete the assessment. Please try again later."); }
      return r.json();
    }).then(render).catch(function (e) { if (e.step) { goTo(e.step, true); } showError(e.message || "Something went wrong."); })
      .then(function () { loading.hidden = true; updateButtons(); });
  });

  /* ---------- optional Azure features + history ---------- */
  var historyCard = $("history-card");
  function historyQuery() { var id = anonId(); return id ? "?anon_id=" + encodeURIComponent(id) : ""; }
  function renderHistory(d) {
    var rows = $("h-rows"); rows.textContent = "";
    d.items.forEach(function (item) {
      rows.appendChild(el("tr", {}, [el("td", {}, [new Date(item.created_at).toLocaleString()]),
        el("td", {}, [el("span", { className: "tag " + (item.risk_level || "") }, [item.risk_level || "n/a"])]),
        el("td", {}, [item.risk_score == null ? "n/a" : Number(item.risk_score).toFixed(2)]), el("td", {}, [(item.modalities_used || []).join(", ")])]));
    });
    var t = d.trend, text = {
      not_enough_data: d.items.length ? "Save a few more check-ins to see a trend." : "No saved check-ins yet.",
      rising: "Your latest estimate is higher than your recent average (" + Number(t.latest || 0).toFixed(2) + " vs " + Number(t.previous_average || 0).toFixed(2) + ").",
      falling: "Your latest estimate is lower than your recent average (" + Number(t.latest || 0).toFixed(2) + " vs " + Number(t.previous_average || 0).toFixed(2) + ").",
      stable: "Your estimates have been fairly stable recently."
    }[t.direction];
    $("h-trend").textContent = text || "";
  }
  function loadHistory(delay) {
    if (!cloud.history) { return; }
    setTimeout(function () {
      fetch("/history" + historyQuery()).then(function (r) { if (!r.ok) { throw new Error(); } return r.json(); })
        .then(renderHistory).catch(function () { $("h-trend").textContent = "History is unavailable right now."; });
    }, delay || 0);
  }
  var deleteArmed = false, deleteButton = $("h-delete");
  deleteButton.addEventListener("click", function () {
    if (!deleteArmed) { deleteArmed = true; deleteButton.textContent = "Click again to delete everything"; return; }
    deleteArmed = false; deleteButton.textContent = "Delete my history";
    fetch("/history" + historyQuery(), { method: "DELETE" }).then(function (r) { if (!r.ok) { throw new Error(); } return r.json(); })
      .then(function (d) { $("h-status").textContent = "Deleted " + d.deleted + " saved check-in(s)."; loadHistory(); })
      .catch(function () { $("h-status").textContent = "Could not delete right now; please try again."; });
  });
  $("h-refresh").addEventListener("click", function () { loadHistory(); });

  fetch("/status").then(function (r) { return r.json(); }).then(function (d) {
    cloud = d.cloud || {};
    if (cloud.hosted) {
      $("privacy-where").firstChild.textContent = "Your answers are sent to this app's assessment service on Microsoft Azure App Service, used to compute the result, and are ";
    }
    if (cloud.speech_content && !(d.models && d.models.voice)) {
      $("cloud-speech-opt").hidden = false; $("cloud-speech-note").hidden = false;
      $("voice-help").textContent = "No acoustic stress model is trained yet; tick the option below to use the tone of what you say instead.";
    }
    if (cloud.history) {
      $("history-opt").hidden = false; $("history-note").hidden = false; historyCard.hidden = false;
      $("privacy-tail").textContent = " unless you choose to save a summary below. Results are not cached by the browser.";
      $("h-id-note").textContent = cloud.signed_in_history
        ? "History is linked to your signed-in account." : "History is linked to a random ID stored in this browser only.";
      loadHistory();
    }
  }).catch(function () { cloud = {}; });

  /* ---------- evaluation transparency ---------- */
  fetch("/evaluation").then(function (r) { return r.json(); }).then(function (d) {
    var body = $("eval-body"); body.textContent = "";
    var q = d.questionnaire, b = d.behavioral, v = d.voice;
    [["Questionnaire", q.model + " on " + q.respondents + " real responses; cross-validated macro-F1 " +
        (q.cross_validated_macro_f1 == null ? "n/a" : q.cross_validated_macro_f1.toFixed(2)) + ". " + q.caveat],
     ["Daily habits", "Trained on " + b.data + " data (test accuracy " + (b.test_accuracy == null ? "n/a" : b.test_accuracy.toFixed(2)) + "). " + b.caveat],
     ["Vocal emotion (separate)", d.voice_emotion.trained ? "Acted-emotion model on " + d.voice_emotion.dataset + ": held-out accuracy " +
        d.voice_emotion.held_out_accuracy.toFixed(2) + " vs chance " + d.voice_emotion.chance_accuracy.toFixed(2) + " (" + d.voice_emotion.held_out_speakers +
        " unseen speakers). " + d.voice_emotion.caveat : d.voice_emotion.caveat],
     ["Voice stress risk", v.trained ? "Trained model present (" + v.dataset_kind + "). " + (v.note || "") : v.caveat]
    ].forEach(function (row) { body.appendChild(el("p", {}, [el("strong", {}, [row[0] + ": "]), row[1]])); });
    body.appendChild(el("p", {}, [d.disclaimer]));
  }).catch(function () { $("eval-body").textContent = "Evaluation summary unavailable."; });
})();
