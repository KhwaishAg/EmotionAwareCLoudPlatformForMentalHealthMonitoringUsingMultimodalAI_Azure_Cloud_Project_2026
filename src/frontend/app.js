(function () {
  "use strict";

  var qContainer = document.getElementById("q-container");
  var bContainer = document.getElementById("b-container");
  var submit = document.getElementById("submit");
  var consent = document.getElementById("consent");
  var errorBox = document.getElementById("error");
  var loading = document.getElementById("loading");
  var fileInput = document.getElementById("voice-file");
  var recStart = document.getElementById("rec-start");
  var recStop = document.getElementById("rec-stop");
  var recClear = document.getElementById("rec-clear");
  var recStatus = document.getElementById("rec-status");
  var schema = null;
  var recordedBase64 = null; // voice kept only in memory, never persisted

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
    module_unavailable: "Unavailable"
  };
  var ADVICE = {
    low: ["Keep the routines that are working: regular sleep, movement and breaks.",
          "Check in with yourself again if your workload or deadlines change."],
    moderate: ["Plan short breaks and protect your sleep, especially around deadlines.",
               "Break big tasks into small steps and talk with a friend, mentor or classmate.",
               "Campus wellness resources can help if this feeling continues."],
    high: ["Consider talking to someone you trust or your campus counselling service soon.",
           "Prioritise sleep, food and short breaks, and ask for help with workload where possible.",
           "If you feel unsafe or in crisis, contact local emergency services or a crisis helpline right away."]
  };

  function el(tag, props, children) {
    var node = document.createElement(tag);
    Object.keys(props || {}).forEach(function (k) { node[k] = props[k]; });
    (children || []).forEach(function (c) { node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function showError(message) { errorBox.textContent = message; errorBox.hidden = !message; }
  function updateSubmit() { submit.disabled = !consent.checked || !schema; }
  consent.addEventListener("change", updateSubmit);

  /* ---------- form construction ---------- */
  function buildQuestionnaire(items) {
    qContainer.textContent = "";
    items.forEach(function (item) {
      var group = el("fieldset", {}, [el("legend", {}, [item.id + ". " + item.text])]);
      if (item.type === "multi") {
        item.options.forEach(function (option) {
          group.appendChild(el("label", { className: "opt" },
            [el("input", { type: "checkbox", value: option, name: item.id }), " " + option]));
        });
      } else {
        var select = el("select", { name: item.id }, [el("option", { value: "" }, ["Select an answer"])]);
        item.options.forEach(function (option) { select.appendChild(el("option", { value: option }, [option])); });
        group.appendChild(select);
      }
      qContainer.appendChild(group);
    });
  }
  function buildBehavioral(fields) {
    bContainer.textContent = "";
    Object.keys(fields).forEach(function (name) {
      var range = fields[name];
      bContainer.appendChild(el("fieldset", {}, [
        el("label", { className: "q" }, [BEHAVIOR_LABELS[name] || name]),
        el("input", { type: "number", name: name, min: range.min, max: range.max, step: "0.5" }),
        el("span", { className: "muted" }, ["Allowed range: " + range.min + " to " + range.max])
      ]));
    });
  }

  fetch("/schema").then(function (r) { if (!r.ok) { throw new Error("schema"); } return r.json(); })
    .then(function (data) {
      schema = data; buildQuestionnaire(data.questionnaire_items); buildBehavioral(data.behavioral_fields); updateSubmit();
    })
    .catch(function () { qContainer.textContent = "Could not load the questionnaire. Is the assessment service running?"; });

  function collectQuestionnaire() {
    var responses = {}, missing = [];
    schema.questionnaire_items.forEach(function (item) {
      if (item.type === "multi") {
        var picked = Array.prototype.map.call(
          qContainer.querySelectorAll('input[name="' + item.id + '"]:checked'), function (n) { return n.value; });
        if (!picked.length) { missing.push(item.id); }
        responses[item.id] = picked.join("; ");
      } else {
        var value = qContainer.querySelector('select[name="' + item.id + '"]').value;
        if (!value) { missing.push(item.id); }
        responses[item.id] = value;
      }
    });
    if (missing.length) { throw new Error("Please answer every questionnaire item (missing: " + missing.join(", ") + ") or untick 'Include questionnaire'."); }
    return { responses: responses };
  }
  function collectBehavioral() {
    var responses = {};
    Object.keys(schema.behavioral_fields).forEach(function (name) {
      var range = schema.behavioral_fields[name];
      var raw = bContainer.querySelector('input[name="' + name + '"]').value;
      if (raw === "") { throw new Error("Please fill every daily-habits field or untick 'Include daily habits'."); }
      var value = Number(raw);
      if (!isFinite(value) || value < range.min || value > range.max) {
        throw new Error((BEHAVIOR_LABELS[name] || name) + " must be between " + range.min + " and " + range.max + ".");
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
  function setVoiceState(text, hasVoice) { recStatus.textContent = text; recStatus.className = "muted"; recClear.hidden = !hasVoice; }

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
      timer = setInterval(function () {
        recStatus.className = "recording"; recStatus.textContent = "Recording… " + Math.round((Date.now() - started) / 1000) + " s";
      }, 250);
    }).catch(function () { showError("Microphone access was denied or unavailable. You can upload a WAV file instead."); });
  });
  recStop.addEventListener("click", function () { if (recorder) { recorder.stop(); } });
  recClear.addEventListener("click", function () { recordedBase64 = null; fileInput.value = ""; setVoiceState("", false); });
  fileInput.addEventListener("change", function () { if (fileInput.files[0]) { recordedBase64 = null; setVoiceState("File selected.", true); } });

  function readVoice() {
    if (recordedBase64) { return Promise.resolve(recordedBase64); }
    var file = fileInput.files[0];
    if (!file) { return Promise.resolve(null); }
    return new Promise(function (resolve, reject) {
      if (file.size > schema.max_audio_bytes) { reject(new Error("Audio file is too large.")); return; }
      var reader = new FileReader();
      reader.onload = function () { resolve(String(reader.result).split(",")[1] || ""); };
      reader.onerror = function () { reject(new Error("Could not read the audio file.")); };
      reader.readAsDataURL(file);
    });
  }

  /* ---------- result rendering (server text goes in via textContent only) ---------- */
  function render(data) {
    var a = data.assessment, level = document.getElementById("r-level");
    var advice = document.getElementById("r-advice"); advice.textContent = "";
    if (a.status !== "available") {
      level.textContent = "No result"; level.className = "badge";
      document.getElementById("r-summary").textContent = "None of the provided inputs could be analysed.";
      document.getElementById("r-bar").style.width = "0%";
      document.getElementById("r-conf").textContent = "See the source cards below for why.";
    } else {
      level.textContent = a.risk_level; level.className = "badge " + a.risk_level;
      document.getElementById("r-summary").textContent = "Estimated stress/risk level (indicator only).";
      document.getElementById("r-bar").style.width = pct(a.risk_score);
      document.getElementById("r-conf").textContent = "Confidence: " + pct(a.confidence) + " · Agreement between sources: " +
        a.agreement.level + " · Data completeness: " + pct(a.data_quality.completeness) +
        (a.modalities_used.length === 1 ? " · based on a single source" : "");
      (ADVICE[a.risk_level] || []).forEach(function (t) { advice.appendChild(el("li", {}, [t])); });
    }
    var cards = document.getElementById("r-cards"); cards.textContent = "";
    ["questionnaire", "behavioral", "voice"].forEach(function (name) {
      var detail = data.modality_details[name], card = el("div", { className: "mini" }, [el("h3", {}, [name])]);
      if (detail) {
        card.appendChild(el("span", { className: "badge " + detail.risk_level }, [detail.risk_level]));
        card.appendChild(el("p", { className: "muted" }, ["Score " + pct(detail.risk_score) + " · confidence " +
          pct(detail.confidence) + " · completeness " + pct(detail.completeness)]));
      } else {
        card.appendChild(el("p", { className: "muted" }, [STATUS_TEXT[data.modality_status[name]] || data.modality_status[name]]));
      }
      cards.appendChild(card);
    });
    var emotionBox = document.getElementById("r-emotion"), emotionBody = document.getElementById("r-emotion-body");
    var ve = data.voice_emotion; emotionBox.hidden = !ve; emotionBody.textContent = "";
    if (ve) {
      if (ve.status === "available" && ve.result) {
        var ranked = Object.keys(ve.result.probabilities).sort(function (x, y) { return ve.result.probabilities[y] - ve.result.probabilities[x]; });
        emotionBody.textContent = "Most likely acted-emotion category: " + ve.result.predicted_emotion + " (" + pct(ve.result.confidence) +
          "). Next: " + ranked.slice(1, 3).map(function (k) { return k + " " + pct(ve.result.probabilities[k]); }).join(", ") + ".";
      } else {
        emotionBody.textContent = STATUS_TEXT[ve.status] ? "Not available: " + STATUS_TEXT[ve.status].replace(/^Unavailable \(|\)$/g, "") : "Not available.";
      }
    }
    var tbody = document.getElementById("r-factors"); tbody.textContent = "";
    a.key_factors.forEach(function (f) {
      tbody.appendChild(el("tr", {}, [el("td", {}, [f.feature]), el("td", {}, [f.modality]),
        el("td", {}, [f.direction.replace(/_/g, " ")]), el("td", {}, [pct(f.contribution)])]));
    });
    document.getElementById("r-disclaimer").textContent = data.disclaimer;
    var result = document.getElementById("result"); result.hidden = false; result.scrollIntoView({ behavior: "smooth" });
  }

  /* ---------- submit ---------- */
  document.getElementById("form").addEventListener("submit", function (event) {
    event.preventDefault(); showError("");
    if (!consent.checked) { showError("Please give consent to continue."); return; }
    var body = {};
    try {
      if (document.getElementById("use-q").checked) { body.questionnaire = collectQuestionnaire(); }
      if (document.getElementById("use-b").checked) { body.behavioral = collectBehavioral(); }
    } catch (e) { showError(e.message); return; }
    submit.disabled = true; loading.hidden = false;
    readVoice().then(function (audio) {
      if (audio) { body.voice = { audio_base64: audio }; }
      if (!body.questionnaire && !body.behavioral && !body.voice) { throw new Error("Please include at least one input."); }
      return fetch("/assessment", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    }).then(function (r) {
      if (r.status === 422) { throw new Error("Some answers were not valid. Please check the form and try again."); }
      if (!r.ok) { throw new Error("The service could not complete the assessment. Please try again later."); }
      return r.json();
    }).then(render).catch(function (e) { showError(e.message || "Something went wrong."); })
      .then(function () { loading.hidden = true; updateSubmit(); });
  });

  /* ---------- evaluation transparency ---------- */
  fetch("/evaluation").then(function (r) { return r.json(); }).then(function (d) {
    var body = document.getElementById("eval-body"); body.textContent = "";
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
  }).catch(function () { document.getElementById("eval-body").textContent = "Evaluation summary unavailable."; });
})();
