"use strict";
/* Transcription: NVIDIA Nemotron Transcribe, served by llama.cpp on its own port (30036), driven
   through the cockpit's relay (POST /api/transcribe, cockpit.py transcribe_call): the browser sends
   the audio file as a base64 data URL with the language, and the cockpit relays it to
   /v1/audio/transcriptions with the box's key, which never reaches this page. The lane reads wav,
   mp3 and flac (llama.cpp b12316 sniffs RIFF/WAVE, ID3 or a MPEG sync word, fLaC), so the file is
   sent as it is; the microphone recording is encoded as WAV here because a browser's MediaRecorder
   gives webm or ogg, which the lane refuses. Its answer is text only ({"text": ...} at b12316:
   no segments, no timestamps, any response_format but json refused), so there is no .srt. */

// The language list: the lane's own page measures French and English, and NVIDIA's model card
// announces 40 languages, 13 of them at its "Leading" quality level; the codes are what
// llama-server puts in its prompt (server-chat.cpp: "(language: %s)"), or empty to detect.
const TR_LANGUAGES = [['', 'Detect it'], ['fr', 'French'], ['en', 'English'], ['de', 'German'],
  ['es', 'Spanish'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'],
  ['ru', 'Russian'], ['cs', 'Czech'], ['sk', 'Slovak'], ['hi', 'Hindi'], ['ko', 'Korean']];
// The decoded cap the server runs too (cockpit.py TRANSCRIBE_MAX_AUDIO): checked here before the
// file is read, so a file past it never leaves the browser.
const TR_MAX_AUDIO = 64 * 1024 * 1024;
// What the lane reads, read out of its server at its pinned revision (llama.cpp b12316,
// tools/mtmd/mtmd-helper.cpp is_audio_file): RIFF/WAVE, MP3 (ID3 or a MPEG sync word) and FLAC.
// A webm or ogg from a phone or a browser is refused there, so it is refused here with the command
// that converts it, and nothing is converted on the server.
const TR_EXT_OK = /\.(wav|mp3|flac)$/i;
const TR_MIME_OK = /audio\/(wav|x-wav|wave|mpeg|mp3|flac|x-flac)|application\/(x-)?flac/i;
const TRS = {audio: null, inflight: null, busy: false, available: false,
             port: 30036, host: '127.0.0.1', result: null, touched: false};

const trVal = id => ($(id) ? $(id).value : '');

function trFillLangs(){
  const sel = $('tcr-lang');
  if (sel.dataset.filled) return;
  sel.dataset.filled = '1';
  TR_LANGUAGES.forEach(([v, label]) => { const o = el('option', null, label); o.value = v; sel.append(o); });
  sel.value = '';
}

function trLaneWhy(){
  const e = engines()[TRANSCRIBE_UNIT];
  if (!F.life) return 'Checking the lane…';
  if (!e) return 'The NVIDIA Nemotron Transcribe lane is not installed on this box.';
  if (TRANSITIONAL.has(e.state)) return 'The lane is booting: Transcribe turns on by itself when it answers.';
  return 'The transcribe lane is not serving. Load it first.';
}

// ── the lane callout: what to do when it is not serving ─────────────────────
function trRenderLane(){
  const box = $('tcr-lane'), e = engines()[TRANSCRIBE_UNIT], s = servingEngine();
  const ready = !!e && (e.state === 'ready' || e.state === 'degraded');
  TRS.available = ready;
  if (!ready){ TRS.busy = false; }
  let sig, build;
  if (!F.life){ sig = 'wait'; build = () => {}; }
  else if (!e){ sig = laneFits(TRANSCRIBE_UNIT) ? 'absent' : 'absent:64'; build = () => {
    const b = el('div', 'banner info'); b.append(el('span', 'lamp'));
    const d = el('div'); d.append(el('b', null, 'The transcribe lane is not installed. '),
      el('span', null, sig === 'absent' ? 'On the box: ./install.sh --with-transcribe (about 37 GB of GGUF and 8 of image).'
                                        : `It ${NEEDS_128}, and this box is a 64 GB one.`));
    b.append(d); box.append(b); }; }
  else if (ready){ sig = 'ready'; build = () => {}; }
  else if (TRANSITIONAL.has(e.state)){
    sig = 'boot'; build = () => {
      const card = el('div', 'progress-card'); card.style.marginBottom = 'var(--s-5)';
      const top = el('div', 'top'); top.append(el('b', null, `Loading ${LANE_NAME[TRANSCRIBE_UNIT]}`), el('span', 'eta', ''));
      const m = el('div', 'meter'); m.append(el('i')); card.append(top, m, el('span', 'help', '')); box.append(card); };
  } else {
    sig = 'down:' + (s ? s[0] : ''); build = () => {
      const card = el('div', 'banner info'); card.style.alignItems = 'center';
      card.append(el('span', 'lamp' + (e.state === 'failed' ? ' err' : '')));
      const d = el('div'); d.style.display = 'flex'; d.style.flexWrap = 'wrap'; d.style.alignItems = 'center'; d.style.gap = 'var(--s-3)';
      const t = el('span', null, e.state === 'failed' ? 'The transcribe lane failed. Its journal is in Logs.'
        : s ? `${laneLabel(s[0])} holds the box. Loading the transcribe lane stops it first.` : 'The transcribe lane is stopped.');
      t.style.flex = '1'; t.style.minWidth = '240px';
      const b = el('button', 'btn primary', `Load ${LANE_NAME[TRANSCRIBE_UNIT]}`); b.type = 'button';
      b.addEventListener('click', () => askJourney('transcribe'));
      d.append(t, b); card.append(d); box.append(card); };
  }
  if (box.dataset.sig !== sig){ box.dataset.sig = sig; clear(box); build(); }
  if (sig === 'boot'){
    const card = box.firstChild, eta = e.eta || READY_DEFAULT[TRANSCRIBE_UNIT], pct = eta && e.elapsed ? Math.min(97, 100 * e.elapsed / eta) : 6;
    card.querySelector('.meter i').style.width = pct.toFixed(1) + '%';
    setText(card.querySelector('.eta'), e.elapsed && eta ? `about ${fmtDur(Math.max(0, eta - e.elapsed))} left` : '');
    setText(card.querySelector('.help'), `${STATE_LABEL[e.state]}${e.detail && e.detail !== 'ready' ? ', ' + e.detail : ''}, ${fmtDur(e.elapsed)} elapsed`);
  }
  cap('tcr-cap', !e ? 'not installed' : ready ? (TRS.busy || TRS.inflight ? 'transcribing' : 'ready') : STATE_LABEL[e.state] || e.state,
      !e ? '' : ready ? (TRS.busy || TRS.inflight ? 'warn' : 'ok') : stateKind(e.state), ready && (TRS.busy || TRS.inflight) ? 'live' : e ? stateLive(e.state) : false);
  badge('transcribe', TRS.inflight || TRS.busy ? '●' : '', 'live');
}

async function trLane(){
  try{
    const r = await fetch('/api/transcribe'); if (r.status === 401) return;
    const d = await r.json();
    TRS.port = d.port || 30036; TRS.host = d.host || '127.0.0.1';
    TRS.busy = !!d.busy;               // the lane's lock: another transcription is running
    trRenderLane(); trSync();
  }catch{ cap('tcr-cap', 'unknown', 'warn'); }
}

function trProblem(){
  if (!TRS.audio) return 'Drop an audio file or record one first.';
  if (TRS.audio.size > TR_MAX_AUDIO)
    return `The file is ${(TRS.audio.size / 1e6).toFixed(1)} MB and the cap is ${TR_MAX_AUDIO / 1048576} MB (about 33 minutes at the 16 kHz mono this page records). Cut it or convert it first.`;
  return '';
}

function trSync(){
  trFillLangs();
  const p = trProblem();
  show('tcr-problem', !!p && TRS.touched); setText('tcr-problem', p);
  const why = laneWorkWaits() || (!TRS.available ? trLaneWhy() : TRS.inflight ? 'Your file is being transcribed.' : TRS.busy ? 'Someone else’s file is being transcribed: one at a time on this lane.' : p);
  $('tcr-run').disabled = !!why;
  show('tcr-why', !!why && why !== p); setText('tcr-why', why);
  // what the audio in hand is, said once it is set
  show('tcr-audio-info', !!TRS.audio);
  if (TRS.audio) setText('tcr-audio-info', `${TRS.audio.name} · ${(TRS.audio.size / 1e6).toFixed(2)} MB${TRS.audio.duration ? ' · ' + fmtDur(TRS.audio.duration) : ''}${TRS.audio.recorded ? ' · recorded here' : ''}`);
  setText('tcr-drop-label', TRS.audio ? 'Replace the audio file' : 'Drop an audio file, or pick one: wav, mp3, flac');
  trCurl();
}

// ── the audio in hand: dropped, picked, or recorded (below) ─────────────────
function trPickFiles(files){
  if (!files || !files.length) return;
  const f = files[0];
  TRS.touched = true;
  if (f.size > TR_MAX_AUDIO){
    TRS.audio = null; trSync();
    toast(`The file is ${(f.size / 1e6).toFixed(1)} MB and the cap is ${TR_MAX_AUDIO / 1048576} MB, about 33 minutes at the 16 kHz mono this page records. Cut it or convert it first.`, 'warn', 8000);
    return;
  }
  if (!TR_EXT_OK.test(f.name || '') && !TR_MIME_OK.test(f.type || '')){
    TRS.audio = null; trSync();
    toast(`${f.name}: the lane reads wav, mp3 and flac only. Convert it first: ffmpeg -i ${f.name} -ar 16000 -ac 1 out.wav`, 'warn', 9000);
    return;
  }
  const fr = new FileReader();
  fr.onload = () => trAcceptAudio(f.name, f.size, f.type || 'audio/wav', String(fr.result), null, false);
  fr.onerror = () => toast(`${f.name} could not be read.`, 'err');
  fr.readAsDataURL(f);
}
function trAcceptAudio(name, size, mime, dataUrl, duration, recorded){
  TRS.audio = {name, size, type: mime, dataUrl, duration: duration || null, recorded: !!recorded};
  TRS.result = null;
  trShowEmpty(); trCurl(); trSync();
}

// ── the microphone: recorded in the browser, encoded as WAV here ─────────────
// getUserMedia exists only in a secure context (HTTPS, or localhost), so on the plain HTTP
// the cockpit is often served on (its Tailscale address), the button is disabled and says why,
// with no error. MediaRecorder gives webm or ogg, which the lane refuses, so the recording is
// decoded and re-encoded as WAV (16 kHz mono PCM16) here: no conversion is added server-side.
function trMicState(){
  if (!window.isSecureContext)
    return {ok: false, why: `The browser opens the microphone only on a secure context (HTTPS, or localhost): this page is served over plain HTTP on ${location.host}. Open the cockpit as https://${location.host}, or on http://localhost:${location.port}, to record here.`};
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    return {ok: false, why: 'This browser has no getUserMedia: record on another browser, or drop an audio file instead.'};
  if (typeof MediaRecorder === 'undefined' || typeof AudioContext === 'undefined' || typeof OfflineAudioContext === 'undefined')
    return {ok: false, why: 'This browser cannot record and re-encode here (no MediaRecorder or AudioContext): drop an audio file instead.'};
  return {ok: true, why: ''};
}
function trSyncMic(){
  const st = trMicState();
  const b = $('tcr-rec');
  if (!st.ok && !TRS.recording){
    b.disabled = true; b.title = st.why;
    setText('tcr-rec-note', st.why); show('tcr-rec-note', true);
  } else {
    b.disabled = !!TRS.recording; b.title = st.why || 'Record from this browser’s microphone';
    if (!TRS.recording) show('tcr-rec-note', false);
  }
}
async function trRecordStart(){
  const st = trMicState();
  if (!st.ok) return toast(st.why, 'warn', 9000);
  try{
    const stream = await navigator.mediaDevices.getUserMedia(
      {audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    const rec = new MediaRecorder(stream);
    const chunks = [];
    rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    let resolveDone;
    const done = new Promise(r => { resolveDone = r; });
    rec.onstop = () => resolveDone(new Blob(chunks, {type: rec.mimeType || 'audio/webm'}));
    rec.start();
    TRS.recording = {rec, stream, done};
    show('tcr-rec-stop', true);
    setText('tcr-rec-note', 'Recording. The browser hands over webm or ogg, so it is decoded and encoded as WAV (16 kHz mono) here, which is what the lane reads.');
    show('tcr-rec-note', true);
    trSyncMic();
  }catch(e){ toast('The microphone was refused: ' + e.message, 'err', 9000); }
}
async function trRecordStop(){
  const rec = TRS.recording; if (!rec) return;
  TRS.recording = null;
  show('tcr-rec-stop', false);
  try{ rec.rec.stop(); } catch { /* already stopped */ }
  rec.stream.getTracks().forEach(t => { try{ t.stop(); } catch { /* already stopped */ } });
  trSyncMic();
  try{
    const blob = await rec.done;
    const actx = new AudioContext();
    let decoded;
    try{ decoded = await actx.decodeAudioData(await blob.arrayBuffer()); }
    finally{ actx.close(); }     // Chrome caps the open AudioContexts (about 6): close it here
    const rate = 16000;                       // the recipe's sampling rate, mono
    const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * rate)), rate);
    const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start();
    const rendered = await off.render();
    const wav = trEncodeWav(rendered.getChannelData(0), rate);
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    trAcceptAudio(`recording-${stamp}.wav`, wav.length, 'audio/wav',
                  'data:audio/wav;base64,' + trBytesToB64(wav), decoded.duration, true);
    toast(`Recorded ${fmtDur(decoded.duration)} as WAV (16 kHz mono), ready to transcribe.`, 'ok', 5000);
  }catch(e){ toast('The recording could not be encoded as WAV: ' + e.message, 'err', 9000); }
}
// WAV, PCM16, mono: the header the lane's miniaudio reads, then the samples clamped to [-1, 1]
function trEncodeWav(samples, sampleRate){
  const n = samples.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const tag = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  tag(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); tag(8, 'WAVE'); tag(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); tag(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++){
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Uint8Array(buf);
}
// base64 of bytes, in chunks: String.fromCharCode on a whole recording overflows the stack
function trBytesToB64(bytes){
  let out = '';
  for (let i = 0; i < bytes.length; i += 8192)
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  return btoa(out);
}

// ── the transcript ───────────────────────────────────────────────────────────
function trShowEmpty(){
  const sc = $('tcr-screen'); if (!sc) return;
  sc.dataset.mode = 'empty'; clear(sc);
  const d = el('div', 'empty');
  d.append(el('b', null, 'Your transcript appears here'), el('span', null, 'Drop an audio file or record one, then transcribe.'));
  sc.append(d);
  setText('tcr-time', '');
  show('tcr-copy', false); show('tcr-dl', false);
}
function trShow(out){
  TRS.result = {text: out.text || '', seconds: out.seconds, usage: out.usage, at: Date.now(),
                base: ((TRS.audio && TRS.audio.name) || 'transcript').replace(/\.[^.]+$/, '') || 'transcript'};
  const sc = $('tcr-screen'); clear(sc); sc.dataset.mode = 'show';
  const pre = el('pre'); pre.className = 'cmd';
  pre.style.cssText = 'white-space:pre-wrap; word-break:break-word; margin:0; max-height:440px; overflow:auto; font-size:var(--t-14)';
  pre.textContent = TRS.result.text;
  sc.append(pre);
  setText('tcr-time', out.seconds != null ? out.seconds.toFixed(1) + ' s' : '');
  show('tcr-copy', true); show('tcr-dl', true);
}
function trDownloadName(){ return (TRS.result ? TRS.result.base : 'transcript') + '.txt'; }
function trDownloadUrl(){
  // percent-encoded, not base64: the transcript is UTF-8 and the file is text
  return 'data:text/plain;charset=utf-8,' + encodeURIComponent(TRS.result ? TRS.result.text : '');
}
function trDownload(){
  if (!TRS.result) return;
  const a = document.createElement('a');
  a.href = trDownloadUrl(); a.download = trDownloadName(); a.click();
}

// ── the call, through the cockpit's relay ────────────────────────────────────
async function trRun(){
  TRS.touched = true;
  const wait = laneWorkWaits(); if (wait) return toast(wait, 'warn');
  const p = trProblem(); if (p) return toast(p, 'warn');
  if (!TRS.available) return toast(trLaneWhy(), 'warn');
  const t0 = Date.now();
  TRS.inflight = t0;
  const sc = $('tcr-screen'); if (sc.dataset.mode !== 'work'){ sc.dataset.mode = 'work'; clear(sc);
    const d = el('div', 'empty'); d.append(el('b', null, 'Transcribing…'), el('span', null, 'About ten times faster than real time on this box.')); sc.append(d); }
  trSync();
  try{
    const body = {audio: TRS.audio.dataUrl, filename: TRS.audio.name, language: trVal('tcr-lang')};
    const {status, ok, out} = await postJSON('/api/transcribe', body);
    if (!ok){
      const why = out.error || (out.refused ? JSON.stringify(out.refused).slice(0, 300) : 'HTTP ' + status);
      if (status === 409){ TRS.busy = true; trSync(); }   // the next trLane poll clears it when the lane is free
      toast((status === 409 ? '' : `Refused (${status}): `) + why, status === 409 ? 'warn' : 'err', 8000);
      return;
    }
    trShow(out);
    toast(`Done in ${out.seconds != null ? out.seconds.toFixed(1) + ' s' : fmtDur((Date.now() - t0) / 1000)}.`, 'ok', 3000);
  }catch(e){ toast('The cockpit could not reach the transcribe lane: ' + e.message, 'err'); }
  finally{ TRS.inflight = null; trSync(); trLane(); }
}

// ── the same call from a terminal: the key is read on the box, never printed ──
function trCurl(){
  const a = TRS.audio, lang = trVal('tcr-lang');
  setText('tcr-curl', [
    '# run this ON the box: the NVIDIA Nemotron Transcribe lane listens on loopback and checks the box key',
    `curl -sS http://${TRS.host}:${TRS.port}/v1/audio/transcriptions \\`,
    `  -H @<(printf 'Authorization: Bearer %s\\n' "$(head -1 ~/.config/qwen38/api-key)") \\`,
    `  -F file=@${shq(a ? a.name : 'audio.wav')} -F response_format=json` + (lang ? ` -F language=${lang}` : ''),
  ].join('\n'));
}

function wireTranscribe(){
  trFillLangs();
  $('tcr-lang').addEventListener('change', () => { trCurl(); trSync(); });
  $('tcr-file').addEventListener('change', e => { trPickFiles(e.target.files); e.target.value = ''; });
  const dz = $('tcr-drop');
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => trPickFiles(e.dataTransfer ? e.dataTransfer.files : null));
  $('tcr-run').addEventListener('click', trRun);
  $('tcr-reset').addEventListener('click', () => { TRS.audio = null; TRS.result = null; TRS.touched = false; trShowEmpty(); trCurl(); trSync(); });
  $('tcr-copy').addEventListener('click', () => { if (TRS.result) copyText(TRS.result.text); });
  $('tcr-dl').addEventListener('click', trDownload);
  $('tcr-curl-copy').addEventListener('click', () => copyText($('tcr-curl').textContent));
  $('tcr-rec').addEventListener('click', trRecordStart);
  $('tcr-rec-stop').addEventListener('click', trRecordStop);
  document.addEventListener('visibilitychange', () => { if (!document.hidden && activeView === 'transcribe') trLane(); });
  trShowEmpty(); trCurl(); trSync(); trSyncMic();
}
onShow('transcribe', trLane);
onLoadChange(trSync);
afterApply(() => { trRenderLane(); trSync(); });
