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
const TRS = {audio: null, inflight: null, busy: false, available: false, watching: null,
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
  document.addEventListener('visibilitychange', () => { if (!document.hidden && activeView === 'transcribe') trLane(); });
  trShowEmpty(); trCurl(); trSync();
}
onShow('transcribe', trLane);
onLoadChange(trSync);
afterApply(() => { trRenderLane(); trSync(); });
