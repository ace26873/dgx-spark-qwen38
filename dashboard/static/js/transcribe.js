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
}

function wireTranscribe(){
  trFillLangs();
  $('tcr-lang').addEventListener('change', trSync);
  $('tcr-reset').addEventListener('click', () => { TRS.audio = null; TRS.result = null; TRS.touched = false; trSync(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden && activeView === 'transcribe') trLane(); });
  trSync();
}
onShow('transcribe', trLane);
onLoadChange(trSync);
afterApply(() => { trRenderLane(); trSync(); });
