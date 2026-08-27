"""Transform the pristine Claude Design export into a live, data-backed page.

    python web/app/wire.py

Reads  web/app/design.dc.html   (exported from Claude Design, never edited)
Writes web/app/index.html       (same design, wired to the API)

Every replacement asserts it matched. If the design is re-exported and a
patched region has moved or changed wording, this fails loudly rather than
silently shipping a page with dead bindings or hardcoded numbers.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "design.dc.html"
OUT = HERE / "index.html"

patches: list[tuple[str, str, str]] = []   # (label, find, replace)


def patch(label: str, find: str, replace: str) -> None:
    patches.append((label, find, replace))


# ── 1. vendored React before the runtime, so it never reaches for a CDN ──
patch(
    "pulse keyframes",
    "  ::-webkit-scrollbar { width: 12px; }",
    "  @keyframes dcpulse { 0%,100% { opacity: 1 } 50% { opacity: .25 } }\n"
    "  ::-webkit-scrollbar { width: 12px; }",
)

patch(
    "vendor react",
    '<script src="./support.js"></script>',
    '<script src="./vendor/react.production.min.js"></script>\n'
    '<script src="./vendor/react-dom.production.min.js"></script>\n'
    '<script src="./support.js"></script>',
)

# ── 2. the mock corpus is deleted outright, not renamed ───────────────────
# Renaming was not enough: the mock arrays reference the mock documents, so a
# half-renamed corpus throws `NW_DOC is not defined` at load. It is also ~36KB
# of invented contracts, which has no business shipping inside a page whose
# entire claim is that everything on screen is quoted from a real file.

LIVE_DATA_DECL = """// ─────────────────────────────────────────────────────────
// WIRED: every value below is fetched from the analysis API at runtime.
// The design's mock corpus was removed. Shapes: api/viewmodel.py.
// ─────────────────────────────────────────────────────────
const API = '';
let CONTRACTS = [], OBLIGATIONS = [], FINDINGS = [], GAPS = [];
let INJECTIONS = [], EVAL = [], COVERAGE = [], COVERAGE_COLS = [];
let UPLOAD_CHECKS = [], STATS = {}, EVAL_STATS = {}, AS_OF = '';

function CAL_SUM(st) { return (st.calendar && st.calendar.summary) || {}; }

function MODEL_LABEL(st) {
  const h = st.health || {};
  return { ok: 'Model ready', unknown: 'Model idle',
           rate_limited: 'Model rate limited', error: 'Model unavailable',
           no_key: 'No model connected' }[h.status] || 'Model unknown';
}

function MODEL_COLOR(st) {
  const h = st.health || {};
  if (h.status === 'ok') return 'var(--color-accent-700)';
  if (h.status === 'unknown') return 'color-mix(in srgb,var(--color-text) 45%,transparent)';
  return 'var(--color-accent)';
}

const CHANGE_ROWS = [
  { key: 'clauses', label: 'clauses extracted' },
  { key: 'actionable_dates', label: 'dates to act on' },
  { key: 'calendar_events', label: 'calendar entries' },
  { key: 'findings', label: 'things to look at' },
  { key: 'flow_down_gaps', label: 'flow-down gaps' },
  { key: 'contracts', label: 'contracts held' },
];

// Splits the document into plain and highlighted runs: the passages already
// found, plus the window currently being read. Offsets are the real ones the
// extractor reported, so the highlight is where the model actually worked.
function buildReaderSegs(st) {
  const text = st.upText || '';
  if (!text) return [];
  const marks = (st.upSpans || []).map(function (s) {
    return { start: s.start, end: s.end, kind: 'found' };
  });
  if (st.upRead) marks.push({ start: st.upRead.start, end: st.upRead.end,
                              kind: 'reading' });
  marks.sort(function (a, b) { return a.start - b.start; });

  const segs = [];
  let cursor = 0;
  marks.forEach(function (m) {
    if (m.start < cursor) return;
    if (m.start > cursor) segs.push({ text: text.slice(cursor, m.start),
                                      style: 'color:inherit', active: '0' });
    segs.push({
      text: text.slice(m.start, m.end),
      active: m.kind === 'reading' ? '1' : '0',
      style: m.kind === 'reading'
        ? 'background:var(--color-accent);color:var(--color-bg)'
        : 'background:var(--color-accent-200)',
    });
    cursor = m.end;
  });
  if (cursor < text.length) segs.push({ text: text.slice(cursor),
                                        style: 'color:inherit', active: '0' });
  return segs;
}

const ASK_SUGGESTIONS = [
  'When does Northwind renew, and what is the deadline to stop it?',
  'Where are we exposed across contracts?',
  'What is missing from the Helios NDA?',
  'What would it cost to leave Northwind early?',
  'Which contract locks us in the longest?'
];

// Static on purpose: this describes OUR pipeline, not any contract's data.
// It is the only copy on the page that is not derived from a document.
const PIPELINE = [
  ['01', 'We read the text and remember where every word sat',
   'Character positions have to survive page joins, headers and hyphenation, or every quote we show you would drift.'],
  ['02', 'We check the file for tampering',
   'Text too small to read, text the same colour as the page, glyphs off the page edge, instructions in metadata. All of it before anything is sent to a model.'],
  ['03', 'We turn wording into claims',
   'The model does one job: converting language into structured claims. It is never asked to calculate anything.'],
  ['04', 'We check every quote against your document',
   'If a quoted span is not found in the source character for character, it is thrown away rather than shown to you.'],
  ['05', 'We work out the dates and scores ourselves',
   'Plain arithmetic over verified claims, with the working shown on every result.']
];

"""


def strip_mock(text: str) -> str:
    """Cut everything from the first mock document to the first real helper."""
    start = text.index("const NW_DOC")
    end = text.index("const MONTHS")
    return text[:start] + LIVE_DATA_DECL + text[end:]

# ── 3. offsets come from the verified span, not from indexOf ──────────────
patch(
    "quote() uses verified offsets",
    "quote(c, qid) { const q = (c.quotes || []).filter(function (x) { return x.id === qid; })[0]; "
    "if (!q) return null; const i = c.doc.indexOf(q.text); return i < 0 ? null : "
    "{ q: q, start: i, end: i + q.text.length }; }",
    """quote(c, qid) {
    const q = (c.quotes || []).filter(function (x) { return x.id === qid; })[0];
    if (!q) return null;
    /* Offsets are the ones the verifier checked, not a text search: repeated
       wording would otherwise highlight the wrong occurrence. */
    if (typeof q.start === 'number') return { q: q, start: q.start, end: q.end };
    const i = c.doc.indexOf(q.text);
    return i < 0 ? null : { q: q, start: i, end: i + q.text.length };
  }""",
)

patch(
    "offsets() shows the source document",
    "offsets(o) { return this.props.showOffsets === false ? '' : 'characters ' "
    "+ o.start + '–' + o.end + ' of the document'; }",
    "offsets(o) { if (this.props.showOffsets === false) return ''; "
    "const q = o.q || {}; const s = q.srcStart != null ? q.srcStart : o.start; "
    "const e = q.srcEnd != null ? q.srcEnd : o.end; "
    "return 'characters ' + s + '–' + e + (q.srcFile ? ' of ' + q.srcFile : ' of the document') "
    "+ (q.page ? ' · page ' + q.page : ''); }",
)

# ── 4. exit cost comes from the backend, not from invented constants ──────
patch(
    "real exit cost",
    "const monthly = c.value / 12;",
    "const apiExit = st.exitCost && st.exitCost.cid === c.id ? st.exitCost : null;\n"
    "    const monthly = c.value / 12;",
)
patch(
    "exit lines from API",
    "const exitLines = [",
    "const exitLines = apiExit ? apiExit.lines.map(function (l) {\n"
    "      return { label: l.label, amount: self.money(l.amount), formula: l.detail,\n"
    "               hasQuote: !!l.quote, quote: l.quote || '',\n"
    "               select: function () { if (l.qid) self.openDrawer(null, l.qid); } };\n"
    "    }) : [",
)
patch(
    "exit total from API",
    "const total = remFees + etc + exportFee + forfeit;",
    "const total = apiExit ? apiExit.total : (remFees + etc + exportFee + forfeit);",
)

# ── 5. prose that would otherwise state numbers we no longer have ─────────
patch(
    "held-back stat",
    '<div style="font-family:var(--font-heading);font-weight:800;font-size:52px;'
    'line-height:1;font-variant-numeric:tabular-nums">1</div>\n'
    '            <div style="font-size:14px;margin-top:10px;line-height:1.55;'
    'max-width:320px">Tessellate Media. We don\'t put dates in your calendar from '
    'a document we don\'t trust.</div>',
    '<div style="font-family:var(--font-heading);font-weight:800;font-size:52px;'
    'line-height:1;font-variant-numeric:tabular-nums">{{ statHeld }}</div>\n'
    '            <div style="font-size:14px;margin-top:10px;line-height:1.55;'
    'max-width:320px">{{ statHeldNote }}</div>',
)
patch(
    "model status indicator",
    '<span style="font-size:13px;color:color-mix(in srgb,var(--color-text) 55%,'
    'transparent)">{{ asOfLabel }}</span>',
    '<span style="font-size:13px;color:color-mix(in srgb,var(--color-text) 55%,'
    'transparent)">{{ asOfLabel }}</span>\n'
    '      <span title="{{ modelDetail }}" style="display:flex;align-items:center;'
    'gap:7px;font-size:12.5px;color:{{ modelColor }}">'
    '<span style="width:8px;height:8px;background:{{ modelColor }};display:block">'
    '</span>{{ modelLabel }}</span>',
)

patch(
    "findings intro",
    "Nineteen things across six contracts. Four of them are about wording that isn't "
    "in the document at all — the kind of gap a folder of PDFs will never show you.",
    "{{ findingsIntro }}",
)
patch(
    "eval intro",
    "Measured on 148 contracts from CUAD that were never used to tune anything. "
    "Every clause type we score is here, including the four we're bad at.",
    "{{ evalIntro }}",
)
patch("eval grounding stat", ">96.4%<", ">{{ evalGrounding }}<")
patch(
    "eval grounding note",
    "the other 3.6% were thrown away before you saw them",
    "{{ evalGroundingNote }}",
)
patch(
    "eval hallucination note",
    "a claim without matching text in your document cannot be displayed",
    "{{ evalZeroNote }}",
)
patch("eval latency", ">11.8s<", ">{{ evalLatency }}<")
patch("eval latency note", "median, on a 22-page document", "{{ evalLatencyNote }}")
patch("eval spans", ">3,412<", ">{{ evalSpans }}<")
patch(
    "eval spans note",
    "all 3,412 matched the document exactly",
    "{{ evalSpansNote }}",
)
patch(
    "security headline",
    "Someone hid three instructions in this vendor's contract",
    "{{ secHeadline }}",
)
patch(
    "security file label",
    "tessellate-msa-2026.pdf · 11 pages · received 18 Apr 2026",
    "{{ secFileLabel }}",
)
patch(
    "chain headline",
    "You promised two customers more than your provider promised you",
    "{{ chainHeadline }}",
)
patch(
    "chain subhead",
    "Three contracts, read together. Every figure below is quoted from one of them; "
    "the differences are arithmetic.",
    "{{ chainSubhead }}",
)

patch(
    "deadline subhead names real contracts",
    "        : 'Meridian Data Systems renewed on 16 August because nobody wrote to "
    "them in time. Northwind is next, and it is the larger of the two.',",
    "        : (missed.length\n"
    "            ? missed[0].oc.party + ' renewed on ' + self.fmt(missed[0].o.due) +\n"
    "              ' because nobody wrote to them in time.' +\n"
    "              (nextUrgent ? ' ' + nextUrgent.oc.party + ' is next.' : '')\n"
    "            : (nextUrgent\n"
    "               ? nextUrgent.oc.party + ' is next, on ' + self.fmt(nextUrgent.o.due) +\n"
    "                 '. Every date here was worked out from wording we can show you.'\n"
    "               : 'Nothing is scheduled inside this window.')),",
)
patch(
    "missed-stat note names the real contract",
    "statMissedNote: missed.length ? 'Meridian Data Systems renewed for another "
    "year at $96,400, and the fees cannot be cancelled.' : 'Nothing has gone past "
    "its date.',",
    "statMissedNote: missed.length\n"
    "        ? missed[0].oc.party + ' renewed for another term'\n"
    "          + (missed[0].oc.value ? ' at ' + self.money(missed[0].oc.value) : '')\n"
    "          + ', and the fees cannot be cancelled.'\n"
    "        : 'Nothing has gone past its date.',",
)

# ── 5b. state, mount and identity fixes ──────────────────────────────────
patch(
    "state",
    "state = { view:'deadlines', cid:'northwind', tab:'findings', sel:null, "
    "drawerFinding:null, openOb:'nw-notice-ob', win:'90', sev:'all', "
    "role:'finance', exitDate:'2027-03-31' };",
    "state = { view:'deadlines', cid:null, tab:'findings', sel:null, "
    "drawerFinding:null, openOb:null, win:'90', sev:'all', "
    "role:'finance', exitDate:'2027-03-31', "
    "loaded:false, err:null, uploading:false, uploadMsg:'', exitCost:null, "
    "askQuestion:'', asking:false, askResult:null, calendar:null, "
    "askActivity:[], askPlan2:[], askPhase:'', askElapsed:'', "
    "upFiles:[], upLog:[], upChanges:null, upText:'', upSpans:[], "
    "upRead:null, upActive:null, upPhase:'', health:null };",
)
patch(
    "fetch on mount",
    "componentDidMount() { this.scrollToSel(); }",
    "componentDidMount() { this.scrollToSel(); this.loadModel(); this.loadHealth(); /* Health is learned from real calls, so re-read it periodically. */ this._health = setInterval(this.loadHealth.bind(this), 20000); } componentWillUnmount() { if (this._health) clearInterval(this._health); }",
)
patch(
    "exit refetch when the date or contract changes",
    "componentDidUpdate(pp, ps) { if (ps.sel !== this.state.sel || "
    "ps.drawerFinding !== this.state.drawerFinding) this.scrollToSel(); }",
    "componentDidUpdate(pp, ps) { "
    "/* The runtime calls this with prevProps only, so prevState is undefined. "
    "   The design assumed two arguments; unguarded it throws on every state "
    "   change and silently breaks re-render. */ "
    "ps = ps || {}; "
    "if (ps.sel !== this.state.sel || "
    "ps.drawerFinding !== this.state.drawerFinding) this.scrollToSel(); "
    "if (this.state.loaded && (ps.exitDate !== this.state.exitDate || "
    "ps.cid !== this.state.cid)) this.loadExit(); }",
)
patch(
    "contract() falls back to the first real contract",
    "contract(id) { const r = CONTRACTS.filter(function (c) { return c.id === id; }); "
    "return r[0]; }",
    "contract(id) { const r = CONTRACTS.filter(function (c) { return c.id === id; }); "
    "return r[0] || CONTRACTS[0]; }",
)
patch(
    "clock comes from the API",
    "today() { return this.props.asOfDate || '2026-08-27'; }",
    "today() { return AS_OF || this.props.asOfDate || '2026-08-27'; }",
)

# ── 6. real file upload ───────────────────────────────────────────────────



# ── 7. Ask: questions answered from the extracted layer ───────────────────
# The design predates this feature, so the view is added here rather than
# edited into the export. Same visual language as the rest of the page.

ASK_VIEW = """
    <!-- ══════ ASK ══════ -->
    <sc-if value="{{ isAsk }}" hint-placeholder-val="{{ false }}">
      <div>
        <div style="padding:36px 32px 28px;border-bottom:2px solid var(--color-divider)">
          <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:14px">Ask</div>
          <h2 style="margin:0 0 12px;font-size:32px">Ask about your contracts</h2>
          <p style="margin:0;font-size:16px;line-height:1.65;max-width:800px;color:color-mix(in srgb,var(--color-text) 75%,transparent)">Answers come from what we already pulled out of your documents and checked &mdash; the clauses, the dates we worked out, and the wording we found missing. Not from re-reading the contract, and not from anything we know about contracts in general. If the answer isn&rsquo;t in there, we say so.</p>
        </div>

        <div style="padding:26px 32px;border-bottom:2px solid var(--color-divider);display:flex;gap:12px;align-items:center;flex-wrap:wrap">
          <input type="text" value="{{ askQuestion }}" onChange="{{ setAskQuestion }}" placeholder="When does Northwind renew, and what stops it?" style="flex:1;min-width:340px;font:inherit;font-size:16px;padding:14px 16px;border:2px solid var(--color-divider);background:var(--color-surface);color:var(--color-text)" />
          <button onClick="{{ submitAsk }}" class="btn btn-primary" style="flex:none">{{ askButtonLabel }}</button>
        </div>

        <div style="display:flex;gap:10px;padding:18px 32px 4px;flex-wrap:wrap;align-items:center">
          <span style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 50%,transparent);margin-right:4px">Try</span>
          <sc-for list="{{ askSuggestions }}" as="q" hint-placeholder-count="4">
            <button onClick="{{ q.go }}" class="btn btn-secondary" style="font-size:13px;padding:7px 12px">{{ q.label }}</button>
          </sc-for>
        </div>

        <sc-if value="{{ askIsRunning }}" hint-placeholder-val="{{ false }}">
          <div style="padding:22px 32px 8px;border-top:1px solid var(--color-divider);background:var(--color-surface)">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
              <span style="width:9px;height:9px;background:var(--color-accent);display:block;animation:dcpulse 1s ease-in-out infinite"></span>
              <span style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--color-accent-700);font-family:var(--font-heading);font-weight:800">{{ askPhase }}</span>
              <span style="font-size:13px;color:color-mix(in srgb,var(--color-text) 50%,transparent);font-variant-numeric:tabular-nums">{{ askElapsed }}</span>
            </div>
          </div>
        </sc-if>

        <sc-if value="{{ askHasActivity }}" hint-placeholder-val="{{ false }}">
          <div style="padding:4px 32px 18px;background:var(--color-surface);border-bottom:1px solid var(--color-divider)">
            <sc-for list="{{ askActivity }}" as="a" hint-placeholder-count="4">
              <div style="display:grid;grid-template-columns:22px 1fr;gap:12px;padding:5px 0 5px {{ a.indent }};align-items:baseline">
                <div style="font-size:12px;color:{{ a.color }};font-family:var(--font-heading);font-weight:800">{{ a.icon }}</div>
                <div>
                  <div style="font-size:14.5px;line-height:1.5;color:{{ a.color }}">{{ a.text }}<span style="color:color-mix(in srgb,var(--color-text) 45%,transparent);font-size:13px"> {{ a.detail }}</span></div>
                  <sc-if value="{{ a.hasQuote }}" hint-placeholder-val="{{ false }}">
                    <div style="border-left:2px solid var(--color-accent-200);padding:2px 0 2px 10px;margin-top:4px;font-size:13.5px;font-style:italic;color:color-mix(in srgb,var(--color-text) 72%,transparent)">&ldquo;{{ a.quote }}&rdquo;</div>
                  </sc-if>
                </div>
              </div>
            </sc-for>
          </div>
        </sc-if>

        <sc-if value="{{ askHasPlan }}" hint-placeholder-val="{{ false }}">
          <div style="padding:24px 32px 4px;border-top:1px solid var(--color-divider)">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:12px">How it worked it out</div>
            <sc-for list="{{ askPlan }}" as="p" hint-placeholder-count="3">
              <div style="display:grid;grid-template-columns:34px 1fr;gap:14px;padding:8px 0;align-items:baseline">
                <div style="font-size:13px;font-variant-numeric:tabular-nums;color:color-mix(in srgb,var(--color-text) 40%,transparent)">{{ p.n }}</div>
                <div style="font-size:15px;line-height:1.55">{{ p.text }}</div>
              </div>
            </sc-for>
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px">
              <sc-for list="{{ askSteps }}" as="s" hint-placeholder-count="3">
                <span class="tag" style="{{ s.style }}">{{ s.label }}</span>
              </sc-for>
            </div>
          </div>
        </sc-if>

        <sc-if value="{{ askHasAnswer }}" hint-placeholder-val="{{ true }}">
          <div style="padding:28px 32px 8px">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:12px">{{ askQuestionEcho }}</div>
            <div style="font-size:20px;line-height:1.65;max-width:860px;text-wrap:pretty">{{ askAnswer }}</div>
            <div style="font-size:13.5px;margin-top:14px;color:color-mix(in srgb,var(--color-text) 58%,transparent)">{{ askMeta }}</div>
          </div>

          <sc-if value="{{ askDegraded }}" hint-placeholder-val="{{ false }}">
          <div style="margin:20px 32px 4px;border:2px solid var(--color-accent);background:var(--color-accent-100);padding:20px 24px;max-width:900px">
            <div style="font-family:var(--font-heading);font-weight:800;font-size:17px;color:var(--color-accent-700);margin-bottom:6px">{{ degradedTitle }}</div>
            <div style="font-size:15.5px;line-height:1.6">{{ degradedBody }}</div>
          </div>
        </sc-if>

        <sc-if value="{{ askUnanswerable }}" hint-placeholder-val="{{ false }}">
            <div style="margin:20px 32px 8px;border:2px solid var(--color-accent);background:var(--color-accent-100);padding:20px 24px;max-width:860px">
              <div style="font-family:var(--font-heading);font-weight:800;font-size:17px;color:var(--color-accent-700);margin-bottom:6px">We would be guessing</div>
              <div style="font-size:15.5px;line-height:1.6">{{ askMissing }}</div>
            </div>
          </sc-if>

          <div style="padding:20px 32px 8px">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ askCitationsLabel }}</div>
          </div>
          <sc-for list="{{ askCitations }}" as="c" hint-placeholder-count="3">
            <div onClick="{{ c.open }}" style="display:grid;grid-template-columns:190px 1fr;gap:24px;padding:20px 32px;border-top:1px solid var(--color-divider);cursor:pointer" style-hover="background:color-mix(in srgb, var(--color-text) 4%, transparent)">
              <div>
                <div style="font-family:var(--font-heading);font-weight:800;font-size:15px">{{ c.contract }}</div>
                <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;margin-top:6px;color:{{ c.kindColor }}">{{ c.kind }}</div>
              </div>
              <div>
                <div style="font-size:15.5px;line-height:1.6;margin-bottom:8px">{{ c.title }}</div>
                <sc-if value="{{ c.hasQuote }}" hint-placeholder-val="{{ true }}">
                  <div style="border-left:2px solid var(--color-accent);padding:4px 0 4px 14px;font-size:14.5px;font-style:italic;line-height:1.65">&ldquo;{{ c.quote }}&rdquo; <span style="font-style:normal;font-size:11.5px;font-variant-numeric:tabular-nums;color:color-mix(in srgb,var(--color-text) 45%,transparent)">{{ c.offsets }}</span></div>
                </sc-if>
                <sc-if value="{{ c.isAbsence }}" hint-placeholder-val="{{ false }}">
                  <div style="font-size:14px;color:color-mix(in srgb,var(--color-text) 60%,transparent)">Nothing to quote &mdash; this is wording that is not in the document.</div>
                </sc-if>
              </div>
            </div>
          </sc-for>
        </sc-if>

        <div style="border-top:2px solid var(--color-divider);margin-top:26px;padding:26px 32px;font-size:14.5px;color:color-mix(in srgb,var(--color-text) 60%,transparent);max-width:860px;line-height:1.65">Every answer is assembled from records that were checked against your documents character by character. The model chooses which records answer you; it never writes the quotes, so it cannot invent one.</div>
      </div>
    </sc-if>
"""

patch("ask view markup", "    <!-- ══════ UPLOAD ══════ -->", ASK_VIEW + "\n    <!-- ══════ UPLOAD ══════ -->")

patch(
    "ask nav entry",
    "      { beat: 'Proof', items: [navItem('eval', 'Accuracy')] }",
    "      { beat: 'Proof', items: [navItem('eval', 'Accuracy')] },\n"
    "      { beat: 'Ask', items: [navItem('ask', 'Ask a question')] }",
)



# ── 8. Calendar: when documents arrived and every date inside them ────────

CALENDAR_VIEW = """
    <!-- ══════ CALENDAR ══════ -->
    <sc-if value="{{ isCalendar }}" hint-placeholder-val="{{ false }}">
      <div>
        <div style="padding:36px 32px 28px;border-bottom:2px solid var(--color-divider)">
          <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:14px">Calendar</div>
          <h2 style="margin:0 0 12px;font-size:32px">Every date we hold</h2>
          <p style="margin:0;font-size:16px;line-height:1.65;max-width:840px;color:color-mix(in srgb,var(--color-text) 75%,transparent)">Three kinds of date, kept apart on purpose. When each document reached us. Deadlines we worked out from the wording. And dates written in the text &mdash; which are <em>not</em> deadlines, and treating them as if they were is how a notice window gets missed by sixty days.</p>
        </div>

        <div style="display:grid;grid-template-columns:repeat(4,1fr);border-bottom:2px solid var(--color-divider)">
          <div style="padding:24px 32px">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:10px">Needs a decision</div>
            <div style="font-family:var(--font-heading);font-weight:800;font-size:44px;line-height:1;font-variant-numeric:tabular-nums;color:var(--color-accent)">{{ calActionable }}</div>
            <div style="font-size:13.5px;margin-top:8px">{{ calActionableNote }}</div>
          </div>
          <div style="padding:24px 32px;border-left:2px solid var(--color-divider)">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:10px">Documents received</div>
            <div style="font-family:var(--font-heading);font-weight:800;font-size:44px;line-height:1;font-variant-numeric:tabular-nums">{{ calDocuments }}</div>
            <div style="font-size:13.5px;margin-top:8px">each stamped when we first read it</div>
          </div>
          <div style="padding:24px 32px;border-left:2px solid var(--color-divider)">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:10px">Worked out by us</div>
            <div style="font-family:var(--font-heading);font-weight:800;font-size:44px;line-height:1;font-variant-numeric:tabular-nums">{{ calComputed }}</div>
            <div style="font-size:13.5px;margin-top:8px">none of these appear in any document</div>
          </div>
          <div style="padding:24px 32px;border-left:2px solid var(--color-divider)">
            <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:10px">Written in the text</div>
            <div style="font-family:var(--font-heading);font-weight:800;font-size:44px;line-height:1;font-variant-numeric:tabular-nums">{{ calWritten }}</div>
            <div style="font-size:13.5px;margin-top:8px">quoted, with the line they sit on</div>
          </div>
        </div>

        <sc-for list="{{ calendarMonths }}" as="m" hint-placeholder-count="4">
          <div>
            <div style="display:flex;align-items:baseline;gap:14px;padding:22px 32px 12px;border-top:2px solid var(--color-divider);background:var(--color-surface)">
              <h4 style="margin:0;font-size:20px">{{ m.label }}</h4>
              <span style="font-size:13.5px;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ m.note }}</span>
            </div>
            <sc-for list="{{ m.events }}" as="e" hint-placeholder-count="3">
              <div onClick="{{ e.open }}" style="display:grid;grid-template-columns:118px 150px 1fr 130px;gap:20px;padding:16px 32px;border-top:1px solid var(--color-divider);align-items:baseline;cursor:pointer" style-hover="background:color-mix(in srgb, var(--color-text) 4%, transparent)">
                <div style="font-size:14.5px;font-variant-numeric:tabular-nums;color:{{ e.dateColor }}">{{ e.dateLabel }}</div>
                <div style="font-size:11px;letter-spacing:.08em;text-transform:uppercase;font-family:var(--font-heading);font-weight:800;color:{{ e.sourceColor }}">{{ e.sourceLabel }}</div>
                <div>
                  <div style="font-size:15.5px;line-height:1.5">{{ e.title }}</div>
                  <div style="font-size:13.5px;margin-top:5px;color:color-mix(in srgb,var(--color-text) 62%,transparent)">{{ e.detail }}</div>
                  <sc-if value="{{ e.hasQuote }}" hint-placeholder-val="{{ false }}">
                    <div style="border-left:2px solid var(--color-accent);padding:3px 0 3px 12px;margin-top:7px;font-size:14px;font-style:italic">&ldquo;{{ e.quote }}&rdquo;</div>
                  </sc-if>
                </div>
                <div style="text-align:right;font-size:13.5px;color:{{ e.dateColor }};font-variant-numeric:tabular-nums">{{ e.daysLabel }}</div>
              </div>
            </sc-for>
          </div>
        </sc-for>
      </div>
    </sc-if>
"""

patch("calendar view markup", "    <!-- ══════ UPLOAD ══════ -->",
      CALENDAR_VIEW + "\n    <!-- ══════ UPLOAD ══════ -->")

patch(
    "calendar loads on navigation",
    "    const go = function (v) { return function () "
    "{ self.setState({ view: v }); }; };",
    "    const go = function (v) {\n"
    "      return function () {\n"
    "        if (v === 'calendar') self.loadCalendar();\n"
    "        self.setState({ view: v });\n"
    "      };\n"
    "    };",
)

patch(
    "calendar nav entry",
    "      { beat: 'Ask', items: [navItem('ask', 'Ask a question')] }",
    "      { beat: 'Dates', items: [navItem('calendar', 'Calendar')] },\n"
    "      { beat: 'Ask', items: [navItem('ask', 'Ask a question')] }",
)



# ── 9. Upload: watch the document being read, and what it changed ─────────
# The whole upload block is replaced rather than patched: the design showed a
# static "what happens next" list, and this shows it actually happening.

UPLOAD_VIEW = """<!-- ══════ UPLOAD ══════ -->
    <sc-if value="{{ isUpload }}" hint-placeholder-val="{{ false }}">
      <div>
        <div style="padding:34px 32px 24px;border-bottom:2px solid var(--color-divider)">
          <div style="font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:12px">Add a document</div>
          <h2 style="margin:0 0 10px;font-size:32px">{{ uploadHeadline }}</h2>
          <p style="margin:0 0 18px;font-size:16px;line-height:1.6;max-width:800px;color:color-mix(in srgb,var(--color-text) 75%,transparent)">{{ uploadSubhead }}</p>
          <input type="file" multiple="true" onChange="{{ onPickFile }}" style="font:inherit;display:block" />
          <div style="margin-top:10px;font-size:14px;color:color-mix(in srgb,var(--color-text) 65%,transparent)">{{ uploadStatus }}</div>
        </div>

        <div style="display:grid;grid-template-columns:minmax(420px,1.05fr) minmax(400px,1fr)">

          <div style="border-right:2px solid var(--color-divider);min-width:0">
            <sc-if value="{{ uploadHasFiles }}" hint-placeholder-val="{{ false }}">
              <div style="padding:18px 28px 6px">
                <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent);margin-bottom:10px">Documents</div>
                <sc-for list="{{ uploadFiles }}" as="f" hint-placeholder-count="2">
                  <div onClick="{{ f.select }}" style="display:grid;grid-template-columns:18px 1fr auto;gap:12px;padding:9px 10px;margin:0 -10px;align-items:baseline;cursor:pointer;background:{{ f.bg }};border-left:3px solid {{ f.edge }}" style-hover="background:color-mix(in srgb, var(--color-text) 5%, transparent)">
                    <span style="font-size:13px;color:{{ f.color }};font-family:var(--font-heading);font-weight:800">{{ f.icon }}</span>
                    <span style="font-size:14.5px;color:{{ f.color }}">{{ f.name }}<span style="font-size:13px;color:color-mix(in srgb,var(--color-text) 48%,transparent)"> {{ f.meta }}</span></span>
                    <span style="font-size:12.5px;font-variant-numeric:tabular-nums;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ f.badge }}</span>
                  </div>
                </sc-for>
              </div>
            </sc-if>

            <div style="padding:16px 28px 8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
              <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ readerTitle }}</div>
              <div style="flex:1;min-width:80px;height:3px;background:color-mix(in srgb,var(--color-text) 12%,transparent)">
                <div style="height:3px;background:var(--color-accent);width:{{ readerProgress }}"></div>
              </div>
              <div style="font-size:12.5px;font-variant-numeric:tabular-nums;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ readerCaption }}</div>
            </div>

            <div ref="{{ readerRef }}" style="margin:0 28px 26px;padding:18px 20px;border:1px solid var(--color-divider);background:var(--color-surface);height:420px;overflow:auto;font-size:13.5px;line-height:1.7;white-space:pre-wrap;word-break:break-word">
              <sc-for list="{{ readerSegs }}" as="s" hint-placeholder-count="3">
                <span data-seg-active="{{ s.active }}" style="{{ s.style }}">{{ s.text }}</span>
              </sc-for>
              <sc-if value="{{ readerEmpty }}" hint-placeholder-val="{{ true }}">
                <span style="color:color-mix(in srgb,var(--color-text) 45%,transparent)">Choose a file and the document appears here, with the wording highlighted as it is found.</span>
              </sc-if>
            </div>
          </div>

          <div style="min-width:0">
            <sc-if value="{{ uploadHasChanges }}" hint-placeholder-val="{{ false }}">
              <div style="padding:18px 28px 4px">
                <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--color-accent-700);font-family:var(--font-heading);font-weight:800;margin-bottom:12px">What changed here</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px 18px">
                  <sc-for list="{{ uploadChanges }}" as="c" hint-placeholder-count="4">
                    <div style="display:flex;align-items:baseline;gap:9px">
                      <span style="font-family:var(--font-heading);font-weight:800;font-size:22px;color:{{ c.color }};font-variant-numeric:tabular-nums">{{ c.delta }}</span>
                      <span style="font-size:14px;line-height:1.4">{{ c.label }}<span style="color:color-mix(in srgb,var(--color-text) 45%,transparent);font-size:12.5px"> {{ c.total }}</span></span>
                    </div>
                  </sc-for>
                </div>
                <button onClick="{{ goDeadlines }}" class="btn btn-secondary" style="margin-top:16px">See the dates it added</button>
              </div>
            </sc-if>

            <div style="padding:18px 28px 8px">
              <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:color-mix(in srgb,var(--color-text) 55%,transparent)">{{ uploadLogTitle }}</div>
            </div>
            <div style="margin:0 28px 26px;height:{{ logHeight }};overflow:auto">
              <sc-for list="{{ uploadLog }}" as="l" hint-placeholder-count="6">
                <div style="display:grid;grid-template-columns:20px 1fr;gap:10px;padding:4px 0 4px {{ l.indent }};align-items:baseline">
                  <span style="font-size:12px;color:{{ l.color }};font-family:var(--font-heading);font-weight:800">{{ l.icon }}</span>
                  <div>
                    <div style="font-size:14px;line-height:1.45;color:{{ l.color }}">{{ l.text }}<span style="color:color-mix(in srgb,var(--color-text) 45%,transparent);font-size:12.5px"> {{ l.detail }}</span></div>
                    <sc-if value="{{ l.hasQuote }}" hint-placeholder-val="{{ false }}">
                      <div style="border-left:2px solid var(--color-accent-200);padding:1px 0 1px 9px;margin-top:3px;font-size:13px;font-style:italic;color:color-mix(in srgb,var(--color-text) 68%,transparent)">&ldquo;{{ l.quote }}&rdquo;</div>
                    </sc-if>
                  </div>
                </div>
              </sc-for>
              <sc-if value="{{ uploadLogEmpty }}" hint-placeholder-val="{{ true }}">
                <sc-for list="{{ pipelineSteps }}" as="p" hint-placeholder-count="5">
                  <div style="display:grid;grid-template-columns:30px 1fr;gap:12px;padding:11px 0;border-top:1px solid var(--color-divider)">
                    <div style="font-family:var(--font-heading);font-weight:800;font-size:13px;color:color-mix(in srgb,var(--color-text) 40%,transparent)">{{ p.n }}</div>
                    <div>
                      <div style="font-family:var(--font-heading);font-weight:800;font-size:14.5px;margin-bottom:2px">{{ p.title }}</div>
                      <div style="font-size:13.5px;line-height:1.5;color:color-mix(in srgb,var(--color-text) 68%,transparent)">{{ p.body }}</div>
                    </div>
                  </div>
                </sc-for>
              </sc-if>
            </div>
          </div>
        </div>
      </div>
    </sc-if>
"""


def replace_upload_view(text: str) -> str:
    """Swap the whole upload block for the live one."""
    start = text.index("<!-- ══════ UPLOAD ══════ -->")
    end = text.index("<!-- ══════ CONTRACTS ══════ -->")
    return text[:start] + UPLOAD_VIEW + "\n    " + text[end:]


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    text = replace_upload_view(strip_mock(SRC.read_text()))
    failures: list[str] = []
    for label, find, replace in patches:
        if find not in text:
            failures.append(label)
            continue
        text = text.replace(find, replace, 1)

    # The live data layer, appended to the design's own Component class.
    text = text.replace(
        "  componentDidMount() {",
        LIVE_METHODS + "\n  componentDidMount() {",
        1,
    )
    if LIVE_METHODS not in text:
        failures.append("live methods")

    text = text.replace("renderVals() {", "renderVals() {\n" + RENDER_PRELUDE, 1)
    if RENDER_PRELUDE not in text:
        failures.append("render prelude")

    anchor = "      roleFinance: !legal, roleLegal: legal,"
    if anchor not in text:
        failures.append("extra vals anchor")
    else:
        text = text.replace(anchor, EXTRA_VALS + anchor, 1)

    if failures:
        print("wire.py: these patches did not match the design:")
        for f in failures:
            print(f"  - {f}")
        print("\nThe design was probably re-exported. Update wire.py to match.")
        return 1

    OUT.write_text(text)
    print(f"wired -> {OUT} ({len(text):,} chars, {len(patches)} patches)")
    return _syntax_check(text)


def _syntax_check(text: str) -> int:
    """Parse the wired script with node, if it is available.

    A patch that concatenates a `//` comment onto one line silently comments out
    everything after it, which is how this script broke the first time.
    """
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if not node:
        print("  (node not found — skipping syntax check)")
        return 0
    i = text.index("data-dc-script")
    start = text.index(">", i) + 1
    script = text[start : text.rindex("</script>")]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
        tmp.write("(function(){\nclass DCLogic{};\n" + script + "\n})();")
        path = tmp.name
    result = subprocess.run([node, "--check", path], capture_output=True, text=True)
    if result.returncode != 0:
        print("  SYNTAX ERROR in the wired script:")
        print(result.stderr[:900])
        return 1
    print("  syntax check passed")
    return 0


LIVE_METHODS = """
  // ── WIRED: live data ────────────────────────────────────────────────
  loadModel() {
    const self = this;
    return fetch(API + '/ui/model').then(function (r) { return r.json(); }).then(function (m) {
      CONTRACTS = m.contracts; OBLIGATIONS = m.obligations; FINDINGS = m.findings;
      GAPS = m.gaps; INJECTIONS = m.injections; EVAL = m.eval;
      COVERAGE = m.coverage; COVERAGE_COLS = m.coverageCols;
      UPLOAD_CHECKS = m.uploadChecks; STATS = m.stats || {};
      EVAL_STATS = m.evalStats || {}; AS_OF = m.asOf;
      self.setState({ loaded: true, err: null });
      self.loadExit();
    }).catch(function (e) { self.setState({ err: String(e) }); });
  }

  loadExit() {
    const self = this, st = this.state;
    const cid = st.cid || (CONTRACTS[0] && CONTRACTS[0].id);
    if (!cid || !st.exitDate) return;
    const body = new FormData();
    body.append('exit_date', st.exitDate);
    fetch(API + '/contracts/' + cid + '/termination-cost', { method: 'POST', body: body })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        const lines = (d.line_items || []).map(function (i) {
          return { label: i.label, amount: i.amount, detail: i.detail || '',
                   quote: i.clause_span ? i.clause_span.quote : '' };
        });
        self.setState({ exitCost: { cid: cid, lines: lines, total: d.total,
                                    notes: d.notes || [] } });
      }).catch(function () {});
  }

  runAsk(question) {
    // Streams the agent's work as it happens rather than waiting for the
    // answer. The tool calls and what they retrieved are the interesting part.
    const self = this;
    const q = (question || '').trim();
    if (!q || this.state.asking) return;
    if (this._es) { this._es.close(); this._es = null; }
    if (this._tick) { clearInterval(this._tick); }

    const started = Date.now();
    this.setState({
      asking: true, askQuestion: q, askResult: null,
      askActivity: [], askPlan2: [], askPhase: 'Planning', askElapsed: '0.0s',
    });
    this._tick = setInterval(function () {
      if (!self.state.asking) return;
      self.setState({ askElapsed: ((Date.now() - started) / 1000).toFixed(1) + 's' });
    }, 100);

    const push = function (row) {
      self.setState({ askActivity: (self.state.askActivity || []).concat([row]) });
    };
    const stop = function () {
      if (self._es) { self._es.close(); self._es = null; }
      if (self._tick) { clearInterval(self._tick); self._tick = null; }
      self.setState({ asking: false });
    };

    const es = new EventSource(API + '/agent/stream?question=' + encodeURIComponent(q));
    this._es = es;

    es.onmessage = function (message) {
      let e;
      try { e = JSON.parse(message.data); } catch (err) { return; }

      if (e.type === 'planning') {
        self.setState({ askPhase: 'Working out how to answer' });
      } else if (e.type === 'plan') {
        self.setState({ askPhase: 'Looking things up', askPlan2: e.steps || [] });
        (e.steps || []).forEach(function (step, i) {
          push({ icon: String(i + 1).padStart(2, '0'), text: step, detail: '',
                 quote: '', hasQuote: false, indent: '0px',
                 color: 'color-mix(in srgb,var(--color-text) 60%,transparent)' });
        });
      } else if (e.type === 'thinking') {
        self.setState({ askPhase: 'Deciding what to look up next' });
      } else if (e.type === 'throttled') {
        self.setState({ askPhase: 'Waiting ' + e.seconds + 's for the rate limit' });
        push({ icon: '~', text: 'Rate limit reached',
               detail: 'waiting ' + e.seconds + 's — free tier is 8,000 tokens a minute',
               quote: '', hasQuote: false, indent: '0px',
               color: 'var(--color-accent-700)' });
      } else if (e.type === 'tool_start') {
        self.setState({ askPhase: 'Running ' + e.tool });
        const args = Object.keys(e.args || {})
          .map(function (k) { return k + '=' + JSON.stringify(e.args[k]); })
          .join(' ');
        push({ icon: '\u2192', text: e.tool, detail: args, quote: '',
               hasQuote: false, indent: '0px', color: 'var(--color-text)' });
      } else if (e.type === 'tool_end') {
        push({ icon: '\u2713', text: e.summary || 'done', detail: '',
               quote: '', hasQuote: false, indent: '22px',
               color: e.ok ? 'var(--color-accent-700)' : 'var(--color-accent)' });
        (e.retrieved || []).forEach(function (r) {
          push({ icon: '\u00b7',
                 text: (r.contract ? r.contract + ' — ' : '') + (r.title || ''),
                 detail: '', quote: r.quote || '', hasQuote: !!r.quote,
                 indent: '22px',
                 color: 'color-mix(in srgb,var(--color-text) 70%,transparent)' });
        });
      } else if (e.type === 'answer') {
        self.setState({
          askPhase: 'Done',
          askResult: { question: q, answer: e.answer, citations: e.citations || [],
                       sufficient: e.sufficient, missing: '',
                       plan: self.state.askPlan2 || [],
                       steps: [], tables: e.tables || [] },
        });
      } else if (e.type === 'exhausted') {
        push({ icon: '!', text: 'Ran out of lookups', detail: '', quote: '',
               hasQuote: false, indent: '0px', color: 'var(--color-accent)' });
      } else if (e.type === 'error') {
        self.setState({ askResult: { question: q, answer: e.message, citations: [],
                                     sufficient: false, missing: '', plan: [],
                                     steps: [], tables: [] } });
        stop();
      } else if (e.type === 'degraded') {
        self.setState({ askResult: { question: q, answer: e.answer,
          citations: e.citations || [], sufficient: false, missing: e.missing || '',
          plan: [], steps: [], tables: [], degraded: true },
          health: e.model || self.state.health });
      } else if (e.type === 'done') {
        self.loadHealth();
        stop();
      }
    };
    es.onerror = function () {
      if (!self.state.askResult) {
        self.setState({ askResult: { question: q,
          answer: 'Lost the connection to the analysis API.', citations: [],
          sufficient: false, missing: '', plan: [], steps: [], tables: [] } });
      }
      stop();
    };
  }

  loadHealth() {
    const self = this;
    fetch(API + '/health/model')
      .then(function (r) { return r.json(); })
      .then(function (h) { self.setState({ health: h }); })
      .catch(function () {
        self.setState({ health: { status: 'error', usable: false,
          message: 'Cannot reach the analysis API.' } });
      });
  }

  loadCalendar() {
    const self = this;
    if (this.state.calendar) return;
    fetch(API + '/calendar')
      .then(function (r) { return r.json(); })
      .then(function (d) { self.setState({ calendar: d }); })
      .catch(function () {});
  }

  onPickFile(e) {
    const self = this;
    const files = e && e.target && e.target.files;
    if (!files || !files.length) return;

    const body = new FormData();
    for (let i = 0; i < files.length; i++) body.append('files', files[i]);
    body.append('our_role', 'buyer');

    this.setState({
      uploading: true, uploadMsg: 'Reading ' + files.length + ' document(s)…',
      upFiles: [], upLog: [], upChanges: null, upText: '', upSpans: [],
      upRead: null, upActive: null, upPhase: 'Reading',
    });

    // Events are replayed on a short timer rather than applied the instant
    // they arrive. A cached document analyses in under a second, and a reader
    // who sees nothing happen learns nothing about what happened.
    const pending = [];
    let draining = false;
    const drain = function () {
      if (!pending.length) { draining = false; return; }
      draining = true;
      self.applyUploadEvent(pending.shift());
      setTimeout(drain, pending.length > 60 ? 8 : 55);
    };
    const queue = function (event) {
      pending.push(event);
      if (!draining) drain();
    };

    fetch(API + '/contracts/stream', { method: 'POST', body: body })
      .then(function (response) {
        if (!response.ok || !response.body) throw new Error('upload failed');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        const pump = function () {
          return reader.read().then(function (chunk) {
            if (chunk.done) return;
            buffer += decoder.decode(chunk.value, { stream: true });
            const lines = buffer.split('\\n');
            buffer = lines.pop();
            lines.forEach(function (line) {
              if (!line.indexOf('data: ')) {
                try { queue(JSON.parse(line.slice(6))); } catch (err) {}
              }
            });
            return pump();
          });
        };
        return pump();
      })
      .catch(function (err) {
        self.setState({ uploading: false,
                        uploadMsg: 'Could not analyse: ' + err.message });
      });
  }

  applyUploadEvent(ev) {
    const st = this.state;
    const log = function (self, row) {
      self.setState({ upLog: (self.state.upLog || []).concat([row]) });
    };

    if (ev.type === 'file_start') {
      const files = (st.upFiles || []).concat([{
        name: ev.filename, status: 'reading', clauses: 0, findings: 0,
        deadlines: 0, changes: null, contract_id: null }]);
      this.setState({ upFiles: files, upActive: files.length - 1,
                      upText: '', upSpans: [], upRead: null,
                      upPhase: 'Reading ' + ev.filename });
      log(this, { icon: '\u25b8', text: ev.filename,
                  detail: 'file ' + (ev.index + 1) + ' of ' + ev.total,
                  color: 'var(--color-text)', indent: '0px', hasQuote: false });

    } else if (ev.type === 'ingested') {
      this.setState({ upText: ev.text });
      log(this, { icon: '\u00b7', text: 'Read the text',
                  detail: ev.chars.toLocaleString() + ' characters, ' +
                          ev.pages + ' page(s)' + (ev.used_ocr ? ', via OCR' : ''),
                  color: 'color-mix(in srgb,var(--color-text) 70%,transparent)',
                  indent: '20px', hasQuote: false });

    } else if (ev.type === 'firewall') {
      const bad = ev.quarantined;
      log(this, { icon: bad ? '!' : '\u00b7',
                  text: bad ? 'Hidden instructions found — held back'
                            : 'Checked for tampering',
                  detail: bad ? ev.indicators.length + ' payload(s)' : 'clean',
                  color: bad ? 'var(--color-accent)'
                             : 'color-mix(in srgb,var(--color-text) 70%,transparent)',
                  indent: '20px', hasQuote: false });

    } else if (ev.type === 'reading') {
      this.setState({ upRead: { start: ev.start, end: ev.end },
                      upPhase: 'Reading part ' + ev.chunk + ' of ' + ev.chunks });

    } else if (ev.type === 'throttled') {
      this.setState({ upPhase: 'Waiting ' + ev.seconds + 's for the rate limit' });
      log(this, { icon: '~', text: 'Rate limit reached',
                  detail: 'waiting ' + ev.seconds + 's — the free tier allows '
                          + '8,000 tokens a minute',
                  color: 'var(--color-accent-700)', indent: '20px',
                  hasQuote: false });

    } else if (ev.type === 'clause') {
      this.setState({ upSpans: (st.upSpans || []).concat([
        { start: ev.start, end: ev.end }]) });
      const files = (st.upFiles || []).slice();
      if (st.upActive != null && files[st.upActive]) {
        files[st.upActive].clauses += 1;
        this.setState({ upFiles: files });
      }
      log(this, { icon: '\u00b7',
                  text: ev.clause_type.replace(/_/g, ' '),
                  detail: 'characters ' + ev.start + '\u2013' + ev.end,
                  quote: ev.quote, hasQuote: true, indent: '20px',
                  color: 'color-mix(in srgb,var(--color-text) 78%,transparent)' });

    } else if (ev.type === 'verified') {
      log(this, { icon: '\u2713', text: 'Every quote checked against the document',
                  detail: ev.kept + ' kept, ' + ev.discarded + ' discarded',
                  color: 'var(--color-accent-700)', indent: '20px',
                  hasQuote: false });

    } else if (ev.type === 'deadline') {
      const files = (st.upFiles || []).slice();
      if (st.upActive != null && files[st.upActive]) {
        files[st.upActive].deadlines += 1;
        this.setState({ upFiles: files });
      }
      log(this, { icon: '\u2192', text: 'Added ' + ev.due + ' to the calendar',
                  detail: ev.kind + ', ' + ev.days + ' days away',
                  color: 'var(--color-accent-700)', indent: '20px',
                  hasQuote: false });

    } else if (ev.type === 'finding') {
      const files = (st.upFiles || []).slice();
      if (st.upActive != null && files[st.upActive]) {
        files[st.upActive].findings += 1;
        this.setState({ upFiles: files });
      }
      log(this, { icon: ev.evidenced ? '\u00b7' : '\u2205', text: ev.title,
                  detail: ev.severity + (ev.evidenced ? '' : ' · nothing to quote'),
                  color: (ev.severity === 'critical' || ev.severity === 'high')
                         ? 'var(--color-accent)'
                         : 'color-mix(in srgb,var(--color-text) 70%,transparent)',
                  indent: '20px', hasQuote: false });

    } else if (ev.type === 'changes') {
      const files = (st.upFiles || []).slice();
      if (st.upActive != null && files[st.upActive]) {
        files[st.upActive].changes = ev;
        files[st.upActive].contract_id = ev.contract_id;
        files[st.upActive].status = 'done';
      }
      this.setState({ upFiles: files, upChanges: ev });

    } else if (ev.type === 'file_error') {
      const files = (st.upFiles || []).slice();
      if (st.upActive != null && files[st.upActive]) files[st.upActive].status = 'error';
      this.setState({ upFiles: files });
      log(this, { icon: '!', text: 'Could not analyse ' + ev.filename,
                  detail: ev.message, color: 'var(--color-accent)',
                  indent: '0px', hasQuote: false });

    } else if (ev.type === 'all_done') {
      this.setState({ uploading: false, upRead: null, upPhase: 'Done',
                      uploadMsg: 'Analysed. ' + ev.contracts +
                                 ' contracts, ' + ev.gaps + ' flow-down gaps.' });
      this.loadModel();
      this.setState({ calendar: null });

    } else if (ev.type === 'error') {
      this.setState({ uploading: false, uploadMsg: ev.message });
    }
  }

  scrollReader() {
    const pane = this.readerEl;
    if (!pane) return;
    const el = pane.querySelector('[data-seg-active="1"]');
    if (el) pane.scrollTop = Math.max(0, el.offsetTop - pane.clientHeight / 3);
  }

  selectUploaded(index) {
    const file = (this.state.upFiles || [])[index];
    this.setState({ upActive: index, upChanges: file ? file.changes : null });
  }
"""

RENDER_PRELUDE = """    // ── WIRED: nothing renders until the API has answered
    if (!this.state.loaded) {
      return { isDeadlines: true, isUpload: false, isContracts: false,
               isDetail: false, isFindings: false, isCoverage: false,
               isChain: false, isSecurity: false, isEval: false,
               deadlines: [], deadlinesEmpty: true, drawerOpen: false,
               navGroups: [], contractRows: [], allFindings: [], gapRows: [],
               injections: [], evalRows: [], coverageRows: [], coverageCols: [],
               riskAxes: [], detailClauses: [], detailFindings: [], detailTabs: [],
               docSegs: [], exitLines: [], pipelineSteps: [], uploadChecks: [],
               isAsk: false, askHasAnswer: false, askCitations: [],
               askIsRunning: false, askHasActivity: false, askActivity: [],
               askPhase: '', askElapsed: '', askDegraded: false,
               degradedTitle: '', degradedBody: '',
               modelLabel: 'Model idle', modelColor: 'transparent',
               modelDetail: '',
               askSuggestions: [], askQuestion: '', askButtonLabel: 'Ask',
               askHasPlan: false, askPlan: [], askSteps: [],
               uploadHasFiles: false, uploadFiles: [], readerSegs: [],
               readerEmpty: true, uploadHasChanges: false, uploadChanges: [],
               uploadLog: [], uploadLogEmpty: true, logHeight: 'auto',
               readerTitle: '', readerCaption: '', readerProgress: '0%',
               uploadHeadline: 'Add a document', uploadSubhead: '',
               uploadLogTitle: '',
               isCalendar: false, calendarMonths: [], calActionable: '—',
               calDocuments: '—', calComputed: '—', calWritten: '—',
               asOfLabel: this.state.err ? 'Cannot reach the analysis API' : 'Loading…',
               deadlineHeadline: this.state.err ? 'The analysis API is not responding'
                                                : 'Reading your contracts…',
               deadlineSubhead: this.state.err
                 ? String(this.state.err) + ' — is the server running?'
                 : 'Fetching verified clauses and computed deadlines.',
               statMissed: '—', statMissedNote: '', stat90: '—', stat90Note: '',
               statHeld: '—', statHeldNote: '', findingsCount: '' };
    }
"""

EXTRA_VALS = """      // ── WIRED: values that used to be written into the design
      statHeld: String(STATS.held || 0),
      statHeldNote: (STATS.held || 0) === 0
        ? 'Nothing is being withheld. Every document passed the tampering checks.'
        : 'Held back for a person to look at. We do not put dates in your calendar from a document we do not trust.',
      findingsIntro: (STATS.findings || 0) + ' things across ' +
        (STATS.contracts || 0) + ' contracts. ' + (STATS.absenceFindings || 0) +
        ' of them are about wording that is not in the document at all — the kind of gap a folder of PDFs will never show you.',
      evalIntro: 'Measured on ' + (EVAL_STATS.documents || 0) +
        ' held-out contracts against a hand-written gold standard, using ' +
        (EVAL_STATS.model || 'the extraction model') + '. Every clause type we score is here, including the ones we are bad at.',
      evalGrounding: EVAL_STATS.groundingRate != null
        ? (EVAL_STATS.groundingRate * 100).toFixed(1) + '%' : '—',
      evalGroundingNote: (EVAL_STATS.discarded || 0) === 0
        ? 'every quote matched the source document exactly'
        : EVAL_STATS.discarded + ' were thrown away before you saw them',
      evalZeroNote: 'a claim without matching text in your document cannot be displayed',
      evalLatency: EVAL_STATS.medianLatency != null ? EVAL_STATS.medianLatency + 's' : '—',
      evalLatencyNote: EVAL_STATS.medianLatency != null
        ? 'median per contract' : 'scored from cache, so not timed',
      evalSpans: String(EVAL_STATS.spansChecked || 0),
      evalSpansNote: 'all ' + (EVAL_STATS.spansExact || 0) + ' matched the document exactly',
      secHeadline: (STATS.injections || 0) === 0
        ? 'No hidden instructions found in any document'
        : 'Someone hid ' + STATS.injections + ' instruction' +
          ((STATS.injections === 1) ? '' : 's') + ' in a supplier document',
      secFileLabel: (INJECTIONS[0] && INJECTIONS[0].where) || 'no flagged document',
      lastUploadLabel: (INJECTIONS[0] && INJECTIONS[0].where) || 'nothing uploaded yet',
      uploadAlertBody: (STATS.injections || 0) === 0
        ? 'Nothing suspicious in the documents processed so far.'
        : STATS.injections + ' pieces of hidden text were trying to tell our reader what to conclude. We kept that document out of your schedule and flagged it for Security.',
      chainHeadline: (STATS.gaps || 0) === 0
        ? 'Everything you promised is covered by what you were given'
        : 'You promised your customers more than your suppliers promised you',
      chainSubhead: (STATS.gaps || 0) + ' differences across ' + (STATS.contracts || 0) +
        ' contracts, read together. Every figure below is quoted from one of them; the differences are arithmetic.',
      // ── WIRED: upload, watched as it happens
      uploadStatus: st.uploadMsg || '',
      onPickFile: this.onPickFile.bind(this),
      uploadHeadline: st.uploading ? st.upPhase || 'Reading'
        : (st.upFiles && st.upFiles.length ? 'Here is what changed'
           : 'Add a document'),
      uploadSubhead: st.uploading
        ? 'Watch the wording being found. Nothing reaches your calendar until it has been checked against the document.'
        : (st.upFiles && st.upFiles.length
           ? 'Pick a document on the left to see what it added.'
           : 'Drop in one file or several. Each becomes its own contract, and you can watch it being read.'),
      goDeadlines: function () { self.setState({ view: 'deadlines' }); },

      uploadHasFiles: !!(st.upFiles && st.upFiles.length),
      uploadFiles: (st.upFiles || []).map(function (f, i) {
        const active = st.upActive === i;
        const icon = f.status === 'done' ? '\u2713'
                   : (f.status === 'error' ? '!' : '\u25b8');
        return {
          name: f.name,
          meta: f.status === 'reading' ? ' · reading' : '',
          icon: icon,
          badge: f.status === 'done'
            ? f.clauses + ' clauses · ' + f.deadlines + ' dates' : '',
          color: f.status === 'error' ? 'var(--color-accent)' : 'var(--color-text)',
          bg: active ? 'color-mix(in srgb, var(--color-text) 6%, transparent)'
                     : 'transparent',
          edge: active ? 'var(--color-accent)' : 'transparent',
          select: function () { self.selectUploaded(i); },
        };
      }),

      readerTitle: st.uploading ? 'Reading' : 'The document',
      readerCaption: st.upRead
        ? 'characters ' + st.upRead.start + '\u2013' + st.upRead.end
        : ((st.upSpans || []).length ? (st.upSpans || []).length + ' passages found' : ''),
      readerProgress: (function () {
        const len = (st.upText || '').length;
        if (!len || !st.upRead) return st.upText ? '100%' : '0%';
        return Math.min(100, Math.round(st.upRead.end / len * 100)) + '%';
      })(),
      readerEmpty: !(st.upText || '').length,
      readerRef: function (el) { self.readerEl = el; },
      readerSegs: buildReaderSegs(st),

      uploadHasChanges: !!st.upChanges,
      uploadChanges: st.upChanges ? CHANGE_ROWS.map(function (row) {
        const d = (st.upChanges.delta || {})[row.key] || 0;
        return {
          label: row.label,
          delta: (d > 0 ? '+' : '') + d,
          total: '(' + ((st.upChanges.totals || {})[row.key] || 0) + ' in all)',
          color: d > 0 ? 'var(--color-accent)'
                       : 'color-mix(in srgb,var(--color-text) 40%,transparent)',
        };
      }) : [],

      uploadLogTitle: st.uploading ? 'What it is doing'
                                   : ((st.upLog || []).length ? 'What it did'
                                      : 'What happens when you add one'),
      uploadLog: st.upLog || [],
      uploadLogEmpty: !((st.upLog || []).length),
      logHeight: (st.upLog || []).length ? '420px' : 'auto',

      // ── WIRED: model health
      modelLabel: MODEL_LABEL(st),
      modelColor: MODEL_COLOR(st),
      modelDetail: ((st.health && (st.health.detail || st.health.message)) || ''),
      askDegraded: !!(st.askResult && st.askResult.degraded),
      degradedTitle: (st.health && st.health.status === 'rate_limited')
        ? 'The model is rate limited' : 'The model did not load',
      degradedBody: (function () {
        const h = st.health || {};
        const wait = h.retry_after ? ' Try again in about '
                     + Math.round(h.retry_after) + ' seconds.' : '';
        const why = h.status === 'no_key'
          ? 'No model is connected, so nothing can be summarised.'
          : (h.status === 'rate_limited'
             ? 'The free tier allows 8,000 tokens a minute and that has been used up.'
             : 'The model is not responding right now.');
        return why + ' What you can see below is a plain word match over the '
             + 'verified records — nothing has been read or summarised, and every '
             + 'passage is quoted from your documents.' + wait;
      })(),

      // ── WIRED: Calendar
      isCalendar: st.view === 'calendar',
      calActionable: String(CAL_SUM(st).actionable || 0),
      calActionableNote: (CAL_SUM(st).overdue || 0) > 0
        ? CAL_SUM(st).overdue + ' already past their date'
        : 'nothing has gone past its date',
      calDocuments: String(CAL_SUM(st).documents || 0),
      calComputed: String(CAL_SUM(st).computed || 0),
      calWritten: String(CAL_SUM(st).written_in_documents || 0),
      calendarMonths: ((st.calendar && st.calendar.months) || []).map(function (m) {
        return {
          label: m.label,
          note: m.count + ' date' + (m.count === 1 ? '' : 's')
                + (m.actionable ? ' · ' + m.actionable + ' needing a decision' : ''),
          events: m.events.map(function (e) {
            const urgent = e.actionable && e.days <= 30;
            return {
              dateLabel: self.fmt(e.date),
              dateColor: e.overdue ? 'var(--color-accent)'
                         : (urgent ? 'var(--color-accent)' : 'var(--color-text)'),
              sourceLabel: e.source === 'system' ? 'Received'
                           : (e.source === 'computed' ? 'Worked out' : 'In the text'),
              sourceColor: e.source === 'computed' ? 'var(--color-accent-700)'
                           : 'color-mix(in srgb,var(--color-text) 45%,transparent)',
              title: e.label + ' — ' + e.contract,
              detail: e.detail || '',
              hasQuote: !!e.quote,
              quote: e.quote || '',
              daysLabel: e.overdue ? Math.abs(e.days) + 'd ago'
                         : (e.days === 0 ? 'today' : 'in ' + e.days + 'd'),
              open: function () {
                if (e.contract_id) self.setState({ view: 'detail', cid: e.contract_id });
              }
            };
          })
        };
      }),

      // ── WIRED: Ask
      isAsk: st.view === 'ask',
      askQuestion: st.askQuestion || '',
      setAskQuestion: function (e) {
        self.setState({ askQuestion: e && e.target ? e.target.value : '' });
      },
      submitAsk: function () { self.runAsk(st.askQuestion); },
      askButtonLabel: st.asking ? 'Reading the records…' : 'Ask',
      askSuggestions: ASK_SUGGESTIONS.map(function (q) {
        return { label: q, go: function () { self.runAsk(q); } };
      }),
      askIsRunning: !!st.asking,
      askPhase: st.askPhase || 'Working',
      askElapsed: st.askElapsed || '',
      askHasActivity: !!(st.askActivity && st.askActivity.length),
      askActivity: st.askActivity || [],
      askHasPlan: !!(st.askResult && st.askResult.plan && st.askResult.plan.length),
      askPlan: ((st.askResult && st.askResult.plan) || []).map(function (t, i) {
        return { n: String(i + 1).padStart(2, '0'), text: t };
      }),
      askSteps: ((st.askResult && st.askResult.steps) || []).map(function (s) {
        return {
          label: s.tool + (s.summary ? ' · ' + s.summary : ''),
          style: s.ok
            ? 'background:color-mix(in srgb,var(--color-text) 8%,transparent);color:var(--color-text);font-size:11.5px;padding:3px 10px'
            : 'background:var(--color-accent-100);color:var(--color-accent-700);font-size:11.5px;padding:3px 10px'
        };
      }),
      askHasAnswer: !!st.askResult,
      askQuestionEcho: st.askResult ? st.askResult.question : '',
      askAnswer: st.askResult ? st.askResult.answer : '',
      askUnanswerable: !!(st.askResult && st.askResult.sufficient === false),
      askMissing: st.askResult
        ? (st.askResult.missing
           ? 'The records do not cover ' + st.askResult.missing +
             '. Rather than fill that in from what contracts usually say, we left it out.'
           : 'The verified records do not answer this. Rather than fill the gap '
             + 'from what contracts usually say, we left it out.')
        : '',
      askMeta: st.askResult
        ? ((st.askResult.steps || []).filter(function (x) {
             return x.tool !== 'finish' && x.tool !== 'answer'; }).length
           + ' lookup(s) over the verified layer; '
           + (st.askResult.citations || []).length + ' record(s) cited.'
           + (st.askResult.stopped_early ? ' Stopped before finishing.' : ''))
        : '',
      askCitationsLabel: st.askResult && st.askResult.citations.length
        ? 'What this rests on' : 'Nothing was cited',
      askCitations: (st.askResult ? st.askResult.citations : []).map(function (c) {
        return {
          contract: c.contract, kind: c.kind, title: c.title,
          kindColor: c.kind === 'absence' ? 'var(--color-accent)'
                     : 'color-mix(in srgb,var(--color-text) 50%,transparent)',
          hasQuote: !!c.quote, quote: c.quote || '',
          isAbsence: !c.quote,
          offsets: c.file && c.start != null
            ? 'characters ' + c.start + '–' + c.end + ' of ' + c.file : '',
          open: function () {
            if (c.contract_id) self.setState({ view: 'detail', cid: c.contract_id });
          }
        };
      }),
"""


if __name__ == "__main__":
    raise SystemExit(main())
