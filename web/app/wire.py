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
    "upload last-doc label",
    "tessellate-msa-2026.pdf · 11 pages · 18 Apr 2026",
    "{{ lastUploadLabel }}",
)
patch(
    "upload alert body",
    "Three pieces of hidden text in this document were trying to tell our reader "
    "what to conclude. We kept it out of your schedule and flagged it for Nina "
    "Boateng in Legal.",
    "{{ uploadAlertBody }}",
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
    "askQuestion:'', asking:false, askResult:null, calendar:null };",
)
patch(
    "fetch on mount",
    "componentDidMount() { this.scrollToSel(); }",
    "componentDidMount() { this.scrollToSel(); this.loadModel(); }",
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
patch(
    "upload control",
    '<button class="btn btn-primary" style="margin-top:22px">Choose a file</button>',
    '<input type="file" onChange="{{ onPickFile }}" '
    'style="margin-top:22px;font:inherit;display:block" />\n'
    '              <div style="margin-top:12px;font-size:14px;'
    'color:color-mix(in srgb,var(--color-text) 70%,transparent)">{{ uploadStatus }}</div>',
)



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


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    text = strip_mock(SRC.read_text())
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
    const self = this;
    const q = (question || '').trim();
    if (!q || this.state.asking) return;
    this.setState({ asking: true, askQuestion: q });
    const body = new FormData();
    body.append('question', q);
    fetch(API + '/agent/ask', { method: 'POST', body: body })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error(res.d.detail || 'ask failed');
        self.setState({ asking: false, askResult: res.d });
      })
      .catch(function (err) {
        self.setState({ asking: false, askResult: {
          question: q, answer: 'Could not reach the analysis API: ' + err.message,
          citations: [], sufficient: false, missing: '', considered: 0 } });
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
    const file = e && e.target && e.target.files && e.target.files[0];
    if (!file) return;
    this.setState({ uploading: true, uploadMsg: 'Reading ' + file.name + '…' });
    const body = new FormData();
    body.append('files', file);
    body.append('title', file.name);
    body.append('counterparty', file.name.replace(/[-_.].*$/, ''));
    body.append('our_role', 'buyer');
    fetch(API + '/contracts', { method: 'POST', body: body })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error(res.d.detail || 'upload failed');
        self.setState({ uploading: false,
                        uploadMsg: 'Analysed ' + file.name + '. ' +
                                   res.d.clauses.length + ' clauses, ' +
                                   res.d.findings.length + ' findings.' });
        self.loadModel();
      })
      .catch(function (err) {
        self.setState({ uploading: false, uploadMsg: 'Could not analyse: ' + err.message });
      });
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
               askSuggestions: [], askQuestion: '', askButtonLabel: 'Ask',
               askHasPlan: false, askPlan: [], askSteps: [],
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
      uploadStatus: this.state.uploadMsg || '',
      onPickFile: this.onPickFile.bind(this),

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
