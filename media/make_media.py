"""Generate submission media: slide HTMLs (-> PDF, PNGs, video) and a cover."""
from pathlib import Path

OUT = Path(__file__).parent
W, H = 1280, 720

CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
.slide { width:1280px; height:720px; background:#0e1320; color:#e8ecf4;
  font-family:'Helvetica Neue',Arial,sans-serif; padding:70px 90px;
  display:flex; flex-direction:column; page-break-after:always; position:relative; overflow:hidden; }
.slide::before { content:''; position:absolute; left:0; top:0; bottom:0; width:8px; background:#7dd956; }
.kicker { color:#7dd956; font-size:22px; letter-spacing:3px; text-transform:uppercase; font-weight:600; margin-bottom:18px; }
h1 { font-size:64px; line-height:1.08; font-weight:700; }
h2 { font-size:46px; line-height:1.15; font-weight:700; margin-bottom:34px; }
.sub { color:#9fb0c8; font-size:26px; margin-top:22px; line-height:1.45; }
ul { list-style:none; margin-top:6px; }
li { font-size:26px; line-height:1.42; margin-bottom:20px; padding-left:34px; position:relative; color:#dbe3f0; }
li::before { content:'>'; position:absolute; left:0; color:#7dd956; font-weight:700; }
li b, .sub b { color:#fff; }
code { font-family:'SF Mono',Menlo,monospace; background:#1a2234; color:#8fd0ff; padding:2px 10px; border-radius:6px; font-size:24px; }
.foot { position:absolute; bottom:34px; left:98px; right:90px; display:flex; justify-content:space-between;
  color:#5c6c85; font-size:19px; font-family:Menlo,monospace; }
table { border-collapse:collapse; margin-top:8px; }
td,th { font-size:25px; padding:13px 30px; border-bottom:1px solid #2a3550; text-align:left; }
th { color:#7dd956; font-weight:600; }
td.win { color:#7dd956; font-weight:700; }
.big { font-size:34px; color:#fff; margin-top:26px; line-height:1.35; }
.spacer { flex:1; }
"""

FOOT = "<div class='foot'><span>github.com/bogdan-lmk/veriroute</span><span>AMD Developer Hackathon: ACT II — Track 1</span></div>"

slides = [
    # 1 — title
    f"""<div class='slide' style='justify-content:center'>
      <div class='kicker'>AMD Developer Hackathon: ACT II · Track 1</div>
      <h1>veriroute</h1>
      <div class='sub' style='font-size:34px'>Verification-driven, token-efficient routing agent</div>
      <div class='sub'>Verified answers cost <b>zero tokens</b>. Only proven failures escalate to Fireworks.</div>
      <div class='spacer'></div>
      <div class='sub' style='font-size:22px'>Bogdan Lameko · <code>ghcr.io/bogdan-lmk/veriroute</code></div>
      {FOOT}</div>""",
    # 2 — problem
    f"""<div class='slide'>
      <div class='kicker'>The problem</div>
      <h2>Fewest tokens, above the accuracy gate</h2>
      <ul>
        <li>Scoring: pass an LLM-judge accuracy gate, then <b>ascending total Fireworks tokens</b> — every wasted token is rank lost</li>
        <li>Reasoning models bill <b>hidden thinking as output tokens</b> — a tight cap returns empty answers, a loose one burns budget</li>
        <li>Prompt-based routers pay for a classification call on <b>every single task</b></li>
        <li>Half the field never scored at all: packaging, timeouts, malformed output</li>
      </ul>
      {FOOT}</div>""",
    # 3 — approach
    f"""<div class='slide'>
      <div class='kicker'>The approach</div>
      <h2>Harness engineering, not prompt magic</h2>
      <ul>
        <li><b>Deterministic policy table</b> routes every task — the LLM never decides its own escalation</li>
        <li><b>Verification by execution</b>: math via program-of-thought, codegen via generated self-tests — local answers ship only when code proves them</li>
        <li><b>Token-oriented model ranking</b>: non-thinking models first, accuracy costs the same capped tokens; Gemma ranks first when allowed</li>
        <li>Local inference is <b>zero tokens</b> — llama.cpp CPU-dispatch runtime already packaged in the image</li>
      </ul>
      {FOOT}</div>""",
    # 4 — guardrails
    f"""<div class='slide'>
      <div class='kicker'>Reliability by construction</div>
      <h2>Every failure mode maps to a guardrail</h2>
      <ul>
        <li>Stub-first <b>atomic results.json</b> before anything can fail; rewritten after every task — SIGKILL-safe (integration-tested)</li>
        <li><code>model in ALLOWED_MODELS</code> asserted <b>before any network I/O</b> — disqualification is structurally impossible</li>
        <li>Hard token budget stop · 28s per-task cap · parallel drain at T-45s · exit 0 in every failure mode</li>
        <li><b>49 tests</b>; one change per submission, predictions logged vs leaderboard</li>
      </ul>
      {FOOT}</div>""",
    # 5 — measured
    f"""<div class='slide'>
      <div class='kicker'>Measured live on Fireworks (minimax-m3, 8 practice tasks)</div>
      <h2>Escalation output policy, chosen by data</h2>
      <table>
        <tr><th>Config</th><th>Tokens</th><th>Answers intact</th></tr>
        <tr><td>cap 300, no instruction</td><td>3,065</td><td>5/8 — truncated mid-answer</td></tr>
        <tr><td>terse + thinking cap 600</td><td>4,411</td><td>8/8, but paid retry burn</td></tr>
        <tr><td class='win'>terse + thinking cap 1000</td><td class='win'>3,573</td><td class='win'>8/8</td></tr>
      </table>
      <div class='big'>Terse suffix cuts simple-task completions 40–60%; thinking-aware caps end the empty-answer retry loop.</div>
      {FOOT}</div>""",
    # 6 — roadmap
    f"""<div class='slide'>
      <div class='kicker'>Roadmap</div>
      <h2>Every verified category → zero tokens</h2>
      <ul>
        <li>Local Qwen-class model behind code verification takes sentiment, NER, summarization, math &amp; codegen off the paid path</li>
        <li>Escalation remains only for proven failures and factual recall — bounded by the token budget</li>
        <li>Target: <b>beat 5,121 tokens</b> (current sole gate-passer) with a wide accuracy margin</li>
      </ul>
      <div class='spacer'></div>
      <div class='big'><code>ghcr.io/bogdan-lmk/veriroute:sub2</code> · <code>github.com/bogdan-lmk/veriroute</code></div>
      {FOOT}</div>""",
]

# per-slide HTML for screenshots
for i, body in enumerate(slides, 1):
    (OUT / f"slide{i}.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}"
        f"body{{width:{W}px;height:{H}px}}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )

# one document for PDF printing
(OUT / "deck.html").write_text(
    "<!doctype html><html><head><meta charset='utf-8'><style>"
    f"{CSS} @page {{ size: {W}px {H}px; margin: 0; }}"
    "</style></head><body>" + "".join(slides) + "</body></html>",
    encoding="utf-8",
)

# cover 1600x900
cover = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}
body{{width:1600px;height:900px}}
.slide{{width:1600px;height:900px;padding:100px 120px;justify-content:center}}
h1{{font-size:110px}} .kicker{{font-size:28px}}
</style></head><body><div class='slide'>
<div class='kicker'>AMD Developer Hackathon: ACT II · Track 1</div>
<h1>veriroute</h1>
<div class='sub' style='font-size:44px;margin-top:30px'>Verification-driven, token-efficient routing agent</div>
<div class='sub' style='font-size:32px'>Verified answers cost <b>zero tokens</b> — only proven failures escalate.</div>
</div></body></html>"""
(OUT / "cover.html").write_text(cover, encoding="utf-8")
print("generated", len(slides), "slides + deck.html + cover.html")
