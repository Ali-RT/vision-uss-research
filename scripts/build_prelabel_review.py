"""Build a local HTML review page for a CUV_ObjectCheck pre-label run.

Reads outputs/prelabel/<run>/boxes.csv (from prelabel_cuv_objectcheck.py),
draws every frame's box into an annotated copy, and writes review.html next to it:
one row per trace with a summary and the full frame strip. Per-trace
verdicts (ok / fix / reject) are kept in the browser's localStorage and
exported as CSV with the button at the top.

Box colors: green = detector class agrees with the tracelist object,
orange = box found but detector class disagrees, red frame border = no box.

    uv run python scripts/build_prelabel_review.py --run prelabel_cuv_v1
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def draw_thumb(row: pd.Series, out_path: Path) -> None:
    img = cv2.imread(row["frame_path"])
    if img is None:
        return
    if pd.notna(row["x0"]):
        color = (60, 200, 60) if row["det_agrees"] else (0, 150, 255)
        p1 = (int(row["x0"]), int(row["y0"]))
        p2 = (int(row["x1"]), int(row["y1"]))
        cv2.rectangle(img, p1, p2, color, 3)
        cv2.putText(img, f'{row["det_class"]} {row["det_conf"]:.2f}',
                    (p1[0], max(18, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 82])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="prelabel_cuv_v1")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "outputs" / "prelabel" / args.run
    df = pd.read_csv(run_dir / "boxes.csv")
    df["det_agrees"] = df["det_agrees"].astype(str).str.lower() == "true"
    thumbs_root = run_dir / "review_thumbs"

    traces = []
    for (side, tid), g in df.groupby(["camera", "sequence_id"], sort=True):
        g = g.sort_values("frame_idx")
        tdir = thumbs_root / f"{side}_{tid}"
        tdir.mkdir(parents=True, exist_ok=True)
        frames = []
        for _, r in g.iterrows():
            thumb = tdir / f'{int(r["frame_idx"]):05d}.jpg'
            if not thumb.exists():
                draw_thumb(r, thumb)
            state = ("none" if pd.isna(r["x0"])
                     else "agree" if r["det_agrees"] else "disagree")
            frames.append({"src": str(thumb.relative_to(run_dir)),
                           "f": int(r["frame_idx"]), "s": state,
                           "c": "" if pd.isna(r["det_class"]) else r["det_class"],
                           "p": None if pd.isna(r["det_conf"]) else round(float(r["det_conf"]), 2)})
        hits = g["x0"].notna()
        traces.append({
            "key": f"{side}_{tid}", "side": side, "trace": int(tid),
            "object": g["target_object"].iloc[0],
            "label": g["label_class"].iloc[0],
            "bin": g["height_bin"].iloc[0],
            "speed": g["speed"].iloc[0],
            "note": "" if pd.isna(g["trace_note"].iloc[0]) else g["trace_note"].iloc[0],
            "n": len(g), "hits": int(hits.sum()),
            "agree": int(g["det_agrees"].sum()),
            "conf": (round(float(g.loc[hits, "det_conf"].mean()), 2)
                     if hits.any() else None),
            "frames": frames,
        })

    page = TEMPLATE.replace("__RUN__", html.escape(args.run)).replace(
        "__DATA__", json.dumps(traces))
    out = run_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    print(f"{len(traces)} traces -> {out}")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pre-label Review</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e2dc;
--ok:#2e9d4a;--warn:#e08a00;--bad:#d23b3b;--acc:#2d5bd7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--card:#1f1f1d;
--ink:#ecece8;--mut:#9a9a93;--line:#33332f}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
padding:12px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
h1{font-size:16px;margin:0 12px 0 0}
button,select{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);
border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.stat{color:var(--mut);margin-left:auto}
main{padding:12px 16px;display:grid;gap:12px}
.tr{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.hd{display:flex;flex-wrap:wrap;gap:10px;align-items:baseline}
.hd b{font-size:15px}.m{color:var(--mut)}
.pill{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line)}
.low{color:var(--acc)}.high{color:var(--bad)}
.bar{height:6px;border-radius:3px;background:var(--line);width:120px;overflow:hidden;display:inline-block;vertical-align:middle}
.bar i{display:block;height:100%;background:var(--ok)}
.v{margin-left:auto;display:flex;gap:4px}
.v button.ok.on{background:var(--ok);border-color:var(--ok)}
.v button.fix.on{background:var(--warn);border-color:var(--warn)}
.v button.rej.on{background:var(--bad);border-color:var(--bad)}
.strip{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:4px;margin-top:8px}
.strip a{position:relative;display:block;border:3px solid transparent;border-radius:4px;overflow:hidden}
.strip a.none{border-color:var(--bad);opacity:.55}
.strip a.agree{border-color:var(--ok)}.strip a.disagree{border-color:var(--warn)}
.strip img{display:block;width:100%;height:auto}
.strip span{position:absolute;left:3px;top:2px;font:11px monospace;color:#fff;text-shadow:0 0 3px #000}
.hidefr .strip a.none{display:none}
#lb{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9}
#lb img{max-width:96vw;max-height:90vh}#lb.on{display:flex}
#lb p{position:absolute;bottom:8px;color:#fff;font:13px monospace}
</style></head><body>
<header>
 <h1>Pre-label review · __RUN__</h1>
 <span id="filters"></span>
 <button id="hidemiss">hide no-box frames</button>
 <select id="vf"><option value="">all verdicts</option><option value="none">unreviewed</option>
  <option value="ok">ok</option><option value="fix">fix</option><option value="rej">reject</option></select>
 <button id="exp">export verdicts CSV</button>
 <span class="stat" id="stat"></span>
</header>
<main id="list"></main>
<div id="lb"><img alt=""><p></p></div>
<script>
const T = __DATA__;
const KEY = "prelabel-verdicts-__RUN__";
let V = {}; try { V = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(V)); } catch (e) {} };
let cls = "", vf = "";
const labels = [...new Set(T.map(t => t.label))].sort();
const fbox = document.getElementById("filters");
["", ...labels].forEach(l => {
  const b = document.createElement("button"); b.textContent = l || "all";
  b.onclick = () => { cls = l; render(); }; b.dataset.l = l; fbox.appendChild(b);
});
document.getElementById("vf").onchange = e => { vf = e.target.value; render(); };
document.getElementById("hidemiss").onclick = e => {
  document.body.classList.toggle("hidefr"); e.target.classList.toggle("on"); };
document.getElementById("exp").onclick = () => {
  const rows = [["side","trace","object","label_class","hits","n","verdict"]];
  T.forEach(t => rows.push([t.side,t.trace,t.object,t.label,t.hits,t.n,V[t.key]||""]));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.map(r=>r.join(",")).join("\n")],{type:"text/csv"}));
  a.download = "prelabel_verdicts.csv"; a.click();
};
const lb = document.getElementById("lb");
lb.onclick = () => lb.classList.remove("on");
function render() {
  fbox.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.l === cls));
  const list = document.getElementById("list"); list.innerHTML = "";
  const shown = T.filter(t => (!cls || t.label === cls) &&
    (!vf || (vf === "none" ? !V[t.key] : V[t.key] === vf)));
  shown.forEach(t => {
    const d = document.createElement("section"); d.className = "tr";
    const pct = Math.round(100 * t.hits / t.n);
    d.innerHTML = `<div class="hd"><b>${t.side} ${t.trace}</b>
      <span>${t.object}</span><span class="pill ${t.bin}">${t.label} · ${t.bin}</span>
      <span class="m">${t.speed}</span>
      <span class="m">boxes ${t.hits}/${t.n} <span class="bar"><i style="width:${pct}%"></i></span>
      agree ${t.agree} · conf ${t.conf ?? "–"}</span>
      ${t.note ? `<span class="pill" style="color:var(--warn)">${t.note}</span>` : ""}
      <span class="v">${["ok","fix","rej"].map(v =>
        `<button class="${v}${V[t.key]===v?" on":""}" data-v="${v}">${v}</button>`).join("")}</span>
      </div><div class="strip"></div>`;
    d.querySelectorAll(".v button").forEach(b => b.onclick = () => {
      V[t.key] = V[t.key] === b.dataset.v ? undefined : b.dataset.v; save(); render(); });
    const s = d.querySelector(".strip");
    t.frames.forEach(f => {
      const a = document.createElement("a"); a.className = f.s; a.href = "#";
      a.innerHTML = `<img loading="lazy" src="${f.src}" alt=""><span>f${f.f}${f.c ? " " + f.c + " " + f.p : ""}</span>`;
      a.onclick = e => { e.preventDefault(); lb.querySelector("img").src = f.src;
        lb.querySelector("p").textContent = `${t.side} ${t.trace} · frame ${f.f} · ${f.c || "no box"} ${f.p ?? ""}`;
        lb.classList.add("on"); };
      s.appendChild(a);
    });
    list.appendChild(d);
  });
  const done = T.filter(t => V[t.key]).length;
  document.getElementById("stat").textContent = `${shown.length} shown · ${done}/${T.length} reviewed`;
}
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
