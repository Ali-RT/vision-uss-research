"""Click capture for Colab notebooks (point or 2-corner box on an image).

Moved here from notebook 03's click round so other notebooks (19) can reuse
it. In Colab the click comes back through google.colab.output.eval_js; locally
the coordinates are typed in.
"""

from __future__ import annotations

import base64
import random
from pathlib import Path


def ask_click_colab(image_path: Path, title: str, mode: str = "point"):
    from google.colab.output import eval_js
    from IPython.display import display as _display, HTML

    b64 = base64.b64encode(image_path.read_bytes()).decode()
    uid = f"clk_{random.randrange(10**9)}"
    hint = ("draw a BOX: click one corner of the object"
            if mode == "box" else "click the object...")
    _display(HTML(f'''
      <div style="font-family:sans-serif;margin:4px 0"><b>{title}</b></div>
      <div style="position:relative;display:inline-block;line-height:0">
        <img id="{uid}" src="data:image/jpeg;base64,{b64}"
             style="max-width:100%;cursor:crosshair;border:2px solid #888"/>
        <div id="{uid}_mk" style="position:absolute;display:none;width:26px;height:26px;
             margin:-13px 0 0 -13px;pointer-events:none;border:3px solid lime;
             border-radius:50%;box-shadow:0 0 0 2px rgba(0,0,0,.6)">
          <div style="position:absolute;left:50%;top:50%;width:4px;height:4px;
               margin:-2px 0 0 -2px;background:red;border-radius:50%"></div>
        </div>
        <div id="{uid}_rect" style="position:absolute;display:none;pointer-events:none;
             border:3px dashed lime;box-shadow:0 0 0 1px rgba(0,0,0,.6)"></div>
      </div>
      <div style="display:flex;align-items:center;gap:14px;margin:6px 0">
        <div style="text-align:center;font-family:sans-serif;font-size:12px;color:#555">
          <canvas id="{uid}_zoom" width="180" height="180"
                  style="border:2px solid #888;background:#222;display:block"></canvas>
          zoomed check
        </div>
        <div>
          <div id="{uid}_st" style="font-family:sans-serif;margin:6px 0;color:#555">
            {hint}</div>
          <button id="{uid}_ok" disabled
                  style="padding:6px 20px;font-weight:bold">Confirm</button>
          <button id="{uid}_skip" style="padding:6px 16px;margin-left:8px">Skip</button>
        </div>
      </div>
    '''))
    coords = eval_js(f'''
      new Promise(resolve => {{
        const MODE = "{mode}";
        const img = document.getElementById("{uid}");
        const mk = document.getElementById("{uid}_mk");
        const rect = document.getElementById("{uid}_rect");
        const st = document.getElementById("{uid}_st");
        const ok = document.getElementById("{uid}_ok");
        const zoom = document.getElementById("{uid}_zoom");
        let pt = null, c1 = null, box = null;
        const drawZoom = (nx, ny) => {{
          const half = 30;
          const ctx = zoom.getContext("2d");
          ctx.imageSmoothingEnabled = false;
          ctx.clearRect(0, 0, zoom.width, zoom.height);
          ctx.drawImage(img, nx - half, ny - half, 2 * half, 2 * half,
                        0, 0, zoom.width, zoom.height);
          ctx.strokeStyle = "lime"; ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(zoom.width / 2, 0); ctx.lineTo(zoom.width / 2, zoom.height);
          ctx.moveTo(0, zoom.height / 2); ctx.lineTo(zoom.width, zoom.height / 2);
          ctx.stroke();
          ctx.strokeStyle = "red"; ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(zoom.width / 2, zoom.height / 2, 9, 0, 6.283);
          ctx.stroke();
        }};
        img.onclick = e => {{
          const r = img.getBoundingClientRect();
          const dx = e.clientX - r.left, dy = e.clientY - r.top;
          const nx = dx * img.naturalWidth / r.width;
          const ny = dy * img.naturalHeight / r.height;
          drawZoom(nx, ny);
          if (MODE === "point") {{
            pt = [nx, ny];
            mk.style.left = dx + "px"; mk.style.top = dy + "px";
            mk.style.display = "block";
            st.textContent = "selected (" + Math.round(nx) + ", " + Math.round(ny)
                           + ") - check the zoom, click again to adjust, then Confirm";
            st.style.color = "#0a0";
            ok.disabled = false;
          }} else if (!c1 || box) {{
            c1 = {{dx: dx, dy: dy, nx: nx, ny: ny}};
            box = null;
            rect.style.display = "none";
            mk.style.left = dx + "px"; mk.style.top = dy + "px";
            mk.style.display = "block";
            st.textContent = "corner 1 set (" + Math.round(nx) + ", " + Math.round(ny)
                           + ") - now click the OPPOSITE corner";
            st.style.color = "#c60";
            ok.disabled = true;
          }} else {{
            box = [Math.min(c1.nx, nx), Math.min(c1.ny, ny),
                   Math.max(c1.nx, nx), Math.max(c1.ny, ny)];
            rect.style.left = Math.min(c1.dx, dx) + "px";
            rect.style.top = Math.min(c1.dy, dy) + "px";
            rect.style.width = Math.abs(dx - c1.dx) + "px";
            rect.style.height = Math.abs(dy - c1.dy) + "px";
            rect.style.display = "block";
            mk.style.display = "none";
            st.textContent = "box set (" + box.map(Math.round).join(", ")
                           + ") - click again to restart, or Confirm";
            st.style.color = "#0a0";
            ok.disabled = false;
          }}
        }};
        ok.onclick = () => {{
          img.style.border = "4px solid lime";
          st.textContent = "confirmed";
          resolve(MODE === "point" ? pt : box);
        }};
        document.getElementById("{uid}_skip").onclick = () => resolve(null);
      }})
    ''')
    return tuple(coords) if coords else None

def ask_click_local(image_path: Path, title: str, mode: str = "point"):
    print(f"{title}\nimage: {image_path}")
    if mode == "box":
        raw = input("enter 'x0,y0,x1,y1' box coords, or blank to skip: ").strip()
        return tuple(map(float, raw.split(","))) if raw else None
    raw = input("enter 'x,y' pixel coords, or blank to skip: ").strip()
    if not raw:
        return None
    x, y = raw.split(",")
    return (float(x), float(y))
