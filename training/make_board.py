"""
Generate a standalone, sortable/filterable HTML view from the current
server/models/projections.csv metadata. No server required.

Output: data/processed/board_2026.html  (open in any browser)
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PROJ = ROOT / "server" / "models" / "projections.csv"
OUT  = ROOT / "data" / "processed" / "board_2026.html"

POS_COLOR = {"QB": "#e0476b", "WR": "#2f9e6f", "RB": "#3a7bd5", "TE": "#c9852b"}


def main():
    df = pd.read_csv(PROJ)
    df = df.sort_values("sim_value", ascending=False).reset_index(drop=True)
    df["overall_rank"] = df.index + 1
    # Positional rank by raw projected points (pure model output for that position)
    df["pos_rank"] = df.groupby("position")["proj_points"].rank(ascending=False, method="first").astype(int)

    cols = ["overall_rank", "pos_rank", "player_display_name", "position", "recent_team",
            "age", "proj_points", "proj_boom_rate", "vor", "sim_value"]
    records = []
    for _, r in df[cols].iterrows():
        records.append({
            "overall": int(r["overall_rank"]),
            "posRank": int(r["pos_rank"]),
            "name": r["player_display_name"],
            "pos": r["position"],
            "team": r["recent_team"] if pd.notna(r["recent_team"]) else "",
            "age": round(float(r["age"]), 0) if pd.notna(r["age"]) else "",
            "pts": round(float(r["proj_points"]), 1),
            "boom": round(float(r["proj_boom_rate"]) * 100, 1),
            "vor": round(float(r["vor"]), 1),
            "sim": round(float(r["sim_value"]), 1),
        })

    html = _TEMPLATE.replace("__DATA__", json.dumps(records)) \
                    .replace("__COLORS__", json.dumps(POS_COLOR)) \
                    .replace("__N__", str(len(records)))
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {len(records)} players -> {OUT}")


_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>2026 Projection Metadata (clean model)</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#11151c;color:#e6e9ef;margin:0;padding:24px;}
 h1{margin:0 0 4px;font-size:20px;} .sub{color:#8b94a3;font-size:13px;margin-bottom:14px;}
 .warn{background:#3a2a12;border:1px solid #7a5a1e;color:#f0c987;padding:8px 12px;border-radius:6px;font-size:12.5px;margin-bottom:16px;}
 .bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px;}
 button{background:#1c2230;color:#cdd3de;border:1px solid #2c3445;padding:6px 13px;border-radius:6px;cursor:pointer;font-size:13px;}
 button.active{background:#2f6fed;color:#fff;border-color:#2f6fed;}
 input{background:#1c2230;color:#e6e9ef;border:1px solid #2c3445;padding:6px 10px;border-radius:6px;font-size:13px;width:200px;}
 table{border-collapse:collapse;width:100%;font-size:13px;}
 th,td{padding:6px 10px;text-align:right;border-bottom:1px solid #1e2531;white-space:nowrap;}
 th{position:sticky;top:0;background:#161b24;cursor:pointer;user-select:none;color:#aab2c0;}
 th:hover{color:#fff;} td.l,th.l{text-align:left;}
 .pos{display:inline-block;width:30px;text-align:center;border-radius:4px;color:#fff;font-weight:600;font-size:11px;padding:2px 0;}
 tr:hover td{background:#171d27;}
 .pr{color:#8b94a3;}
</style></head><body>
 <h1>2026 Projection Metadata <span class=pr style="font-size:13px">— __N__ players</span></h1>
 <div class=sub>Static projection metadata for inspection only. The live draft assistant ranks with the DK EV policy.</div>
 <div class=warn>Use the server/extension for live draft rankings; this page is a sortable metadata view.</div>
 <div class=bar>
   <button data-f="ALL" class=active>ALL</button>
   <button data-f="QB">QB</button><button data-f="WR">WR</button>
   <button data-f="RB">RB</button><button data-f="TE">TE</button>
   <input id=q placeholder="search player / team...">
 </div>
 <table><thead><tr>
   <th data-k=overall>Ovr</th><th data-k=posRank>Pos#</th>
   <th class=l data-k=name>Player</th><th class=l data-k=pos>Pos</th><th class=l data-k=team>Tm</th>
   <th data-k=age>Age</th><th data-k=pts>Proj Pts</th><th data-k=boom>Boom%</th>
   <th data-k=vor>VOR</th><th data-k=sim>SimVal</th>
 </tr></thead><tbody id=body></tbody></table>
<script>
 const DATA=__DATA__, COLORS=__COLORS__;
 let filt="ALL", sortK="overall", asc=true, q="";
 const body=document.getElementById("body");
 function render(){
   let rows=DATA.filter(r=>(filt==="ALL"||r.pos===filt));
   if(q){const s=q.toLowerCase();rows=rows.filter(r=>r.name.toLowerCase().includes(s)||(r.team||"").toLowerCase().includes(s));}
   rows.sort((a,b)=>{let x=a[sortK],y=b[sortK];if(typeof x==="string"){x=x.toLowerCase();y=(y||"").toLowerCase();}return (x>y?1:x<y?-1:0)*(asc?1:-1);});
   body.innerHTML=rows.map(r=>`<tr>
     <td>${r.overall}</td><td class=pr>${r.pos}${r.posRank}</td>
     <td class=l>${r.name}</td>
     <td class=l><span class=pos style="background:${COLORS[r.pos]}">${r.pos}</span></td>
     <td class=l>${r.team}</td><td>${r.age}</td>
     <td><b>${r.pts}</b></td><td>${r.boom}</td><td>${r.vor}</td><td>${r.sim}</td></tr>`).join("");
 }
 document.querySelectorAll("button[data-f]").forEach(b=>b.onclick=()=>{
   document.querySelectorAll("button[data-f]").forEach(x=>x.classList.remove("active"));
   b.classList.add("active");filt=b.dataset.f;
   if(filt!=="ALL"){sortK="posRank";asc=true;} else {sortK="overall";asc=true;}
   render();
 });
 document.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{
   const k=th.dataset.k; if(sortK===k)asc=!asc; else {sortK=k;asc=(k==="overall"||k==="posRank"||k==="name"||k==="pos"||k==="team");}
   render();
 });
 document.getElementById("q").oninput=e=>{q=e.target.value;render();};
 render();
</script></body></html>"""


if __name__ == "__main__":
    main()
