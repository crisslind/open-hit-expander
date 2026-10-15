"""PLIP interaction fingerprint analysis for saved Boltz protein–ligand complexes.

PLIP is optional and is executed via its command line. No inference is rerun.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
import html
import json
import re
import shutil
import subprocess

import pandas as pd


INTERACTION_TYPES = {
    "hydrophobic_interaction": ("Hydrophobic", "#737373"),
    "hydrogen_bond": ("Hydrogen bond", "#2563EB"),
    "pi_stack": ("π–π stacking", "#16A34A"),
    "salt_bridge": ("Salt bridge", "#DCA500"),
    "pi_cation_interaction": ("π–cation", "#EA580C"),
    "halogen_bond": ("Halogen bond", "#0D9488"),
    "water_bridge": ("Water bridge", "#7DD3FC"),
    "metal_complex": ("Metal coordination", "#9333EA"),
}
DETAIL_COLUMNS = ["job_id", "residue", "reschain", "resnr", "restype", "interaction", "distance_angstrom", "geometry_json", "structure_path"]
STATUS_COLUMNS = ["job_id", "status", "interaction_count", "message", "structure_path"]


def prepare_plip_pdb(path, output, protein_chain="A", ligand_chain="B"):
    """Convert model 1 to PDB and mark ligand chain as HETATM for PLIP.

    Fail on unsupported multi-residue ligands rather than silently producing an
    invalid interaction fingerprint. Preserve residue numbers and chain IDs.
    """
    import gemmi
    structure = gemmi.read_structure(str(path))
    if len(structure) < 1:
        raise ValueError("Structure has no coordinate model")
    protein = [c for c in structure[0] if c.name == protein_chain]
    ligand = [c for c in structure[0] if c.name == ligand_chain]
    if not protein or not ligand:
        raise ValueError(f"Expected protein chain {protein_chain!r} and ligand chain {ligand_chain!r}")
    residues = [r for c in ligand for r in c if len(r)]
    if len(residues) != 1:
        raise ValueError(f"Expected one ligand residue on chain {ligand_chain}; found {len(residues)}")
    if len(ligand_chain) != 1 or len(protein_chain) != 1:
        raise ValueError("PDB output requires single-character chain IDs")
    # Normalize only the ligand residue identity for robust PLIP recognition.
    residues[0].name = "LIG"
    # gemmi emits HETATM when het_flag='H'; preserve protein atom records.
    residues[0].het_flag = "H"
    raw = structure.make_pdb_string()
    ligand_records = [s for s in raw.splitlines() if s.startswith(("ATOM  ", "HETATM")) and s[21:22] == ligand_chain]
    if not ligand_records:
        raise ValueError("Ligand atoms were lost during PDB conversion")
    # Explicitly enforce HETATM in case converter ignores het_flag.
    lines = [("HETATM" + s[6:]) if s.startswith("ATOM  ") and s[21:22] == ligand_chain else s
             for s in raw.splitlines()]
    output = Path(output)
    output.write_text("\n".join(lines) + "\n")
    return output


def parse_plip_xml(xml, job_id, structure_path, ligand_chain="B", protein_chain="A"):
    """Parse interactions for the intended ligand; retain geometry fields."""
    root = ET.parse(xml).getroot()
    records = []
    found_site = False
    for site in root.findall(".//bindingsite"):
        # PLIP binding-site identifier is often LIG:B:1; ligand properties also
        # carry the chain. Ignore any unrelated ligand/cosolvent sites.
        ident = site.findtext("identifiers/hetid") or ""
        ligchain = site.findtext("identifiers/chain") or ""
        site_id = site.get("id", "")
        if ligchain and ligchain != ligand_chain:
            continue
        if ident and ident != "LIG":
            continue
        if not ligchain and site_id and not (f":{ligand_chain}:" in site_id or site_id.endswith(f":{ligand_chain}")):
            continue
        found_site = True
        for node in site.findall("./interactions/*/*"):
            if node.tag not in INTERACTION_TYPES:
                continue
            fields = {child.tag: (child.text or "").strip() for child in node}
            chain = fields.get("reschain", "")
            if chain != protein_chain:
                continue
            number = fields.get("resnr", "")
            restype = fields.get("restype", "")
            if not number or not restype:
                continue
            dist = next((fields[k] for k in ("dist", "dist_d-a", "distance", "dist_a-w") if fields.get(k)), None)
            try:
                distance = float(dist) if dist is not None else None
            except ValueError:
                distance = None
            records.append({
                "job_id": str(job_id), "residue": f"{chain}:{restype}{number}",
                "reschain": chain, "resnr": number, "restype": restype,
                "interaction": INTERACTION_TYPES[node.tag][0],
                "distance_angstrom": distance,
                "geometry_json": json.dumps(fields, ensure_ascii=False),
                "structure_path": str(structure_path),
            })
    if not found_site:
        raise ValueError(f"PLIP did not identify ligand LIG on chain {ligand_chain}")
    return records


def analyze_interactions(results, output_dir, structure_root=None, protein_chain="A", ligand_chain="B", plip_executable="plip", timeout=180):
    """Analyze selected (best) structures, export details, binary IFP and HTML.

    Returns (details, fingerprints, statuses, html_path). Failed jobs are kept
    in statuses and are never represented as zero-interaction fingerprints.
    """
    executable = shutil.which(plip_executable) or (str(Path(plip_executable)) if Path(plip_executable).is_file() else None)
    if not executable:
        raise RuntimeError("PLIP command not found. Install PLIP separately and ensure `plip` is on PATH.")
    from .visualization import available_complexes
    table = available_complexes(results, structure_root)
    if table.empty:
        raise ValueError("No existing structure files could be resolved")
    if table.job_id.astype(str).duplicated().any():
        raise ValueError("Duplicate job_id entries: use one selected structure per ligand")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, statuses = [], []
    for _, row in table.iterrows():
        job = str(row.job_id)
        path = Path(row.resolved_structure_path)
        try:
            with TemporaryDirectory(prefix="plip_job_") as tmp:
                tmp = Path(tmp)
                prepared = prepare_plip_pdb(path, tmp / "complex.pdb", protein_chain, ligand_chain)
                p = subprocess.run([executable, "-f", str(prepared), "-x", "-o", str(tmp / "report"), "-q"],
                                   text=True, capture_output=True, timeout=timeout, check=False)
                if p.returncode:
                    raise RuntimeError(f"PLIP exit {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
                reports = list((tmp / "report").rglob("*.xml"))
                if len(reports) != 1:
                    raise RuntimeError(f"Expected one PLIP XML report, found {len(reports)}")
                found = parse_plip_xml(reports[0], job, path, ligand_chain, protein_chain)
                records.extend(found)
                statuses.append({"job_id": job, "status": "ok", "interaction_count": len(found), "message": "", "structure_path": str(path)})
        except Exception as exc:
            statuses.append({"job_id": job, "status": "failed", "interaction_count": 0, "message": str(exc), "structure_path": str(path)})
    details = pd.DataFrame(records, columns=DETAIL_COLUMNS)
    status = pd.DataFrame(statuses, columns=STATUS_COLUMNS)
    good_ids = status.loc[status.status == "ok", "job_id"].tolist()
    feature_names = sorted({(r["residue"], r["interaction"]) for r in records}, key=lambda v: (_residue_sort(v[0]), v[1]))
    features = pd.DataFrame(0, index=pd.Index(good_ids, name="job_id"), columns=pd.MultiIndex.from_tuples(feature_names, names=["residue", "interaction"]), dtype=int)
    for r in records:
        features.loc[r["job_id"], (r["residue"], r["interaction"])] = 1
    flat = features.copy()
    flat.columns = [f"{a}|{b}" for a, b in flat.columns]
    flat.to_csv(out / "interaction_fingerprints.csv")
    details.to_csv(out / "interaction_details.csv", index=False)
    status.to_csv(out / "interaction_status.csv", index=False)
    report = out / "interaction_heatmap.html"
    report.write_text(build_heatmap_html(details, status))
    return details, flat, status, report


def _residue_sort(value):
    m = re.match(r"([^:]+):([A-Za-z]+)(-?\d+)(.*)", value)
    return (m.group(1), int(m.group(3)), m.group(4)) if m else (value, 0, "")


def build_heatmap_html(details, status):
    # Self-contained interactive fingerprint report with sorting and summaries.
    palette = {label: color for label, color in INTERACTION_TYPES.values()}
    ok = status.loc[status.status == "ok", "job_id"].astype(str).tolist()
    records = []
    for item in details.to_dict("records"):
        distance = item.get("distance_angstrom")
        records.append({
            "job": str(item["job_id"]), "residue": str(item["residue"]),
            "type": str(item["interaction"]),
            "distance": round(float(distance), 2) if pd.notna(distance) else None,
        })
    records.sort(key=lambda r: (_residue_sort(r["residue"]), r["job"], r["type"]))
    failures = status.loc[status.status != "ok", ["job_id", "message"]]
    warning = "" if failures.empty else '<p class="warning">Excluded failed analyses: ' + '; '.join(
        html.escape(str(r.job_id) + ': ' + str(r.message)) for r in failures.itertuples()
    ) + '</p>'
    payload = json.dumps({"ligands": ok, "records": records, "palette": palette}, ensure_ascii=False).replace("<", "\\u003c")
    return r'''<!doctype html><html><head><meta charset="utf-8"><title>Interaction Explorer</title>
<style>
body{font:13px system-ui,sans-serif;margin:18px;color:#263440;background:#fff}
h2{margin:0 0 4px}h3{margin:18px 0 8px}small,.hint{color:#687684}
.controls,.legend{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:12px 0}
button,select{font:inherit;border:1px solid #c9d2db;border-radius:5px;background:white;padding:6px}
label{white-space:nowrap;cursor:pointer}.swatch{display:inline-block;width:12px;height:12px;vertical-align:middle;border-radius:2px}
.viewport{overflow:auto;max-height:550px;border:1px solid #d9e0e7;border-radius:6px}
table{border-collapse:separate;border-spacing:2px}th{white-space:nowrap;background:#f0f3f6;position:sticky;top:0;padding:6px 5px;font-size:11px}td{min-width:39px;height:26px;background:#f0f2f4;padding:0}
.first{position:sticky;left:0;z-index:2;text-align:left;min-width:125px;padding:4px 8px;background:#f0f3f6}thead .first{z-index:3}
.bar{display:flex;width:100%;height:100%}.segment{flex:1;min-width:0}
.panel{border:1px solid #e0e5ea;border-radius:7px;padding:12px;margin-top:16px;overflow:auto}
.plotrow{display:grid;grid-template-columns:minmax(115px,180px) minmax(180px,1fr) 40px;gap:8px;align-items:center;margin:5px 0}
.stack{display:flex;height:19px;background:#f0f2f4;border-radius:3px;overflow:hidden}
.matrix td{width:24px;min-width:24px;height:24px;text-align:center;font-size:9px;color:#273442}
.matrix th{position:sticky;top:0}.warning{color:#9b2727}
</style></head><body>
<h2>Protein–ligand Interaction Explorer</h2>
<small>Each colored cell segment represents an interaction type. Hover for distances. Failed analyses are excluded.</small>
<div class="legend" id="legend"></div>
<div class="controls">
<label>Ligand sort <select id="ligsort"><option value="id">Job ID</option><option value="count">Interaction count ↓</option><option value="residues">Interacting residues ↓</option></select></label>
<label>Residue sort <select id="ressort"><option value="sequence">Sequence order</option><option value="frequency">Frequency ↓</option></select></label>
<label><input type="checkbox" id="hideempty" checked> Hide columns with no enabled interactions</label>
</div>
<h3>Interaction fingerprint</h3><div class="viewport" id="fingerprint"></div>
<div class="panel"><h3>Interactions per ligand</h3><div class="hint">Counts of detected interactions, not binding-strength scores</div><div id="counts"></div></div>
<div class="panel"><h3>Residue conservation</h3><div class="hint">Percentage of successfully analyzed ligands interacting with each residue</div><div id="conservation"></div></div>
<div class="panel"><h3>Ligand interaction similarity</h3><div class="hint">Binary residue × interaction-type Tanimoto similarity (1 = identical fingerprints). Empty fingerprints display n/a.</div><div class="viewport" id="similarity"></div></div>
''' + warning + r'''<script>
const INPUT = ''' + payload + r''';
const palette=INPUT.palette, allTypes=Object.keys(palette), jobs=INPUT.ligands;
const records=INPUT.records;
const els={legend:document.getElementById('legend'),fingerprint:document.getElementById('fingerprint'),counts:document.getElementById('counts'),conservation:document.getElementById('conservation'),similarity:document.getElementById('similarity')};
const byKey=new Map();for(const r of records){const k=JSON.stringify([r.job,r.residue]);if(!byKey.has(k))byKey.set(k,[]);byKey.get(k).push(r)}
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const natural=(a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true});
const residueSort=(a,b)=>{const pa=a.match(/^(.*?:[A-Za-z]+)(-?\d+)(.*)$/),pb=b.match(/^(.*?:[A-Za-z]+)(-?\d+)(.*)$/);return pa&&pb?(Number(pa[2])-Number(pb[2])||natural(a,b)):natural(a,b)};
for(const t of allTypes){const l=document.createElement('label');l.innerHTML='<input type="checkbox" class="filter" checked> <span class="swatch" style="background:'+palette[t]+'"></span> '+escapeHTML(t);l.querySelector('input').dataset.type=t;els.legend.appendChild(l)}
function draw(){
 const allowed=new Set([...document.querySelectorAll('.filter:checked')].map(e=>e.dataset.type));
 const included=records.filter(r=>allowed.has(r.type));
 const residueNames=[...new Set(included.map(r=>r.residue))];
 const nCounts=new Map(),counts=new Map(),types=new Map(),features=new Map();
 for(const j of jobs){counts.set(j,0);nCounts.set(j,new Set());types.set(j,new Map());features.set(j,new Set())}
 const rJobs=new Map();
 for(const r of included){counts.set(r.job,(counts.get(r.job)||0)+1);nCounts.get(r.job)?.add(r.residue);const t=types.get(r.job);if(t)t.set(r.type,(t.get(r.type)||0)+1);features.get(r.job)?.add(r.residue+'|'+r.type);if(!rJobs.has(r.residue))rJobs.set(r.residue,new Set());rJobs.get(r.residue).add(r.job)}
 const ligOrder=[...jobs];const ligSort=document.getElementById('ligsort').value;
 ligOrder.sort((a,b)=>ligSort==='count'?(counts.get(b)-counts.get(a)||natural(a,b)):ligSort==='residues'?(nCounts.get(b).size-nCounts.get(a).size||natural(a,b)):natural(a,b));
 let resOrder=[...new Set(records.map(r=>r.residue))];if(document.getElementById('hideempty').checked)resOrder=resOrder.filter(r=>rJobs.has(r));
 const resSort=document.getElementById('ressort').value;
 resOrder.sort((a,b)=>resSort==='frequency'?((rJobs.get(b)?.size||0)-(rJobs.get(a)?.size||0)||residueSort(a,b)):residueSort(a,b));
 let tbl='<table><thead><tr><th class="first">Ligand</th>'+resOrder.map(r=>'<th>'+escapeHTML(r)+'</th>').join('')+'</tr></thead><tbody>';
 for(const j of ligOrder){tbl+='<tr><th class="first">'+escapeHTML(j)+'</th>';
 for(const r of resOrder){const hits=(byKey.get(JSON.stringify([j,r]))||[]).filter(x=>allowed.has(x.type));const distinct=[...new Set(hits.map(x=>x.type))];const tip=hits.map(x=>x.type+': '+(x.distance??'n/a')+' Å').join('&#10;');tbl+='<td title="'+escapeHTML(tip)+'"><div class="bar">'+distinct.map(t=>'<span class="segment" style="background:'+palette[t]+'"></span>').join('')+'</div></td>'}
 tbl+='</tr>'}els.fingerprint.innerHTML=tbl+'</tbody></table>';
 const max=Math.max(1,...ligOrder.map(j=>counts.get(j)));els.counts.innerHTML=ligOrder.map(j=>'<div class="plotrow"><span>'+escapeHTML(j)+'</span><div class="stack" title="'+counts.get(j)+' interactions" style="width:'+Math.max(1,100*counts.get(j)/max)+'%">'+[...types.get(j)].map(([t,n])=>'<span style="width:'+(100*n/Math.max(1,counts.get(j)))+'%;background:'+palette[t]+'" title="'+escapeHTML(t)+': '+n+'"></span>').join('')+'</div><b>'+counts.get(j)+'</b></div>').join('');
 els.conservation.innerHTML=[...rJobs].sort((a,b)=>b[1].size-a[1].size||residueSort(a[0],b[0])).map(([r,js])=>'<div class="plotrow"><span>'+escapeHTML(r)+'</span><div class="stack"><span style="width:'+(100*js.size/Math.max(1,jobs.length))+'%;background:#4f78a2"></span></div><b>'+Math.round(100*js.size/Math.max(1,jobs.length))+'%</b></div>').join('')||'<p>No interactions of selected types.</p>';
 let sim='<table class="matrix"><thead><tr><th class="first">Ligand</th>'+ligOrder.map(j=>'<th>'+escapeHTML(j)+'</th>').join('')+'</tr></thead><tbody>';
 for(const a of ligOrder){sim+='<tr><th class="first">'+escapeHTML(a)+'</th>';for(const b of ligOrder){const A=features.get(a),B=features.get(b),inter=[...A].filter(x=>B.has(x)).length,union=new Set([...A,...B]).size,v=union?inter/union:null;sim+='<td title="'+escapeHTML(a)+' / '+escapeHTML(b)+'" style="background:'+(v===null?'#edf0f3':`rgba(46,125,105,${.12+.78*v})`)+'">'+(v===null?'–':v.toFixed(2))+'</td>'}sim+='</tr>'}els.similarity.innerHTML=sim+'</tbody></table>';
}
document.querySelectorAll('.filter,#ligsort,#ressort,#hideempty').forEach(e=>e.addEventListener('change',draw));draw();
</script></body></html>'''


def display_interaction_heatmap(details, status, height=560):
    """Display a static self-contained interaction heatmap in a Jupyter output."""
    from IPython.display import HTML, display
    doc = build_heatmap_html(details, status)
    frame = '<iframe sandbox="allow-scripts" title="Interaction fingerprint heatmap" style="width:100%;border:0;height:' + str(int(height)) + 'px" srcdoc="' + html.escape(doc, quote=True) + '"></iframe>'
    display(HTML(frame))
