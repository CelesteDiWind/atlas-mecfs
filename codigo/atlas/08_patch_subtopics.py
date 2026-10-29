"""Una sola vez (2026-10-24): segundo nivel de temas en el visor del atlas EM/SFC, portado del atlas de la complejidad.
Al aislar un tema con subtemas aparece debajo la lista de sus subtemas; sus puntos usan la paleta completa (subtema k = color k) y
el resto del mapa pasa a gris. Un tema sin subtemas (XMRV, dolor) se pinta como siempre. Solo el corpus OpenAlex tiene subtemas
(clusters[c].subs y p.st vienen de homologia/openalex/scripts/o24_subtopics.py y o12_export.py); la selección manual no cambia.
Comprueba cada anclaje: si la plantilla cambió, se detiene."""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "viewer", "template.html")
s = open(P, encoding="utf-8").read()
assert "subLegend" not in s, "ya aplicado"


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new, 1)


rep('#yearLegend { display: grid; gap: 4px; }',
    '#subWrap { margin: 8px 0 0 14px; padding-left: 8px; border-left: 2px solid var(--line, rgba(127,127,127,.3)); }\n'
    '#subLegend { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 1px; }\n'
    '#subLegend button { width: 100%; display: grid; grid-template-columns: 10px 1fr auto; align-items: center; gap: 8px; border: 0; background: transparent;'
    ' border-radius: 6px; padding: 4px 6px; text-align: left; cursor: pointer; font-size: .78rem; line-height: 1.25; color: inherit; }\n'
    '#subLegend button:hover, #subLegend button[aria-pressed="true"] { background: var(--surface-2); }\n'
    '#subLegend.filtering button[aria-pressed="false"] { opacity: .45; }\n'
    '#subLegend .dot { width: 9px; height: 9px; border-radius: 50%; }\n'
    '#subLegend .n { font-family: var(--f-data); font-size: .7rem; color: var(--muted); font-variant-numeric: tabular-nums; }\n'
    '#yearLegend { display: grid; gap: 4px; }')
rep('      <ul id="legend"></ul>',
    '      <ul id="legend"></ul>\n'
    '      <div id="subWrap" hidden>\n'
    '        <p class="label-sm" id="subTitle" data-i18n="subTitle">Subtemas · clic para aislar</p>\n'
    '        <ul id="subLegend"></ul>\n'
    '      </div>')
rep('legendTopics: "Temas · clic para aislar",', 'legendTopics: "Temas · clic para aislar", subTitle: "Subtemas · clic para aislar",')
rep('legendTopics: "Topics · click to isolate",', 'legendTopics: "Topics · click to isolate", subTitle: "Subtopics · click to isolate",')
rep('let mode = "topic", activeTopic = null,', 'let mode = "topic", activeTopic = null, activeSub = null,')
rep('const isActive = (i) => (activeTopic === null || P[i].c === activeTopic) &&',
    'const isActive = (i) => (activeTopic === null || P[i].c === activeTopic) && (activeSub === null || P[i].st === activeSub) &&')
rep('      if (mode === "topic") tmp.copy(col.series[p.c]);',
    '      // con un tema con subtemas aislado, sus subtemas toman la paleta completa y los demás temas pasan a gris\n'
    '      if (mode === "topic" && (activeTopic === null || !hasSubs(activeTopic))) tmp.copy(col.series[p.c]);\n'
    '      else if (mode === "topic") tmp.copy(p.c === activeTopic && p.st != null ? col.series[p.st % col.series.length] : col.nodata);')
rep('  function paint() {',
    '  const hasSubs = (c) => OA && C[c] && C[c].subs && C[c].subs.length > 1;\n'
    '  // nombre del subtema en el idioma elegido (data/subtopic_labels_oa.json); sin nombre, sus tres términos más distintivos\n'
    '  const subName = (sb) => (lang === "en" ? sb.name_en : sb.name) || sb.terms.slice(0, 3).join(" · ");\n'
    '  function paint() {')
rep('''  function setTopic(t) {
    activeTopic = t;''', '''  function renderSubs() {
    const ul = $("subLegend"), c = activeTopic === null ? null : C[activeTopic];
    $("subWrap").hidden = !c || !hasSubs(activeTopic) || mode !== "topic";
    ul.innerHTML = "";
    if ($("subWrap").hidden) return;
    ul.classList.toggle("filtering", activeSub !== null);
    c.subs.forEach((sb) => {
      const li = document.createElement("li");
      li.innerHTML = `<button type="button" aria-pressed="${activeSub === sb.id}"><span class="dot" style="background:${seriesVar(sb.id)}"></span><span></span><span class="n">${sb.n}</span></button>`;
      li.querySelector("span:nth-child(2)").textContent = subName(sb);
      li.querySelector("button").title = sb.terms.join(", ");
      li.querySelector("button").addEventListener("click", () => { activeSub = activeSub === sb.id ? null : sb.id; renderSubs(); paint(); updateCount(); });
      ul.appendChild(li);
    });
  }
  function setTopic(t) {
    activeTopic = t; activeSub = null;''')
rep('''    legend.classList.toggle("filtering", t !== null);
    paint(); updateCount();''', '''    legend.classList.toggle("filtering", t !== null);
    renderSubs();
    paint(); updateCount();''')
rep('''    legend.hidden = m === "year";''', '''    legend.hidden = m === "year";
    renderSubs();''')
rep('''    chip(`<i style="background:${seriesVar(p.c)}"></i><span></span>`).querySelector("span").textContent = cname(p.c);
    if (p.y) chip(String(p.y), "mono");''', '''    chip(`<i style="background:${seriesVar(p.c)}"></i><span></span>`).querySelector("span").textContent = cname(p.c);
    const sb = hasSubs(p.c) && p.st != null ? C[p.c].subs.find((x) => x.id === p.st) : null;
    if (sb) { const s = chip("<span></span>", "rel"); s.querySelector("span").textContent = subName(sb); s.title = sb.terms.join(", "); }
    if (p.y) chip(String(p.y), "mono");''')
rep('''    renderCycleList();
    if (selected !== null) renderPanel(selected);
  }''', '''    renderCycleList();
    renderSubs();
    if (selected !== null) renderPanel(selected);
  }''')
open(P, "w", encoding="utf-8").write(s)
print("subtemas añadidos al visor de EM/SFC")
