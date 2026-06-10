'use strict';

// ── Formatting ────────────────────────────────────────────────────────────────

function fmtISK(v) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 2 }) + ' ISK';
}

function fmtM3(v) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 2 }) + ' m³';
}

function fmtNum(v) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US');
}

function fmtPct(v) {
  return (v * 100).toFixed(1) + '%';
}

function secColor(sec) {
  if (sec >= 0.5) return 'green';
  if (sec >= 0.1) return 'yellow';
  return 'red';
}

// ── Autocomplete ──────────────────────────────────────────────────────────────

function makeAutocomplete({ inputId, listId, hiddenId, endpoint, labelKey, valueKey, extraLabel }) {
  const input = document.getElementById(inputId);
  const list = document.getElementById(listId);
  const hidden = document.getElementById(hiddenId);
  let timer, activeIdx = -1;

  input.addEventListener('input', () => {
    hidden.value = '';
    clearTimeout(timer);
    const q = input.value.trim();
    if (q.length < 2) { closeList(); return; }
    timer = setTimeout(() => fetchSuggestions(q), 250);
  });

  input.addEventListener('keydown', (e) => {
    const items = list.querySelectorAll('li');
    if (e.key === 'ArrowDown') { activeIdx = Math.min(activeIdx + 1, items.length - 1); highlight(items); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { activeIdx = Math.max(activeIdx - 1, 0); highlight(items); e.preventDefault(); }
    else if (e.key === 'Enter' && activeIdx >= 0) { items[activeIdx].click(); e.preventDefault(); }
    else if (e.key === 'Escape') closeList();
  });

  document.addEventListener('click', (e) => {
    if (!input.contains(e.target) && !list.contains(e.target)) closeList();
  });

  function highlight(items) {
    items.forEach((li, i) => li.classList.toggle('active', i === activeIdx));
  }

  async function fetchSuggestions(q) {
    try {
      const res = await fetch(`/api/v1/${endpoint}?q=${encodeURIComponent(q)}&limit=10`);
      const data = await res.json();
      renderList(data);
    } catch { closeList(); }
  }

  function renderList(items) {
    list.innerHTML = '';
    activeIdx = -1;
    if (!items.length) { closeList(); return; }
    items.forEach(item => {
      const li = document.createElement('li');
      const extra = extraLabel ? extraLabel(item) : '';
      li.innerHTML = item[labelKey] + (extra ? `<span class="sec ${secColor(item.security)}">${extra}</span>` : '');
      li.addEventListener('click', () => {
        input.value = item[labelKey];
        hidden.value = item[valueKey];
        closeList();
      });
      list.appendChild(li);
    });
    list.classList.add('open');
  }

  function closeList() {
    list.classList.remove('open');
    list.innerHTML = '';
    activeIdx = -1;
  }
}

makeAutocomplete({
  inputId: 'item-search', listId: 'item-list', hiddenId: 'item-id',
  endpoint: 'search', labelKey: 'name', valueKey: 'type_id',
});

makeAutocomplete({
  inputId: 'system-search', listId: 'system-list', hiddenId: 'system-id',
  endpoint: 'search-systems', labelKey: 'name', valueKey: 'system_id',
  extraLabel: item => item.security != null ? item.security.toFixed(1) : '',
});

// ── BPC tree walker ───────────────────────────────────────────────────────────

function collectBPC(nodes, out = []) {
  for (const node of nodes) {
    if (node.bpc_copies_needed > 0) {
      out.push(node);
    }
    if (node.children && node.children.length) {
      collectBPC(node.children, out);
    }
  }
  return out;
}

// ── BOM tree rendering ────────────────────────────────────────────────────────

function renderBOMNode(node, isRoot) {
  const hasChildren = node.children && node.children.length > 0;
  const wrapper = document.createElement('div');
  wrapper.className = 'bom-node';

  const row = document.createElement('div');
  row.className = 'bom-row';

  const toggle = document.createElement('span');
  toggle.className = 'bom-toggle';
  toggle.textContent = hasChildren ? '▼' : '';

  const name = document.createElement('span');
  name.className = 'bom-name';
  name.textContent = node.name;

  const qty = document.createElement('span');
  qty.className = 'bom-qty muted';
  qty.textContent = '×' + fmtNum(node.quantity);

  const cost = document.createElement('span');
  cost.className = 'bom-cost isk';
  cost.textContent = fmtISK(node.total_cost);

  const bpc = document.createElement('span');
  bpc.className = 'bom-bpc';
  if (node.bpc_copies_needed > 0) {
    bpc.textContent = `${node.bpc_copies_needed}×BPC (${node.max_runs_per_bpc}/copy)`;
  }

  row.appendChild(toggle);
  row.appendChild(name);
  row.appendChild(qty);
  row.appendChild(cost);
  row.appendChild(bpc);
  wrapper.appendChild(row);

  if (hasChildren) {
    const children = document.createElement('div');
    children.className = 'bom-children';
    for (const child of node.children) {
      children.appendChild(renderBOMNode(child, false));
    }
    wrapper.appendChild(children);

    row.addEventListener('click', () => {
      const collapsed = children.classList.toggle('collapsed');
      toggle.textContent = collapsed ? '▶' : '▼';
    });
  }

  return wrapper;
}

// ── Main calculate ────────────────────────────────────────────────────────────

async function calculate() {
  const typeId = document.getElementById('item-id').value;
  const systemId = document.getElementById('system-id').value;

  if (!typeId) { showError('Select an item to build.'); return; }
  if (!systemId) { showError('Select a build system.'); return; }

  setLoading(true);
  clearError();
  document.getElementById('results').style.display = 'none';
  document.getElementById('compare-section').style.display = 'none';

  const body = {
    type_id: parseInt(typeId),
    system_id: parseInt(systemId),
    runs: parseInt(document.getElementById('runs').value) || 1,
    me_level: parseInt(document.getElementById('me-level').value),
    fw_level: parseInt(document.getElementById('fw-level').value),
    material_source: document.getElementById('material-source').value,
    structure_bonus: parseFloat(document.getElementById('structure-bonus').value),
    logistics_cost_isk_per_m3: parseFloat(document.getElementById('logistics').value) || 0,
  };

  // Compare-specific fields
  const compareBody = { ...body };
  compareBody.leftover_logistics_isk_per_m3 = parseFloat(document.getElementById('leftover-logistics').value) || 0;
  const maxLeftoverRaw = document.getElementById('max-leftover-isk').value.trim();
  if (maxLeftoverRaw !== '') {
    compareBody.max_leftover_isk = parseFloat(maxLeftoverRaw);
  }

  try {
    const [buildRes, compareRes] = await Promise.all([
      fetch('/api/v1/build-cost', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
      fetch('/api/v1/compare-material-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(compareBody),
      }),
    ]);

    if (!buildRes.ok) {
      const err = await buildRes.json();
      throw new Error(err.detail || buildRes.statusText);
    }

    const build = await buildRes.json();
    const compare = compareRes.ok ? await compareRes.json() : null;

    renderBuild(build, compare);
    if (compare) renderCompare(compare);

    document.getElementById('results').style.display = 'block';
  } catch (e) {
    showError(e.message);
  } finally {
    setLoading(false);
  }
}

// ── Render helpers ────────────────────────────────────────────────────────────

function renderBuild(data, compareData) {
  const bd = data.cost_breakdown;
  const co = compareData?.compressed_ore;
  const constraintMet = co?.leftover_constraint_met !== false;
  const leftoverNet = co?.leftover_net_isk || 0;
  // Fall back to direct buy when the leftover limit cannot be satisfied
  const useOre = constraintMet && leftoverNet > 0;

  const noteEl = document.getElementById('total-note');
  if (!constraintMet) {
    noteEl.textContent = 'Leftover limit impossible with compressed ore — direct buy price used.';
    noteEl.style.display = 'block';
  } else {
    noteEl.style.display = 'none';
  }

  const netTotal = useOre
    ? (co.total_isk + co.refining_fee
       + bd.manufacturing_fees + bd.reaction_fees + (bd.refining_fees || 0)
       + bd.logistics_costs - leftoverNet)
    : data.total_cost;

  document.getElementById('total-cost').textContent = fmtISK(netTotal);
  document.getElementById('total-label').textContent = useOre ? 'Net total cost ' : 'Total cost ';

  const rows = useOre
    ? [
        ['Material costs (ore purchase)', co.total_isk],
        ['Ore refining fee',              co.refining_fee],
        ['Manufacturing fees',            bd.manufacturing_fees],
        ['Reaction fees',                 bd.reaction_fees],
        ['Refining fees',                 bd.refining_fees],
        ['Logistics',                     bd.logistics_costs],
      ]
    : [
        ['Material costs',      bd.material_costs],
        ['Manufacturing fees',  bd.manufacturing_fees],
        ['Reaction fees',       bd.reaction_fees],
        ['Refining fees',       bd.refining_fees],
        ['Logistics',           bd.logistics_costs],
      ];

  const tbody = document.getElementById('breakdown-body');
  tbody.innerHTML = '';
  for (const [label, val] of rows) {
    if (!val) continue;
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${label}</td><td class="num isk">${fmtISK(val)}</td><td class="num muted">${fmtPct(val / netTotal)}</td>`;
    tbody.appendChild(tr);
  }

  if (useOre) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="green">Leftover material credit (net)</td>
      <td class="num green">−${fmtISK(leftoverNet)}</td>
      <td class="num muted">${fmtPct(leftoverNet / netTotal)}</td>
    `;
    tbody.appendChild(tr);

    const divTr = document.createElement('tr');
    divTr.style.borderTop = '1px solid var(--border)';
    divTr.innerHTML = `
      <td style="font-weight:600">Net total</td>
      <td class="num isk" style="font-weight:600">${fmtISK(netTotal)}</td>
      <td class="num muted">100%</td>
    `;
    tbody.appendChild(divTr);
  }

  // BPC table
  const bpcNodes = collectBPC(data.bom_tree);
  const bpcBody = document.getElementById('bpc-body');
  bpcBody.innerHTML = '';
  if (bpcNodes.length) {
    for (const node of bpcNodes) {
      const tr = document.createElement('tr');
      const totalRuns = node.bpc_copies_needed * node.max_runs_per_bpc;
      tr.innerHTML = `
        <td>${node.name}</td>
        <td class="num">${fmtNum(node.bpc_copies_needed)}</td>
        <td class="num muted">${fmtNum(node.max_runs_per_bpc)}</td>
        <td class="num muted">${fmtNum(totalRuns)}</td>
      `;
      bpcBody.appendChild(tr);
    }
  } else {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td colspan="4" class="muted">No intermediate blueprints</td>';
    bpcBody.appendChild(tr);
  }

  // BOM tree
  const bomTree = document.getElementById('bom-tree');
  bomTree.innerHTML = '';
  for (const node of data.bom_tree) {
    bomTree.appendChild(renderBOMNode(node, false));
  }
}

function renderCompare(data) {
  const db = data.direct_buy;
  const co = data.compressed_ore;

  // True net ore cost: purchase + refining fee - leftover net credit
  const oreNetTotal = co.total_isk + co.refining_fee - (co.leftover_net_isk || 0);
  const directWins = db.total_isk <= oreNetTotal;
  const savings = Math.abs(db.total_isk - oreNetTotal);
  const savingsPct = db.total_isk > 0 ? (savings / db.total_isk * 100).toFixed(1) : '0';

  const summaryBody = document.getElementById('compare-summary-body');
  summaryBody.innerHTML = '';

  const hasLogistics = (co.leftover_logistics_isk || 0) > 0;
  const oreNetMaterial = co.total_isk - (co.leftover_net_isk || 0);  // excl. refining fee

  const rows = [
    { label: 'Net material cost',     direct: fmtISK(db.total_isk),    ore: fmtISK(oreNetMaterial),           isNet: true, cls: 'isk' },
    { label: '  ore purchase',        direct: '',                        ore: fmtISK(co.total_isk),             sub: true,   cls: 'isk' },
    { label: '  leftover credit',     direct: '',                        ore: '−' + fmtISK(co.leftover_net_isk || 0), sub: true, cls: 'isk' },
  ];

  // Show logistics breakdown sub-rows only when haul cost is non-zero
  if (hasLogistics) {
    rows.push(
      { label: '    sell value',       direct: '', ore: fmtISK(co.leftover_total_isk || 0),     sub: true, indent: true, cls: 'isk' },
      { label: '    − haul cost',      direct: '', ore: '−' + fmtISK(co.leftover_logistics_isk || 0), sub: true, indent: true, cls: 'isk' },
    );
  }

  rows.push(
    { label: 'Refining fee',           direct: '—',                      ore: fmtISK(co.refining_fee),          cls: 'isk' },
    { label: 'Net total',              direct: fmtISK(db.total_isk),    ore: fmtISK(oreNetTotal),              isNet: true, cls: 'isk' },
    { label: 'Volume to haul',         direct: fmtM3(db.total_m3),      ore: fmtM3(co.total_m3),              cls: 'm3'  },
    { label: '  minerals (direct)',    direct: fmtM3(db.total_m3),      ore: '',                               sub: true,   cls: 'm3'  },
    { label: '  compressed ore',       direct: '',                        ore: fmtM3(co.total_m3),              sub: true,   cls: 'm3'  },
    { label: 'Volume after refining',  direct: '—',                      ore: fmtM3(co.refined_total_m3),      cls: 'm3'  },
  );

  for (const row of rows) {
    const tr = document.createElement('tr');
    const dClass = row.isNet && directWins ? 'compare-winner' : '';
    const cClass = row.isNet && !directWins ? 'compare-winner' : '';
    const labelStyle = row.indent
      ? 'color:var(--text-muted);font-size:11px;padding-left:32px'
      : row.sub ? 'color:var(--text-muted);font-size:12px' : '';
    tr.innerHTML = `
      <td style="${labelStyle}">${row.label}</td>
      <td class="num ${row.cls} ${dClass}">${row.direct || ''}</td>
      <td class="num ${row.cls} ${cClass}">${row.ore || ''}</td>
    `;
    summaryBody.appendChild(tr);
  }

  const savingsTr = document.createElement('tr');
  const winner = directWins ? 'Direct buy' : 'Compressed ore';
  savingsTr.innerHTML = `<td class="muted">Savings</td><td colspan="2" class="num green">
    ${winner} cheaper by ${fmtISK(savings)} (${savingsPct}%)</td>`;
  summaryBody.appendChild(savingsTr);

  // Ore breakdown
  const oreBody = document.getElementById('ore-body');
  oreBody.innerHTML = '';
  for (const item of (co.ore_items || [])) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.ore_name}</td>
      <td class="muted">${item.for_mineral_name}</td>
      <td class="num">${fmtNum(item.quantity)}</td>
      <td class="num isk">${fmtISK(item.total_isk)}</td>
      <td class="num yellow">${fmtISK(item.refining_fee)}</td>
      <td class="num green">−${fmtISK(item.byproduct_credit)}</td>
      <td class="num isk">${fmtISK(item.effective_isk)}</td>
      <td class="num m3">${fmtM3(item.volume_m3)}</td>
      <td class="num m3">${fmtM3(item.refined_m3)}</td>
    `;
    oreBody.appendChild(tr);
  }
  for (const item of (co.direct_items || [])) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="muted">(direct buy)</td>
      <td>${item.name}</td>
      <td class="num">${fmtNum(item.quantity)}</td>
      <td class="num isk">${fmtISK(item.total_isk)}</td>
      <td colspan="3" class="muted">no ore source</td>
      <td class="num m3">${fmtM3(item.volume_m3)}</td>
      <td class="muted">—</td>
    `;
    oreBody.appendChild(tr);
  }

  document.getElementById('compare-section').style.display = 'block';
  buildShoppingLists(data);
  renderLeftovers(co);
}

// ── Leftover materials ────────────────────────────────────────────────────────

function renderLeftovers(co) {
  const leftovers = co.leftover_items || [];
  const constraintMet = co.leftover_constraint_met !== false;

  // When constraint not met we already show direct buy price — hide leftover section
  if (!constraintMet || !leftovers.length) {
    document.getElementById('leftover-section').style.display = 'none';
    return;
  }

  const hasLogistics = (co.leftover_logistics_isk || 0) > 0;

  // Table header
  const thead = document.getElementById('leftover-thead');
  if (hasLogistics) {
    thead.innerHTML = `<tr>
      <th>Material</th>
      <th class="num">Quantity</th>
      <th class="num">Jita buy</th>
      <th class="num">Volume</th>
      <th class="num">Haul cost</th>
      <th class="num">Net value</th>
    </tr>`;
  } else {
    thead.innerHTML = `<tr>
      <th>Material</th>
      <th class="num">Quantity</th>
      <th class="num">Jita buy price</th>
      <th class="num">Total ISK</th>
    </tr>`;
  }

  // Table body
  const tbody = document.getElementById('leftover-body');
  tbody.innerHTML = '';
  for (const item of leftovers) {
    const tr = document.createElement('tr');
    if (hasLogistics) {
      tr.innerHTML = `
        <td>${item.name}</td>
        <td class="num">${fmtNum(item.quantity)}</td>
        <td class="num isk">${fmtISK(item.buy_price)}</td>
        <td class="num m3">${fmtM3(item.volume_m3)}</td>
        <td class="num yellow">−${fmtISK(item.logistics_isk)}</td>
        <td class="num green">${fmtISK(item.net_isk)}</td>
      `;
    } else {
      tr.innerHTML = `
        <td>${item.name}</td>
        <td class="num">${fmtNum(item.quantity)}</td>
        <td class="num isk">${fmtISK(item.buy_price)}</td>
        <td class="num green">${fmtISK(item.total_isk)}</td>
      `;
    }
    tbody.appendChild(tr);
  }

  // Table footer
  const tfoot = document.getElementById('leftover-tfoot');
  if (hasLogistics) {
    tfoot.innerHTML = `
      <tr>
        <td colspan="5" style="text-align:right; color:var(--text-muted); font-size:12px">Gross sell value</td>
        <td class="num isk">${fmtISK(co.leftover_total_isk)}</td>
      </tr>
      <tr>
        <td colspan="5" style="text-align:right; color:var(--text-muted); font-size:12px">Haul cost</td>
        <td class="num yellow">−${fmtISK(co.leftover_logistics_isk)}</td>
      </tr>
      <tr>
        <td colspan="5" style="text-align:right; color:var(--text-muted); font-size:12px">Net leftover credit</td>
        <td class="num green">${fmtISK(co.leftover_net_isk)}</td>
      </tr>`;
  } else {
    tfoot.innerHTML = `
      <tr>
        <td colspan="3" style="text-align:right; color:var(--text-muted); font-size:12px">Total leftover value</td>
        <td class="num green">${fmtISK(co.leftover_total_isk)}</td>
      </tr>`;
  }

  document.getElementById('leftover-section').style.display = 'block';
}

// ── Shopping list ─────────────────────────────────────────────────────────────

let _shopLists = { direct: '', ore: '' };
let _activeShopTab = 'ore';

function buildShoppingLists(compareData) {
  const db = compareData.direct_buy;
  const co = compareData.compressed_ore;

  _shopLists.direct = db.items
    .map(i => `${i.name} x ${i.quantity}`)
    .join('\n');

  const oreMap = new Map();
  for (const item of (co.ore_items || [])) {
    oreMap.set(item.ore_name, (oreMap.get(item.ore_name) || 0) + item.quantity);
  }
  for (const item of (co.direct_items || [])) {
    oreMap.set(item.name, (oreMap.get(item.name) || 0) + item.quantity);
  }
  const buyLines = [...oreMap.entries()].map(([name, qty]) => `${name} x ${qty}`).join('\n');

  const leftovers = co.leftover_items || [];
  const sellLines = leftovers.length
    ? '\n-- Sell leftovers --\n' + leftovers.map(i => `${i.name} x ${i.quantity}`).join('\n')
    : '';

  _shopLists.ore = buyLines + sellLines;

  // True net cost comparison determines cheaper badge
  const oreNetTotal = co.total_isk + co.refining_fee - (co.leftover_net_isk || 0);
  const directCost = db.total_isk;
  const oreWins = oreNetTotal <= directCost;

  function tabHTML(label, displayCost, isCheaper) {
    return `<span class="shop-tab-label">${label}</span>`
      + `<span class="shop-tab-cost ${isCheaper ? 'green' : ''}">${fmtISK(displayCost)}${isCheaper ? ' ✓' : ''}</span>`;
  }

  document.getElementById('tab-ore').innerHTML    = tabHTML('Compressed ore', co.total_isk, oreWins);
  document.getElementById('tab-direct').innerHTML = tabHTML('Direct buy',     directCost,   !oreWins);

  _activeShopTab = oreWins ? 'ore' : 'direct';
  document.getElementById('shopping-list').value = _shopLists[_activeShopTab];
  document.getElementById('tab-ore').classList.toggle('active',    _activeShopTab === 'ore');
  document.getElementById('tab-direct').classList.toggle('active', _activeShopTab === 'direct');
  document.getElementById('shopping-section').style.display = 'block';
}

function switchShopTab(tab) {
  _activeShopTab = tab;
  document.getElementById('shopping-list').value = _shopLists[tab];
  document.getElementById('tab-ore').classList.toggle('active',    tab === 'ore');
  document.getElementById('tab-direct').classList.toggle('active', tab === 'direct');
}

function copyShoppingList() {
  const text = document.getElementById('shopping-list').value;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1500);
  });
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function setLoading(on) {
  const btn = document.getElementById('calc-btn');
  btn.disabled = on;
  btn.innerHTML = on ? '<span class="spinner"></span>Calculating…' : 'Calculate';
}

function showError(msg) {
  const box = document.getElementById('error-box');
  box.textContent = msg;
  box.style.display = 'block';
}

function clearError() {
  const box = document.getElementById('error-box');
  box.textContent = '';
  box.style.display = 'none';
}

// Enter key on inputs triggers calculate
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && document.activeElement.tagName === 'INPUT') {
    const listOpen = document.querySelector('.autocomplete-list.open');
    if (!listOpen) calculate();
  }
});
