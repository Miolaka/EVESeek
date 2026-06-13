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

function makeAutocomplete({ inputId, listId, hiddenId, endpoint, labelKey, valueKey, extraLabel, onSelect }) {
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
        if (onSelect) onSelect(item);
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
  onSelect: () => { bpoMeOverrides = {}; document.getElementById('bpo-section').style.display = 'none'; },
});

makeAutocomplete({
  inputId: 'system-search', listId: 'system-list', hiddenId: 'system-id',
  endpoint: 'search-systems', labelKey: 'name', valueKey: 'system_id',
  extraLabel: item => item.security != null ? item.security.toFixed(1) : '',
});

// ── Per-BPO ME overrides (persists across recalculates, resets on item change) ─
let bpoMeOverrides = {};
let lastBuiltTypeId = null;

// ── BPC section ───────────────────────────────────────────────────────────────

let _userBpcCosts = {};   // type_id → user-entered ISK for non-copyable BPCs

function renderBpcList(bpoList, bpcTotalCalc) {
  const tbody = document.getElementById('bpc-body');
  const tfoot = document.getElementById('bpc-foot');
  tbody.innerHTML = '';
  tfoot.innerHTML = '';
  if (!bpoList || !bpoList.length) return;

  let calcTotal = bpcTotalCalc || 0;
  let hasUserInput = false;

  for (const b of bpoList) {
    const isReact = b.activity_id === 11;
    const actLabel = isReact
      ? '<span class="muted">Reaction</span>'
      : b.is_root ? 'Manufacturing <span class="muted">(hull)</span>' : 'Manufacturing';

    let costCell;
    if (isReact) {
      costCell = '<span class="muted">BPO</span>';
    } else if (!b.is_copyable) {
      hasUserInput = true;
      const saved = _userBpcCosts[b.type_id] || '';
      costCell = `<input type="number" class="bpc-cost-input" min="0" step="1000000"
        placeholder="Enter cost…" value="${saved}"
        data-type-id="${b.type_id}">`;
    } else {
      costCell = `<span class="isk">${fmtISK(b.copy_cost)}</span>`;
    }

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${b.name}</td>
      <td class="muted" style="font-size:12px">${actLabel}</td>
      <td class="num">${fmtNum(b.total_runs)}</td>
      <td class="num">${fmtNum(b.copies_needed)}</td>
      <td class="num muted">${fmtNum(b.runs_per_copy)}</td>
      <td class="num">${costCell}</td>`;
    tbody.appendChild(tr);

    if (!b.is_copyable && !isReact) {
      tr.querySelector('.bpc-cost-input').addEventListener('input', e => {
        const v = parseFloat(e.target.value) || 0;
        _userBpcCosts[b.type_id] = v;
        _refreshBpcTotal(bpoList, bpcTotalCalc);
      });
    }
  }

  _refreshBpcTotal(bpoList, bpcTotalCalc);
}

function _refreshBpcTotal(bpoList, calcTotal) {
  const tfoot = document.getElementById('bpc-foot');
  let userTotal = 0;
  for (const b of bpoList) {
    if (!b.is_copyable && b.activity_id !== 11) {
      userTotal += _userBpcCosts[b.type_id] || 0;
    }
  }
  const grand = (calcTotal || 0) + userTotal;
  tfoot.innerHTML = `
    <tr class="bpc-total-row">
      <td colspan="5" style="text-align:right;color:var(--text-muted);font-size:12px">Total BPC acquisition cost</td>
      <td class="num isk">${fmtISK(grand)}</td>
    </tr>`;
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
    structure_bonus: 0.01,
    logistics_cost_isk_per_m3: parseFloat(document.getElementById('logistics').value) || 0,
    build_t1_hull: document.getElementById('build-t1-hull').checked,
  };

  const cfg = loadConfig();
  const activityMeBonus = computeActivityMeBonus(cfg);
  body.activity_me_bonus = activityMeBonus;

  const parsedTypeId = parseInt(typeId);
  if (parsedTypeId !== lastBuiltTypeId) {
    bpoMeOverrides = {};
    _userBpcCosts = {};
    lastBuiltTypeId = parsedTypeId;
  }
  if (Object.keys(bpoMeOverrides).length > 0) {
    body.me_overrides = bpoMeOverrides;
  }

  // Compare-specific fields
  const compareBody = { ...body };
  compareBody.leftover_logistics_isk_per_m3 = parseFloat(document.getElementById('leftover-logistics').value) || 0;
  const maxLeftoverRaw = document.getElementById('max-leftover-isk').value.trim();
  if (maxLeftoverRaw !== '') {
    compareBody.max_leftover_isk = parseFloat(maxLeftoverRaw);
  }

  // Fire compare request in background — don't await it yet
  const comparePromise = fetch('/api/v1/compare-material-source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(compareBody),
  }).catch(() => null);

  try {
    const buildRes = await fetch('/api/v1/build-cost', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!buildRes.ok) {
      const err = await buildRes.json();
      throw new Error(err.detail || buildRes.statusText);
    }

    const build = await buildRes.json();

    // Render build result immediately — don't wait for compare
    renderBuild(build, null);
    renderBpoList(build.bpo_list || []);
    renderBpcList(build.bpo_list || [], build.bpc_total_copy_cost || 0);
    document.getElementById('results').style.display = 'block';
    setLoading(false);

    // Now await compare and update the compare section when ready
    const compareRes = await comparePromise;
    if (compareRes && compareRes.ok) {
      try {
        const compare = await compareRes.json();
        renderBuild(build, compare);
        renderCompare(compare);
      } catch (_) { /* compare parse failed — leave compare section hidden */ }
    }
  } catch (e) {
    showError(e.message);
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

  // BPC table — now driven from bpo_list; called separately after renderBuild
  // BOM tree
  const bomTree = document.getElementById('bom-tree');
  bomTree.innerHTML = '';
  for (const node of data.bom_tree) {
    bomTree.appendChild(renderBOMNode(node, false));
  }
}

function renderBpoList(bpoList) {
  const section = document.getElementById('bpo-section');
  const tbody = document.getElementById('bpo-body');
  if (!bpoList || bpoList.length === 0) { section.style.display = 'none'; return; }

  tbody.innerHTML = '';
  bpoList.forEach(bpo => {
    const isReaction = bpo.activity_id === 11;
    const tr = document.createElement('tr');
    const actLabel = isReaction ? '<span class="muted">Reaction</span>' : 'Manufacturing';
    const rootNote = bpo.is_root ? ' <span class="muted">(hull)</span>' : '';
    const disabled = isReaction || bpo.is_root ? 'disabled style="opacity:0.4"' : '';
    tr.innerHTML = `
      <td>${bpo.name}${rootNote}</td>
      <td>${actLabel}</td>
      <td class="num">
        <input type="number" class="me-input" min="0" max="10" value="${bpo.me_level}"
          data-type-id="${bpo.type_id}" ${disabled}>
      </td>`;
    tbody.appendChild(tr);

    if (!isReaction && !bpo.is_root) {
      tr.querySelector('.me-input').addEventListener('change', e => {
        bpoMeOverrides[bpo.type_id] = parseInt(e.target.value);
      });
    }
  });

  section.style.display = 'block';
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

// ── Structure config data ─────────────────────────────────────────────────

const STRUCTURE_TYPES = [
  { id: 'npc',     name: 'NPC Station',  rig_size: 0, base_me_mfg: 0.0, base_me_react: 0.0 },
  { id: 'raitaru', name: 'Raitaru',      rig_size: 2, base_me_mfg: 1.0, base_me_react: 0.0 },
  { id: 'azbel',   name: 'Azbel',        rig_size: 3, base_me_mfg: 1.0, base_me_react: 0.0 },
  { id: 'sotiyo',  name: 'Sotiyo',       rig_size: 4, base_me_mfg: 1.0, base_me_react: 0.0 },
  { id: 'athanor', name: 'Athanor',      rig_size: 2, base_me_mfg: 0.0, base_me_react: 0.0 },
  { id: 'tatara',  name: 'Tatara',       rig_size: 3, base_me_mfg: 0.0, base_me_react: 0.0 },
];

const ACTIVITY_SLOTS = [
  { key: 'basic_small_ship',  label: 'Small Ships' },
  { key: 'basic_med_ship',    label: 'Medium Ships' },
  { key: 'basic_large_ship',  label: 'Large Ships' },
  { key: 'adv_small_ship',    label: 'Advanced Small Ships' },
  { key: 'adv_med_ship',      label: 'Advanced Medium Ships' },
  { key: 'adv_large_ship',    label: 'Advanced Large Ships' },
  { key: 'cap_ship',          label: 'Capital Ships' },
  { key: 'cap_comp',          label: 'Capital Components' },
  { key: 'cap_adv_comp',      label: 'Capital Advanced Components' },
  { key: 'adv_comp',          label: 'Advanced Components' },
  { key: 'equipment',         label: 'Modules and Equipment' },
  { key: 'ammo',              label: 'Ammo and Charges' },
  { key: 'drones',            label: 'Drones and Fighters' },
  { key: 'structure',         label: 'Structures & Citadels' },
  { key: 'fuel_blocks',       label: 'Fuel Blocks' },
  { key: 'comp_react',        label: 'Composite Reactions' },
  { key: 'hyb_react',         label: 'Hybrid Reactions' },
  { key: 'bio_react',         label: 'Bio and Gas-Phase Reactions' },
];

// Slots that are reactions (used for base_me_react vs base_me_mfg)
const REACTION_SLOTS = new Set(['comp_react', 'hyb_react', 'bio_react']);

let _rigData = null;

// Pre-load rig data at startup so structure bonuses apply without opening the modal
(async () => {
  try {
    const res = await fetch('/api/v1/config/rigs');
    let data = await res.json();
    _rigData = data.filter(r => r.base_me_pct > 0 && !r.name.includes('Thukker'));
  } catch (e) {
    console.error('Failed to pre-load rig data', e);
    _rigData = [];
  }
})();

// Map rig name → list of activity slots it applies to
function rigActivitySlots(name) {
  if (name.includes('XL-Set Ship Manufacturing'))
    return ['basic_small_ship','basic_med_ship','basic_large_ship',
            'adv_small_ship','adv_med_ship','adv_large_ship','cap_ship'];
  if (name.includes('XL-Set Equipment and Consumable'))
    return ['equipment','ammo'];
  if (name.includes('XL-Set Structure and Component'))
    return ['structure','cap_comp','cap_adv_comp','adv_comp'];
  if (name.includes('Reactor Efficiency'))           // L-Set covers all reactions
    return ['comp_react','hyb_react','bio_react'];
  if (name.includes('Composite Reactor'))            return ['comp_react'];
  if (name.includes('Hybrid Reactor'))               return ['hyb_react'];
  if (name.includes('Biochemical Reactor'))          return ['bio_react'];
  if (name.includes('Basic Small Ship'))             return ['basic_small_ship'];
  if (name.includes('Basic Medium Ship'))            return ['basic_med_ship'];
  if (name.includes('Basic Large Ship'))             return ['basic_large_ship'];
  if (name.includes('Advanced Small Ship'))          return ['adv_small_ship'];
  if (name.includes('Advanced Medium Ship'))         return ['adv_med_ship'];
  if (name.includes('Advanced Large Ship'))          return ['adv_large_ship'];
  if (name.includes('Capital Ship'))                 return ['cap_ship'];
  if (name.includes('Advanced Component'))           return ['adv_comp','cap_adv_comp'];
  if (name.includes('Basic Capital Component'))      return ['cap_comp'];
  if (name.includes('Drone and Fighter'))            return ['drones'];
  if (/Ammunition|Ammo/.test(name))                 return ['ammo'];
  if (name.includes('Equipment') && !name.includes('Consumable')) return ['equipment'];
  if (name.includes('Structure'))                    return ['structure'];
  return [];
}

// ── Config state management (localStorage) ───────────────────────────────────

function defaultConfig() {
  return {
    structures: [
      { id: 's1', name: 'ec',      type: 'sotiyo',  security: 'null', rigs: [null, null, null] },
      { id: 's2', name: 'refinery',type: 'tatara',  security: 'null', rigs: [null, null, null] },
    ],
    assignments: {
      basic_small_ship: 's1', basic_med_ship: 's1', basic_large_ship: 's1',
      adv_small_ship:   's1', adv_med_ship:   's1', adv_large_ship:  's1',
      cap_ship: 's1', cap_comp: 's1', cap_adv_comp: 's1',
      adv_comp: 's1', equipment: 's1', ammo: 's1', drones: 's1', fuel_blocks: 's1',
      comp_react: 's2', hyb_react: 's2', bio_react: 's2',
    },
  };
}

function loadConfig() {
  try {
    const raw = localStorage.getItem('eveseek_structure_config');
    return raw ? JSON.parse(raw) : defaultConfig();
  } catch { return defaultConfig(); }
}

function saveConfigToStorage(cfg) {
  localStorage.setItem('eveseek_structure_config', JSON.stringify(cfg));
}

// Compute effective ME bonus (as fraction 0-1) per activity slot from config
function computeActivityMeBonus(cfg) {
  const bonuses = {};
  for (const slot of ACTIVITY_SLOTS) {
    const structId = cfg.assignments[slot.key];
    const struct   = cfg.structures.find(s => s.id === structId);
    if (!struct) { bonuses[slot.key] = 0; continue; }

    const stype    = STRUCTURE_TYPES.find(t => t.id === struct.type) || STRUCTURE_TYPES[0];
    const isReact  = REACTION_SLOTS.has(slot.key);
    let total      = (isReact ? stype.base_me_react : stype.base_me_mfg) / 100;

    if (_rigData) {
      for (const rigTypeId of struct.rigs) {
        if (!rigTypeId) continue;
        const rig = _rigData.find(r => r.type_id === rigTypeId);
        if (!rig) continue;
        const slots = rigActivitySlots(rig.name);
        if (!slots.includes(slot.key)) continue;
        const secMult = struct.security === 'hi' ? rig.hi_mult
                      : struct.security === 'lo' ? rig.lo_mult
                      : rig.nu_mult;
        total += rig.base_me_pct / 100 * secMult;
      }
    }
    bonuses[slot.key] = total;
  }
  return bonuses;
}

// ── Modal open/close ──────────────────────────────────────────────────────────

async function openConfig() {
  // _rigData is pre-loaded at startup; wait briefly if still loading
  if (!_rigData) {
    await new Promise(r => setTimeout(r, 500));
  }
  _configDraft = JSON.parse(JSON.stringify(loadConfig()));
  renderConfigModal(_configDraft);
  document.getElementById('config-overlay').style.display = 'flex';
}

function closeConfig() {
  document.getElementById('config-overlay').style.display = 'none';
}

function closeConfigOnOverlay(e) {
  if (e.target === document.getElementById('config-overlay')) closeConfig();
}

let _configDraft = null;

function saveConfig() {
  saveConfigToStorage(_configDraft);
  closeConfig();
}

function resetConfig() {
  _configDraft = defaultConfig();
  renderConfigModal(_configDraft);
}

// ── Modal rendering ───────────────────────────────────────────────────────────

function renderConfigModal(cfg) {
  renderStructures(cfg);
  renderAssignments(cfg);
}

function renderStructures(cfg) {
  const container = document.getElementById('config-structures');
  container.innerHTML = '';
  for (const struct of cfg.structures) {
    container.appendChild(buildStructCard(struct, cfg));
  }
}

function buildStructCard(struct, cfg) {
  const stype = STRUCTURE_TYPES.find(t => t.id === struct.type) || STRUCTURE_TYPES[3];
  const rigSize = stype.rig_size;

  // Filter rigs compatible with this structure's rig size
  const compatRigs = _rigData
    ? _rigData.filter(r => r.rig_size === rigSize)
    : [];

  const card = document.createElement('div');
  card.className = 'struct-card';
  card.dataset.structId = struct.id;

  // Header: name input + remove button
  const hdr = document.createElement('div');
  hdr.className = 'struct-card-header';

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.className = 'struct-name-input';
  nameInput.value = struct.name;
  nameInput.addEventListener('input', () => {
    struct.name = nameInput.value;
    refreshAssignmentDropdowns(cfg);
  });

  const removeBtn = document.createElement('button');
  removeBtn.className = 'btn-remove-struct';
  removeBtn.textContent = '✕';
  removeBtn.onclick = () => {
    cfg.structures = cfg.structures.filter(s => s.id !== struct.id);
    // Clear assignments that used this structure
    for (const k of Object.keys(cfg.assignments)) {
      if (cfg.assignments[k] === struct.id) cfg.assignments[k] = cfg.structures[0]?.id || '';
    }
    renderConfigModal(cfg);
  };

  hdr.appendChild(nameInput);
  hdr.appendChild(removeBtn);
  card.appendChild(hdr);

  // Type + security row
  const typeRow = document.createElement('div');
  typeRow.className = 'struct-row';

  const typeSelect = document.createElement('select');
  typeSelect.className = 'struct-select';
  for (const st of STRUCTURE_TYPES) {
    const opt = document.createElement('option');
    opt.value = st.id;
    opt.textContent = st.name;
    if (st.id === struct.type) opt.selected = true;
    typeSelect.appendChild(opt);
  }
  typeSelect.addEventListener('change', () => {
    struct.type = typeSelect.value;
    struct.rigs = [null, null, null];  // reset rigs on structure type change
    renderStructures(cfg);
  });

  const secLabel = document.createElement('span');
  secLabel.className = 'struct-row';
  secLabel.style.cssText = 'font-size:12px;color:#7a8a7a;text-align:center';
  secLabel.textContent = 'Security:';

  const secSelect = document.createElement('select');
  secSelect.className = 'struct-select';
  [['hi','Highsec'],['lo','Lowsec'],['null','Null / WH']].forEach(([v, l]) => {
    const opt = document.createElement('option');
    opt.value = v; opt.textContent = l;
    if (v === struct.security) opt.selected = true;
    secSelect.appendChild(opt);
  });
  secSelect.addEventListener('change', () => { struct.security = secSelect.value; });

  typeRow.appendChild(typeSelect);
  typeRow.appendChild(secLabel);
  typeRow.appendChild(secSelect);
  card.appendChild(typeRow);

  // Rig slots
  for (let i = 0; i < 3; i++) {
    const rigRow = document.createElement('div');
    rigRow.className = 'struct-rig-row';

    const rigLabel = document.createElement('span');
    rigLabel.className = 'struct-rig-label';
    rigLabel.textContent = `Rig ${i + 1}`;

    const rigSelect = document.createElement('select');
    rigSelect.className = 'struct-rig-select';

    const noRig = document.createElement('option');
    noRig.value = ''; noRig.textContent = 'No Rig';
    rigSelect.appendChild(noRig);

    for (const rig of compatRigs) {
      const opt = document.createElement('option');
      opt.value = rig.type_id;
      // Format: "XL-Set Ship Mfg I  (null: 4.2%)"
      const nuEff = (rig.base_me_pct / 100 * rig.nu_mult * 100).toFixed(1);
      opt.textContent = `${rig.name}  (null: ${nuEff}%)`;
      if (rig.type_id === struct.rigs[i]) opt.selected = true;
      rigSelect.appendChild(opt);
    }

    rigSelect.addEventListener('change', () => {
      struct.rigs[i] = rigSelect.value ? parseInt(rigSelect.value) : null;
    });

    rigRow.appendChild(rigLabel);
    rigRow.appendChild(rigSelect);
    card.appendChild(rigRow);
  }

  return card;
}

function renderAssignments(cfg) {
  const container = document.getElementById('config-assignments');
  container.innerHTML = '';

  for (const slot of ACTIVITY_SLOTS) {
    const row = document.createElement('div');
    row.className = 'assign-row';

    const label = document.createElement('span');
    label.className = 'assign-label';
    label.textContent = slot.label;

    const sel = document.createElement('select');
    sel.className = 'assign-select';
    sel.dataset.slot = slot.key;

    for (const struct of cfg.structures) {
      const opt = document.createElement('option');
      opt.value = struct.id;
      opt.textContent = struct.name || struct.id;
      if (cfg.assignments[slot.key] === struct.id) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => {
      cfg.assignments[slot.key] = sel.value;
    });

    row.appendChild(label);
    row.appendChild(sel);
    container.appendChild(row);
  }
}

function refreshAssignmentDropdowns(cfg) {
  const container = document.getElementById('config-assignments');
  if (!container) return;
  const selects = container.querySelectorAll('select.assign-select');
  for (const sel of selects) {
    const current = cfg.assignments[sel.dataset.slot];
    sel.innerHTML = '';
    for (const struct of cfg.structures) {
      const opt = document.createElement('option');
      opt.value = struct.id;
      opt.textContent = struct.name || struct.id;
      if (struct.id === current) opt.selected = true;
      sel.appendChild(opt);
    }
  }
}

function addStructure() {
  const id = 's' + Date.now();
  _configDraft.structures.push({
    id, name: 'Structure ' + _configDraft.structures.length,
    type: 'sotiyo', security: 'null', rigs: [null, null, null],
  });
  renderConfigModal(_configDraft);
}

function autoAssign() {
  if (!_rigData || !_configDraft) return;

  for (const slot of ACTIVITY_SLOTS) {
    let bestId    = _configDraft.structures[0]?.id || '';
    let bestBonus = -Infinity;

    for (const struct of _configDraft.structures) {
      const stype = STRUCTURE_TYPES.find(t => t.id === struct.type);
      const isR   = REACTION_SLOTS.has(slot.key);
      let bonus   = stype ? (isR ? stype.base_me_react : stype.base_me_mfg) / 100 : 0;

      for (const rigId of struct.rigs) {
        if (!rigId) continue;
        const rig = _rigData.find(r => r.type_id === rigId);
        if (!rig) continue;
        if (!rigActivitySlots(rig.name).includes(slot.key)) continue;
        const sm = struct.security === 'hi' ? rig.hi_mult
                 : struct.security === 'lo' ? rig.lo_mult : rig.nu_mult;
        bonus += rig.base_me_pct / 100 * sm;
      }

      if (bonus > bestBonus) { bestBonus = bonus; bestId = struct.id; }
    }

    _configDraft.assignments[slot.key] = bestId;
  }

  renderAssignments(_configDraft);
}

// Enter key on inputs triggers calculate
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && document.activeElement.tagName === 'INPUT') {
    const listOpen = document.querySelector('.autocomplete-list.open');
    if (!listOpen) calculate();
  }
});
