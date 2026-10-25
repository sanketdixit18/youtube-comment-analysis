/* ==========================================================================
   YouTube Sentiment Analyzer - front-end logic (vanilla JavaScript)

   Security note: every piece of text that comes from the server or from a
   YouTube comment is inserted with textContent / createElement, never with
   innerHTML, so a malicious comment such as "<img onerror=...>" is shown as
   harmless text.
   ========================================================================== */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ *
   * Constants & state
   * ------------------------------------------------------------------ */
  const SENTIMENTS = ['Positive', 'Negative', 'Neutral'];
  const CLS = { Positive: 'pos', Negative: 'neg', Neutral: 'neu' };
  const ICON = { Positive: '▲', Negative: '▼', Neutral: '●' };
  const THEME_KEY = 'yt-sentiment-theme';
  const EXAMPLE_URL = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ';
  const API = {
    status: '/api/status',
    modelInfo: '/api/model-info',
    analyze: '/api/analyze',
    demo: '/api/demo',
    comment: '/api/analyze-comment',
  };

  const state = {
    data: null,            // last analysis result from the server
    filter: 'All',
    query: '',
    sort: 'original',
    page: 1,
    pageSize: 10,
    wordSource: 'overall_words',
  };
  const charts = { doughnut: null, bar: null, words: null };
  const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ------------------------------------------------------------------ *
   * Small helpers
   * ------------------------------------------------------------------ */
  const $ = (id) => document.getElementById(id);

  function el(tag, opts, children) {
    const node = document.createElement(tag);
    const o = opts || {};
    if (o.className) node.className = o.className;
    if (o.text !== undefined && o.text !== null) node.textContent = o.text;
    if (o.attrs) Object.keys(o.attrs).forEach((k) => node.setAttribute(k, o.attrs[k]));
    if (o.style) Object.keys(o.style).forEach((k) => { node.style[k] = o.style[k]; });
    (children || []).forEach((c) => { if (c) node.appendChild(c); });
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  const fmtInt = (n) => Number(n).toLocaleString('en-US');
  const fmtPct = (fraction, digits) => (fraction * 100).toFixed(digits === undefined ? 0 : digits) + '%';
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function debounce(fn, wait) {
    let timer = null;
    return function () {
      const args = arguments;
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(null, args), wait);
    };
  }

  /* ------------------------------------------------------------------ *
   * Toast notifications
   * ------------------------------------------------------------------ */
  function toast(message, type, duration) {
    const box = $('toasts');
    while (box.children.length >= 4) box.removeChild(box.firstChild);

    const node = el('div', {
      className: 'toast toast-' + (type || 'info'),
      attrs: { role: type === 'error' ? 'alert' : 'status' },
    });
    node.appendChild(el('span', { className: 'toast-msg', text: message }));
    const close = el('button', {
      className: 'toast-close', text: '×',
      attrs: { type: 'button', 'aria-label': 'Dismiss notification' },
    });
    const remove = () => {
      node.classList.add('is-leaving');
      setTimeout(() => { if (node.parentNode) node.parentNode.removeChild(node); }, 260);
    };
    close.addEventListener('click', remove);
    node.appendChild(close);
    box.appendChild(node);
    setTimeout(remove, duration || 4500);
  }

  /* ------------------------------------------------------------------ *
   * Theme (dark / light) - saved in localStorage
   * ------------------------------------------------------------------ */
  function applyTheme(theme, save) {
    document.documentElement.setAttribute('data-theme', theme);
    if (save) {
      try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* storage blocked - ignore */ }
    }
    $('themeToggle').setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
    if (state.data) renderCharts(false);          // charts read colours from CSS variables
  }

  function initTheme() {
    const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    applyTheme(theme, false);
    $('themeToggle').addEventListener('click', () => {
      const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      applyTheme(next, true);
    });
  }

  /* ------------------------------------------------------------------ *
   * Navigation (mobile menu + highlight current section)
   * ------------------------------------------------------------------ */
  function initNav() {
    const nav = $('siteNav');
    const toggle = $('menuToggle');
    const setOpen = (open) => {
      nav.classList.toggle('is-open', open);
      toggle.setAttribute('aria-expanded', String(open));
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    };
    toggle.addEventListener('click', () => setOpen(!nav.classList.contains('is-open')));
    nav.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => setOpen(false)));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });

    if (!('IntersectionObserver' in window)) return;
    const links = {};
    nav.querySelectorAll('a').forEach((a) => { links[a.getAttribute('href').slice(1)] = a; });
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        Object.keys(links).forEach((id) => links[id].classList.toggle('is-current', id === entry.target.id));
      });
    }, { rootMargin: '-35% 0px -60% 0px' });
    Object.keys(links).forEach((id) => { const s = $(id); if (s) observer.observe(s); });
  }

  /* ------------------------------------------------------------------ *
   * API helper
   * ------------------------------------------------------------------ */
  async function api(path, options) {
    const opts = options || {};
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), opts.timeout || 60000);
    const init = { method: opts.method || 'GET', headers: { Accept: 'application/json' }, signal: controller.signal };
    if (opts.body !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }

    let response;
    try {
      response = await fetch(path, init);
    } catch (e) {
      const err = new Error(e && e.name === 'AbortError'
        ? 'The request took too long. Please try again.'
        : 'Cannot reach the server. Make sure it is still running.');
      err.code = 'network';
      throw err;
    } finally {
      clearTimeout(timer);
    }

    let data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    if (!response.ok) {
      const err = new Error((data && data.error) || 'Request failed (' + response.status + ').');
      err.code = data && data.code;
      err.status = response.status;
      throw err;
    }
    if (data === null) throw new Error('The server sent an unreadable response.');
    return data;
  }

  /* ------------------------------------------------------------------ *
   * Server status pill
   * ------------------------------------------------------------------ */
  async function loadStatus() {
    const pill = $('apiPill');
    try {
      const s = await api(API.status);
      if (!s.model_loaded) {
        pill.className = 'pill pill-bad';
        pill.textContent = 'Model not loaded';
        showError(s.model_error || 'The sentiment model is not loaded. Run "python ml/train.py" and restart the server.');
      } else if (s.youtube_api_configured) {
        pill.className = 'pill pill-ok';
        pill.textContent = 'YouTube API connected';
      } else {
        pill.className = 'pill pill-warn';
        pill.textContent = 'No API key – demo data only';
        $('modeHint').appendChild(document.createTextNode(
          ' No YouTube API key is configured, so Analyze Video will show the sample dataset instead.'));
      }
    } catch (e) {
      pill.className = 'pill pill-bad';
      pill.textContent = 'Server unreachable';
    }
  }

  /* ------------------------------------------------------------------ *
   * Errors + loading state
   * ------------------------------------------------------------------ */
  function showError(message) {
    const box = $('errorBox');
    box.textContent = message;
    box.hidden = false;
  }

  function hideError() {
    const box = $('errorBox');
    box.hidden = true;
    box.textContent = '';
  }

  let loadingTimer = null;
  function setBusy(busy, label) {
    ['analyzeBtn', 'demoBtn', 'emptyDemoBtn', 'exampleBtn'].forEach((id) => { $(id).disabled = busy; });
    $('analyzeForm').setAttribute('aria-busy', String(busy));
    $('loading').hidden = !busy;
    clearInterval(loadingTimer);
    if (!busy) return;

    $('loadingTitle').textContent = label;
    const steps = Array.prototype.slice.call($('loadingSteps').children);
    let index = 0;
    const tick = () => {
      steps.forEach((step, i) => { step.className = i < index ? 'is-done' : (i === index ? 'is-active' : ''); });
      index = Math.min(index + 1, steps.length - 1);
    };
    tick();
    loadingTimer = setInterval(tick, 650);
  }

  /* ------------------------------------------------------------------ *
   * Running an analysis (live URL or demo)
   * ------------------------------------------------------------------ */
  async function runAnalysis(kind) {
    hideError();
    let path, body, label;

    if (kind === 'demo') {
      path = API.demo;
      label = 'Analyzing the sample comments…';
    } else {
      const url = $('urlInput').value.trim();
      if (!url) {
        showError('Please enter a YouTube video URL, or press Demo Mode to use sample comments.');
        $('urlInput').focus();
        return;
      }
      path = API.analyze;
      body = { url: url, max_comments: Number($('maxComments').value), allow_fallback: true };
      label = 'Fetching and analyzing comments…';
    }

    setBusy(true, label);
    const started = Date.now();
    try {
      const data = await api(path, { method: 'POST', body: body, timeout: 90000 });
      const elapsed = Date.now() - started;
      if (elapsed < 900) await sleep(900 - elapsed);     // let the loading animation be seen
      render(data);
      if (data.fallback) {
        toast('Showing demo data instead of live comments.', 'warn', 6000);
      } else {
        toast('Analysis complete: ' + fmtInt(data.summary.total) + ' comments.', 'success');
      }
    } catch (e) {
      showError(e.message);
      toast(e.message, 'error', 6000);
    } finally {
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------------ *
   * Rendering the results
   * ------------------------------------------------------------------ */
  function render(data) {
    state.data = data;
    state.filter = 'All';
    state.query = '';
    state.sort = 'original';
    state.page = 1;
    state.wordSource = 'overall_words';

    $('searchInput').value = '';
    $('sortSelect').value = 'original';
    syncFilterChips();
    syncWordTabs();

    $('emptyState').hidden = true;
    $('results').hidden = false;

    renderHeader(data);
    renderSummary(data.summary);
    renderPercentages(data.summary);
    renderCharts(true);
    renderInsights(data.insights);
    renderTable();

    $('results').scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
  }

  function renderHeader(data) {
    const badge = $('modeBadge');
    badge.className = 'badge ' + (data.mode === 'live' ? 'badge-live' : 'badge-demo');
    badge.textContent = data.mode === 'live' ? 'Live YouTube API data' : 'Demo / sample data';

    const notice = $('noticeBox');
    if (data.notice) {
      notice.textContent = data.notice;
      notice.hidden = false;
    } else {
      notice.hidden = true;
      notice.textContent = '';
    }

    const video = data.video || {};
    $('videoTitle').textContent = video.title || 'Untitled video';
    const meta = [];
    if (video.channel) meta.push(video.channel);
    if (video.comment_count) meta.push(fmtInt(video.comment_count) + ' comments on YouTube');
    meta.push(fmtInt(data.summary.total) + ' comments analyzed');
    $('videoMeta').textContent = meta.join(' · ');

    const thumb = $('videoThumb');
    let safeThumb = false;
    try {
      const u = new URL(video.thumbnail || '');
      safeThumb = u.protocol === 'https:' && /(^|\.)ytimg\.com$/.test(u.hostname);
    } catch (e) { safeThumb = false; }
    if (safeThumb) {
      thumb.src = video.thumbnail;
      thumb.hidden = false;
    } else {
      thumb.removeAttribute('src');
      thumb.hidden = true;
    }
  }

  function setNumber(node, value) {
    if (reducedMotion || value < 2) {
      node.textContent = fmtInt(value);
      return;
    }
    const start = performance.now();
    const duration = 650;
    const frame = (now) => {
      const t = Math.min(1, (now - start) / duration);
      node.textContent = fmtInt(Math.round(value * (1 - Math.pow(1 - t, 3))));
      if (t < 1) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  function renderSummary(summary) {
    const c = summary.counts;
    setNumber($('statTotal'), summary.total);
    setNumber($('statPositive'), c.Positive);
    setNumber($('statNegative'), c.Negative);
    setNumber($('statNeutral'), c.Neutral);
    $('statTotalSub').textContent = 'Average confidence ' + fmtPct(summary.average_confidence);
    $('statPositiveSub').textContent = summary.percentages.Positive.toFixed(1) + '% of comments';
    $('statNegativeSub').textContent = summary.percentages.Negative.toFixed(1) + '% of comments';
    $('statNeutralSub').textContent = summary.percentages.Neutral.toFixed(1) + '% of comments';

    const overallClass = { Positive: 'is-pos', Negative: 'is-neg', Neutral: 'is-neu', Mixed: 'is-mixed' }[summary.overall] || '';
    const card = $('statOverallCard');
    card.classList.remove('is-pos', 'is-neg', 'is-neu', 'is-mixed');
    if (overallClass) card.classList.add(overallClass);
    $('statOverall').textContent = summary.overall;
    const net = summary.net_score;
    $('statOverallSub').textContent = 'Net score ' + (net > 0 ? '+' : '') + net.toFixed(2);
    $('statOverallSub').title = 'Net score = (positive − negative) ÷ total comments';
  }

  function renderPercentages(summary) {
    const list = $('percentList');
    clear(list);
    const bars = [];
    SENTIMENTS.forEach((s) => {
      const pct = summary.percentages[s];
      const fill = el('div', { className: 'percent-fill fill-' + CLS[s] });
      bars.push([fill, pct]);
      const row = el('div', { className: 'percent-row' }, [
        el('span', { className: 'percent-label' }, [
          el('span', { className: 'dot dot-' + CLS[s] }),
          el('span', { text: s + ' — ' + pct.toFixed(1) + '%' }),
        ]),
        el('span', { className: 'percent-count', text: fmtInt(summary.counts[s]) + ' comments' }),
        el('div', { className: 'percent-track', attrs: { role: 'presentation' } }, [fill]),
      ]);
      list.appendChild(row);
    });
    // animate the bars after they are in the DOM
    setTimeout(() => bars.forEach((b) => { b[0].style.width = b[1] + '%'; }), 40);
    $('avgConfidence').textContent =
      'Average model confidence across all comments: ' + fmtPct(summary.average_confidence) + '.';
  }

  /* ---------- charts ---------- */
  function chartTheme() {
    return {
      pos: cssVar('--pos'), neg: cssVar('--neg'), neu: cssVar('--neu'),
      text: cssVar('--text'), muted: cssVar('--muted'), grid: cssVar('--border'),
      surface: cssVar('--surface'), primary: cssVar('--primary'),
      font: getComputedStyle(document.body).fontFamily,
    };
  }

  const centerTextPlugin = {
    id: 'centerText',
    afterDraw(chart, args, opts) {
      const meta = chart.getDatasetMeta(0);
      if (!meta || !meta.data.length) return;
      const x = meta.data[0].x;
      const y = meta.data[0].y;
      const ctx = chart.ctx;
      ctx.save();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = opts.color;
      ctx.font = '800 30px ' + opts.font;
      ctx.fillText(opts.total, x, y - 8);
      ctx.fillStyle = opts.muted;
      ctx.font = '500 12px ' + opts.font;
      ctx.fillText('comments', x, y + 16);
      ctx.restore();
    },
  };

  const barLabelPlugin = {
    id: 'barLabels',
    afterDatasetsDraw(chart, args, opts) {
      const ctx = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      ctx.save();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = opts.color;
      ctx.font = '700 14px ' + opts.font;
      meta.data.forEach((bar, i) => ctx.fillText(fmtInt(chart.data.datasets[0].data[i]), bar.x, bar.y - 6));
      ctx.restore();
    },
  };

  function renderCharts(animate) {
    if (typeof Chart === 'undefined') {
      toast('The chart library could not be loaded, so charts are hidden.', 'warn');
      return;
    }
    const data = state.data;
    if (!data) return;
    const t = chartTheme();
    const counts = SENTIMENTS.map((s) => data.summary.counts[s]);
    const colors = [t.pos, t.neg, t.neu];
    const total = data.summary.total;

    Chart.defaults.font.family = t.font;
    Chart.defaults.color = t.muted;
    const animation = (animate && !reducedMotion) ? { duration: 800 } : false;

    // ----- doughnut -----
    if (charts.doughnut) charts.doughnut.destroy();
    charts.doughnut = new Chart($('doughnutChart'), {
      type: 'doughnut',
      data: {
        labels: SENTIMENTS,
        datasets: [{ data: counts, backgroundColor: colors, borderColor: t.surface, borderWidth: 3, hoverOffset: 8 }],
      },
      plugins: [centerTextPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, cutout: '68%', animation: animation,
        plugins: {
          legend: { position: 'bottom', labels: { color: t.text, usePointStyle: true, pointStyle: 'circle', padding: 18, font: { size: 13, weight: '600' } } },
          tooltip: {
            callbacks: {
              label: (item) => ' ' + item.label + ': ' + fmtInt(item.parsed) + ' (' + (item.parsed * 100 / total).toFixed(1) + '%)',
            },
          },
          centerText: { total: fmtInt(total), color: t.text, muted: t.muted, font: t.font },
        },
      },
    });
    $('doughnutChart').setAttribute('aria-label',
      'Doughnut chart: ' + SENTIMENTS.map((s, i) => s + ' ' + counts[i]).join(', '));

    // ----- bar -----
    if (charts.bar) charts.bar.destroy();
    charts.bar = new Chart($('barChart'), {
      type: 'bar',
      data: { labels: SENTIMENTS, datasets: [{ label: 'Comments', data: counts, backgroundColor: colors, borderRadius: 10, borderSkipped: false, maxBarThickness: 90 }] },
      plugins: [barLabelPlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: animation,
        layout: { padding: { top: 26 } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (item) => ' ' + fmtInt(item.parsed.y) + ' comments' } },
          barLabels: { color: t.text, font: t.font },
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: t.text, font: { size: 13, weight: '600' } }, border: { color: t.grid } },
          y: { beginAtZero: true, ticks: { precision: 0, color: t.muted }, grid: { color: t.grid }, border: { display: false } },
        },
      },
    });
    $('barChart').setAttribute('aria-label',
      'Bar chart: ' + SENTIMENTS.map((s, i) => s + ' ' + counts[i]).join(', '));

    renderWordChart(animate);
  }

  function renderWordChart(animate) {
    if (typeof Chart === 'undefined' || !state.data) return;
    const t = chartTheme();
    const words = (state.data.insights[state.wordSource] || []);
    const canvas = $('wordChart');
    const emptyMsg = $('wordChartEmpty');
    if (charts.words) { charts.words.destroy(); charts.words = null; }

    canvas.hidden = words.length === 0;
    emptyMsg.hidden = words.length !== 0;
    if (!words.length) return;

    const color = state.wordSource === 'positive_words' ? t.pos : (state.wordSource === 'negative_words' ? t.neg : t.primary);
    charts.words = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: words.map((w) => w.word),
        datasets: [{ label: 'Comments containing the word', data: words.map((w) => w.count), backgroundColor: color, borderRadius: 6, borderSkipped: false, maxBarThickness: 22 }],
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        animation: (animate && !reducedMotion) ? { duration: 600 } : false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (item) => ' ' + item.parsed.x + ' comments' } } },
        scales: {
          x: { beginAtZero: true, ticks: { precision: 0, color: t.muted }, grid: { color: t.grid }, border: { display: false } },
          y: { grid: { display: false }, ticks: { color: t.text, font: { size: 13, weight: '600' } }, border: { color: t.grid } },
        },
      },
    });
    canvas.setAttribute('aria-label', 'Horizontal bar chart of the most frequent words: ' +
      words.map((w) => w.word + ' ' + w.count).join(', '));
  }

  /* ---------- insights ---------- */
  function renderWordList(listNode, words) {
    clear(listNode);
    if (!words.length) {
      listNode.appendChild(el('li', { className: 'list-empty', text: 'No words found for this group.' }));
      return;
    }
    const max = words[0].count || 1;
    words.forEach((w) => {
      listNode.appendChild(el('li', {}, [
        el('div', { className: 'w-main' }, [
          el('span', { className: 'w-name', text: w.word }),
          el('div', { className: 'w-bar' }, [el('span', { style: { width: Math.max(6, (w.count / max) * 100) + '%' } })]),
        ]),
        el('span', { className: 'w-count', text: String(w.count) }),
      ]));
    });
  }

  function renderWordCloud(words) {
    const box = $('wordCloud');
    clear(box);
    if (!words.length) {
      box.appendChild(el('p', { className: 'list-empty', text: 'Not enough words to build a word cloud.' }));
      return;
    }
    const counts = words.map((w) => w.count);
    const max = Math.max.apply(null, counts);
    const min = Math.min.apply(null, counts);
    const minSize = 0.95;
    const maxSize = 2.7;

    // place the biggest words in the middle: [..., 3rd, 1st, 2nd, ...]
    const sorted = words.slice().sort((a, b) => b.count - a.count || a.word.localeCompare(b.word));
    const arranged = [];
    sorted.forEach((w, i) => { if (i % 2 === 0) arranged.push(w); else arranged.unshift(w); });

    arranged.forEach((w) => {
      const scale = max === min ? 0.5 : Math.sqrt((w.count - min) / (max - min));
      const size = minSize + (maxSize - minSize) * scale;
      box.appendChild(el('span', {
        className: 'cw cw-' + CLS[w.sentiment],
        text: w.word,
        attrs: { title: w.word + ': ' + w.count + ' comment' + (w.count === 1 ? '' : 's') + ' (mostly ' + w.sentiment.toLowerCase() + ')' },
        style: { fontSize: size.toFixed(2) + 'rem' },
      }));
    });
  }

  function renderInsights(insights) {
    renderWordList($('posWords'), insights.positive_words);
    renderWordList($('negWords'), insights.negative_words);
    renderWordList($('allWords'), insights.overall_words);
    renderWordCloud(insights.word_cloud);
  }

  function syncWordTabs() {
    document.querySelectorAll('.tab').forEach((tab) => {
      const active = tab.getAttribute('data-words') === state.wordSource;
      tab.classList.toggle('is-active', active);
      tab.setAttribute('aria-selected', String(active));
    });
  }

  /* ---------- comment table ---------- */
  function getFilteredComments() {
    const q = state.query.trim().toLowerCase();
    const list = state.data.comments.filter((c) =>
      (state.filter === 'All' || c.sentiment === state.filter) &&
      (!q || c.text.toLowerCase().indexOf(q) !== -1));

    const order = { Positive: 0, Negative: 1, Neutral: 2 };
    switch (state.sort) {
      case 'conf-desc': list.sort((a, b) => b.confidence - a.confidence || a.id - b.id); break;
      case 'conf-asc': list.sort((a, b) => a.confidence - b.confidence || a.id - b.id); break;
      case 'sentiment': list.sort((a, b) => order[a.sentiment] - order[b.sentiment] || b.confidence - a.confidence); break;
      case 'text': list.sort((a, b) => a.text.localeCompare(b.text, undefined, { sensitivity: 'base' })); break;
      default: break;                      // original order (by id)
    }
    return list;
  }

  function syncFilterChips() {
    const counts = state.data ? state.data.summary.counts : null;
    document.querySelectorAll('#filterChips .chip').forEach((chip) => {
      const f = chip.getAttribute('data-filter');
      const active = f === state.filter;
      chip.classList.toggle('is-active', active);
      chip.setAttribute('aria-pressed', String(active));
      chip.classList.remove('chip-pos', 'chip-neg', 'chip-neu');
      if (f !== 'All') chip.classList.add('chip-' + CLS[f]);
      chip.textContent = f === 'All'
        ? (counts ? 'All (' + fmtInt(state.data.summary.total) + ')' : 'All')
        : (counts ? f + ' (' + fmtInt(counts[f]) + ')' : f);
    });
  }

  function renderTable() {
    if (!state.data) return;
    const list = getFilteredComments();
    const totalAll = state.data.comments.length;
    const pages = Math.max(1, Math.ceil(list.length / state.pageSize));
    state.page = Math.min(Math.max(1, state.page), pages);
    const start = (state.page - 1) * state.pageSize;
    const slice = list.slice(start, start + state.pageSize);

    const body = $('commentsBody');
    clear(body);
    const colors = { Positive: 'var(--pos)', Negative: 'var(--neg)', Neutral: 'var(--neu)' };
    slice.forEach((c) => {
      body.appendChild(el('tr', {}, [
        el('td', { className: 'col-num', text: String(c.id) }),
        el('td', { className: 'cell-text', text: c.text }),
        el('td', { className: 'cell-sent' }, [
          el('span', { className: 'badge badge-' + CLS[c.sentiment] }, [
            el('span', { text: ICON[c.sentiment], attrs: { 'aria-hidden': 'true' } }),
            el('span', { text: c.sentiment }),
          ]),
        ]),
        el('td', { className: 'cell-conf' }, [
          el('div', { className: 'conf' }, [
            el('b', { text: fmtPct(c.confidence) }),
            el('div', { className: 'conf-bar', attrs: { 'aria-hidden': 'true' } }, [
              el('span', { style: { width: (c.confidence * 100).toFixed(0) + '%', background: colors[c.sentiment] } }),
            ]),
          ]),
        ]),
      ]));
    });

    const empty = list.length === 0;
    document.querySelector('#comments .table-wrap').hidden = empty;
    $('tableEmpty').hidden = !empty;

    const filtered = list.length !== totalAll;
    $('pageInfo').textContent = empty
      ? '0 comments'
      : 'Showing ' + fmtInt(start + 1) + '–' + fmtInt(start + slice.length) + ' of ' + fmtInt(list.length) +
        ' comments' + (filtered ? ' (filtered from ' + fmtInt(totalAll) + ')' : '');
    renderPagination(pages);
  }

  function renderPagination(pages) {
    const box = $('pageButtons');
    clear(box);
    if (pages <= 1) return;

    const addButton = (label, page, opts) => {
      const o = opts || {};
      const btn = el('button', { className: 'page-btn', text: label, attrs: { type: 'button' } });
      if (o.ariaLabel) btn.setAttribute('aria-label', o.ariaLabel);
      if (o.current) btn.setAttribute('aria-current', 'page');
      if (o.disabled) btn.disabled = true;
      btn.addEventListener('click', () => { state.page = page; renderTable(); });
      box.appendChild(btn);
    };

    addButton('‹', state.page - 1, { disabled: state.page === 1, ariaLabel: 'Previous page' });
    const first = Math.max(1, Math.min(state.page - 2, pages - 4));
    const last = Math.min(pages, first + 4);
    if (first > 1) {
      addButton('1', 1, { ariaLabel: 'Page 1' });
      if (first > 2) box.appendChild(el('span', { className: 'muted', text: '…' }));
    }
    for (let p = first; p <= last; p++) {
      addButton(String(p), p, { current: p === state.page, ariaLabel: 'Page ' + p });
    }
    if (last < pages) {
      if (last < pages - 1) box.appendChild(el('span', { className: 'muted', text: '…' }));
      addButton(String(pages), pages, { ariaLabel: 'Page ' + pages });
    }
    addButton('›', state.page + 1, { disabled: state.page === pages, ariaLabel: 'Next page' });
  }

  function csvCell(value) {
    let v = String(value);
    if (/^[=+\-@\t\r]/.test(v)) v = "'" + v;          // stop spreadsheet formula injection
    return '"' + v.replace(/"/g, '""') + '"';
  }

  function exportCsv() {
    if (!state.data) return;
    const rows = getFilteredComments();
    if (!rows.length) { toast('There are no comments to export.', 'warn'); return; }
    const lines = [['#', 'Comment', 'Sentiment', 'Confidence'].map(csvCell).join(',')];
    rows.forEach((c) => lines.push([c.id, c.text, c.sentiment, fmtPct(c.confidence)].map(csvCell).join(',')));
    const blob = new Blob(['﻿' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'youtube-sentiment-' + state.data.mode + '-' + new Date().toISOString().slice(0, 10) + '.csv';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    toast('Exported ' + fmtInt(rows.length) + ' comments to CSV.', 'success');
  }

  /* ------------------------------------------------------------------ *
   * "Analyze a Comment"
   * ------------------------------------------------------------------ */
  function renderCommentResult(r) {
    const box = $('commentResult');
    clear(box);
    box.style.borderStyle = 'solid';
    const c = CLS[r.sentiment];

    box.appendChild(el('div', { className: 'result-head' }, [
      el('div', {}, [
        el('div', { className: 'result-label', text: 'Sentiment' }),
        el('div', { className: 'result-sentiment s-' + c, text: r.sentiment }),
      ]),
      el('div', {}, [
        el('div', { className: 'result-label', text: 'Confidence' }),
        el('div', { className: 'result-conf', text: fmtPct(r.confidence) }),
      ]),
    ]));

    const probs = el('div', { className: 'prob-list' });
    SENTIMENTS.forEach((s) => {
      const p = (r.probabilities && r.probabilities[s]) || 0;
      probs.appendChild(el('div', { className: 'prob-row' }, [
        el('span', { text: s }),
        el('div', { className: 'percent-track' }, [el('div', { className: 'percent-fill fill-' + CLS[s], style: { width: (p * 100).toFixed(1) + '%' } })]),
        el('b', { text: fmtPct(p) }),
      ]));
    });
    box.appendChild(probs);

    box.appendChild(el('p', { className: 'explain', text: r.explanation }));
    if (r.key_terms && r.key_terms.length) {
      box.appendChild(el('div', { className: 'term-chips' },
        r.key_terms.map((term) => el('span', { className: 'term', text: term }))));
    }
  }

  function renderCommentMessage(title, text) {
    const box = $('commentResult');
    clear(box);
    box.style.borderStyle = 'dashed';
    box.appendChild(el('div', { className: 'result-empty' }, [
      el('strong', { text: title }),
      el('span', { className: 'muted', text: text }),
    ]));
  }

  async function submitComment(event) {
    event.preventDefault();
    const text = $('commentInput').value.trim();
    if (!text) {
      renderCommentMessage('Please type a comment first', 'Write or paste any text, then press Analyze Sentiment.');
      toast('The comment is empty.', 'warn');
      $('commentInput').focus();
      return;
    }
    const button = $('commentBtn');
    button.disabled = true;
    button.textContent = 'Analyzing…';
    try {
      const result = await api(API.comment, { method: 'POST', body: { text: text } });
      renderCommentResult(result);
    } catch (e) {
      renderCommentMessage('Could not analyze this comment', e.message);
      toast(e.message, 'error', 6000);
    } finally {
      button.disabled = false;
      button.textContent = 'Analyze Sentiment';
    }
  }

  /* ------------------------------------------------------------------ *
   * Model information
   * ------------------------------------------------------------------ */
  function tile(label, value) {
    return el('div', { className: 'tile' }, [el('span', { text: label }), el('strong', { text: value })]);
  }

  function renderModelInfo(info) {
    const box = $('modelInfo');
    clear(box);
    const pct = (v) => (v === null || v === undefined ? 'n/a' : (v * 100).toFixed(1) + '%');

    box.appendChild(el('div', { className: 'tiles' }, [
      tile('Model', info.model),
      tile('Vectorizer', info.vectorizer),
      tile('Classes', (info.classes || []).join(' / ')),
      tile('Training samples', fmtInt(info.train_samples)),
      tile('Testing samples', fmtInt(info.test_samples)),
      tile('Accuracy', pct(info.accuracy)),
      tile('Precision (macro)', pct(info.precision)),
      tile('Recall (macro)', pct(info.recall)),
      tile('F1 Score (macro)', pct(info.f1_score)),
      tile('Vocabulary size', fmtInt(info.vocabulary_size)),
    ]));

    // per-class table
    const perClass = info.per_class || {};
    const tbody = el('tbody');
    (info.classes || []).forEach((name) => {
      const m = perClass[name];
      if (!m) return;
      tbody.appendChild(el('tr', {}, [
        el('td', { text: name }), el('td', { text: pct(m.precision) }), el('td', { text: pct(m.recall) }),
        el('td', { text: pct(m.f1_score) }), el('td', { text: String(m.support) }),
      ]));
    });
    const table = el('table', { className: 'metric-table' }, [
      el('thead', {}, [el('tr', {}, ['Class', 'Precision', 'Recall', 'F1', 'Support'].map((h) => el('th', { text: h, attrs: { scope: 'col' } })))]),
      tbody,
    ]);
    const left = el('div', {}, [el('h3', { text: 'Per-class results (test set)' }), el('div', { className: 'table-wrap' }, [table])]);

    // confusion matrix
    const cm = info.confusion_matrix || { labels: [], matrix: [] };
    const maxValue = Math.max.apply(null, [1].concat.apply([], cm.matrix));
    const grid = el('div', { className: 'cm', attrs: { role: 'table', 'aria-label': 'Confusion matrix: rows are actual labels, columns are predicted labels' } });
    grid.appendChild(el('div', { className: 'cm-head', text: '' }));
    cm.labels.forEach((l) => grid.appendChild(el('div', { className: 'cm-head', text: 'Pred. ' + l })));
    cm.matrix.forEach((row, i) => {
      grid.appendChild(el('div', { className: 'cm-row', text: 'Actual ' + cm.labels[i] }));
      row.forEach((value, j) => {
        const cell = el('div', { className: 'cm-cell' + (i === j ? ' cm-diag' : ''), text: String(value) });
        cell.style.setProperty('--v', (value / maxValue).toFixed(2));
        grid.appendChild(cell);
      });
    });
    const right = el('div', {}, [
      el('h3', { text: 'Confusion matrix' }), grid,
      el('p', { className: 'muted small', text: 'Rows = actual label, columns = predicted label. The diagonal shows correct predictions.' }),
    ]);

    box.appendChild(el('div', { className: 'model-grid' }, [left, right]));

    const cv = info.cross_validation;
    const cvText = cv ? ' 5-fold cross-validation accuracy: ' + pct(cv.accuracy_mean) + ' ± ' + pct(cv.accuracy_std) + '.' : '';
    box.appendChild(el('p', {
      className: 'model-note',
      text: 'These metrics were calculated on ' + fmtInt(info.test_samples) + ' test comments that the model never saw during training.' +
        cvText + ' The training set is small and hand-labelled, so treat the numbers as a demonstration, not as production accuracy.',
    }));
  }

  async function loadModelInfo() {
    const box = $('modelInfo');
    try {
      renderModelInfo(await api(API.modelInfo));
    } catch (e) {
      clear(box);
      box.appendChild(el('p', { className: 'muted', text: 'Model information is not available: ' + e.message }));
      const retry = el('button', { className: 'btn btn-ghost btn-sm retry', text: 'Try again', attrs: { type: 'button' } });
      retry.addEventListener('click', () => { box.textContent = 'Loading model details…'; loadModelInfo(); });
      box.appendChild(retry);
    }
  }

  /* ------------------------------------------------------------------ *
   * Event wiring
   * ------------------------------------------------------------------ */
  function init() {
    initTheme();
    initNav();

    $('analyzeForm').addEventListener('submit', (e) => { e.preventDefault(); runAnalysis('url'); });
    $('demoBtn').addEventListener('click', () => runAnalysis('demo'));
    $('emptyDemoBtn').addEventListener('click', () => runAnalysis('demo'));
    $('exampleBtn').addEventListener('click', () => {
      $('urlInput').value = EXAMPLE_URL;
      $('urlInput').focus();
    });

    // table controls
    const onSearch = debounce(() => {
      state.query = $('searchInput').value;
      state.page = 1;
      renderTable();
    }, 150);
    $('searchInput').addEventListener('input', onSearch);
    $('sortSelect').addEventListener('change', (e) => { state.sort = e.target.value; state.page = 1; renderTable(); });
    $('pageSize').addEventListener('change', (e) => { state.pageSize = Number(e.target.value); state.page = 1; renderTable(); });
    document.querySelectorAll('#filterChips .chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        state.filter = chip.getAttribute('data-filter');
        state.page = 1;
        syncFilterChips();
        renderTable();
      });
    });
    $('exportBtn').addEventListener('click', exportCsv);

    // word-frequency tabs
    document.querySelectorAll('.tab').forEach((tab) => {
      tab.addEventListener('click', () => {
        state.wordSource = tab.getAttribute('data-words');
        syncWordTabs();
        renderWordChart(true);
      });
    });

    // single comment
    $('commentForm').addEventListener('submit', submitComment);
    $('commentInput').addEventListener('input', () => {
      $('charCount').textContent = $('commentInput').value.length + ' / ' + $('commentInput').maxLength;
    });
    document.querySelectorAll('[data-example]').forEach((chip) => {
      chip.addEventListener('click', () => {
        const input = $('commentInput');
        input.value = chip.getAttribute('data-example');
        input.dispatchEvent(new Event('input'));
        input.focus();
      });
    });

    loadStatus();
    loadModelInfo();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
