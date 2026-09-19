import { useEffect, useRef, useState } from 'react';
import { platforms, type PlatformId } from './data';
import { tokens, selectTokens, gateDecision, frameAt, phases, CHAPTER_SECONDS, DURATION, type Trigger } from './model';

const chapters = [
  { label: 'The bottleneck', title: 'Good plans can still arrive too late.', description: 'Every new observation adds context. Repeated planner calls turn that growing context into waiting time.', point: 'Task success alone misses the cost of replanning.', location: 'Observe → trigger → planner', evidence: 'Paper · AirSim K=8 · Table 1' },
  { label: 'When to call', title: 'A trigger is a request, not a planner call.', description: 'BRACE checks stability before spending computation. Keep executing when a routine replan can wait; admit recovery when it matters.', point: 'Control when to replan before optimizing the call.', location: 'Trigger → BRACE gate → execute / replan', evidence: 'Illustrative controller states' },
  { label: 'How much to spend', title: 'Keep what matters within a budget.', description: 'BRACE assigns the budget. E-RECAP is the compression module: protect task anchors and recent state, then retain useful middle context.', point: 'The budget comes from the controller; compression makes it usable.', location: 'BRACE budget → E-RECAP → planner input', evidence: 'Illustrative tokens and utility scores' },
  { label: 'Count the whole cost', title: 'The deadline covers the entire call.', description: 'Compression, retrieval, planning and update all consume time. BRACE logs each phase so the complete call can be checked against its deadline.', point: 'Account for module overhead as well as planner latency.', location: 'Compress → retrieve → planner → update → audit', evidence: 'Illustrative phase accounting' },
  { label: 'What changes', title: 'Fewer deadline misses. The same task success.', description: 'Across the main platform settings, BRACE + E-RECAP reduces context and deadline violations. Select a platform to inspect the reported comparison.', point: 'Measure real-time behavior alongside task success.', location: 'Closed-loop evaluation', evidence: 'Paper · Main platform results · Table 1' },
];

function initialTheme() {
  try { return localStorage.getItem('brace-theme') === 'dark' ? 'dark' : 'light'; }
  catch { return 'light'; }
}

export default function App() {
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [theme, setTheme] = useState(initialTheme);
  const [trigger, setTrigger] = useState<Trigger>('routine');
  const [platformId, setPlatformId] = useState<PlatformId>('airsim');
  const [selected, setSelected] = useState<string | null>(null);
  const [details, setDetails] = useState(false);
  const shell = useRef<HTMLElement>(null);
  const chapterJump = useRef(false);
  const { chapter, progress } = frameAt(time);
  const scene = chapters[chapter];
  const embedded = new URLSearchParams(location.search).has('embed');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const reveal = reducedMotion ? 1 : Math.min(1, 0.18 + progress * 1.65);
  const tokenBudget = reveal < .4 ? 24 : reveal < .75 ? 18 : 12;
  const kept = selectTokens(tokens, tokenBudget);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let id: number;
    const tick = (now: number) => {
      const elapsed = Math.min(.15, (now - last) / 1000);
      last = now;
      setTime(t => Math.min(DURATION, t + elapsed));
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing]);

  useEffect(() => { if (time >= DURATION) setPlaying(false); }, [time]);
  useEffect(() => {
    const pause = () => { if (document.hidden) setPlaying(false); };
    document.addEventListener('visibilitychange', pause);
    return () => document.removeEventListener('visibilitychange', pause);
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('brace-theme', theme); } catch { /* Storage is optional. */ }
  }, [theme]);
  useEffect(() => {
    const sync = (event: StorageEvent) => { if (event.key === 'brace-theme') setTheme(event.newValue === 'dark' ? 'dark' : 'light'); };
    window.addEventListener('storage', sync);
    return () => window.removeEventListener('storage', sync);
  }, []);
  useEffect(() => {
    if (!embedded || !shell.current) return;
    const observer = new ResizeObserver(() => {
      parent.postMessage({ type: 'brace-animation-height', height: Math.ceil(shell.current!.getBoundingClientRect().height + 28) }, location.origin);
    });
    observer.observe(shell.current);
    return () => observer.disconnect();
  }, [embedded]);
  useEffect(() => {
    if (!embedded) return;
    const receive = (event: MessageEvent) => {
      if (event.origin === location.origin && event.source === parent && event.data?.type === 'brace-animation-pause') setPlaying(false);
    };
    window.addEventListener('message', receive);
    return () => window.removeEventListener('message', receive);
  }, [embedded]);

  function goTo(index: number) {
    chapterJump.current = true;
    // Chapter jumps show the settled explanation; Play replays its transformation.
    setTime(Math.max(0, Math.min(4, index)) * CHAPTER_SECONDS + 9);
    setPlaying(false);
    setDetails(false);
    setSelected(null);
  }
  function togglePlay() {
    if (playing) { setPlaying(false); return; }
    if (time >= DURATION) setTime(0);
    else if (chapterJump.current) setTime(chapter * CHAPTER_SECONDS);
    chapterJump.current = false;
    setDetails(false);
    setPlaying(true);
  }
  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).closest('button, a, input, summary, select, textarea')) return;
      if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
      if (event.key === 'ArrowRight') { event.preventDefault(); goTo(chapter + 1); }
      if (event.key === 'ArrowLeft') { event.preventDefault(); goTo(chapter - 1); }
    };
    window.addEventListener('keydown', keyboard);
    return () => window.removeEventListener('keydown', keyboard);
  }, [playing, time, chapter]);

  return <main ref={shell} className={`app-shell ${embedded ? 'embedded' : ''}`}>
    <header className="topbar">
      <a className="brand" href="../" target={embedded ? '_top' : undefined}><span className="brand-mark">B</span><strong>BRACE</strong><span className="brand-note">Interactive animation</span></a>
      <div className="top-actions"><a href="../" target={embedded ? '_top' : undefined}>Project ↗</a><button className="theme-switch" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`}>{theme === 'light' ? '◐' : '◑'}</button></div>
    </header>
    <nav className="chapters" aria-label="Animation chapters">
      {chapters.map((item, i) => <button key={item.label} className={i === chapter ? 'active' : ''} aria-current={i === chapter ? 'step' : undefined} onClick={() => goTo(i)}><span>0{i + 1}</span>{item.label}</button>)}
    </nav>
    <section className="workbench" aria-label="BRACE walkthrough">
      <div className="scene-heading" aria-live="polite">
        <p className="eyebrow">{String(chapter + 1).padStart(2, '0')} / {String(chapters.length).padStart(2, '0')} <span>{scene.location}</span></p>
        <h1>{scene.title}</h1><p className="description">{scene.description}</p>
      </div>
      <div className="stage" data-chapter={chapter}>
        {chapter === 0 && <Problem reveal={reveal} />}
        {chapter === 1 && <Gate reveal={reveal} trigger={trigger} choose={value => { setTrigger(value); setPlaying(false); }} />}
        {chapter === 2 && <Compression keptIds={kept.map(t => t.id)} selected={selected} select={id => { setSelected(id); setPlaying(false); }} />}
        {chapter === 3 && <Audit reveal={reveal} />}
        {chapter === 4 && <Evidence platformId={platformId} choose={id => { setPlatformId(id); setPlaying(false); }} />}
      </div>
      <div className="takeaway"><span>THE POINT</span><strong>{scene.point}</strong></div>
      <footer className="playback" aria-label="Playback controls">
        <button className="play primary" onClick={togglePlay}>{playing ? 'Ⅱ Pause' : time >= DURATION ? '↻ Replay' : '▶ Play tour'}</button>
        <button className="icon-button" onClick={() => { setTime(0); setPlaying(true); setDetails(false); }} aria-label="Replay from beginning">↺</button>
        <label className="timeline"><span className="sr-only">Animation timeline</span><input type="range" min="0" max={DURATION} step=".1" value={time} aria-valuetext={`${scene.label}, ${Math.round(time)} of ${DURATION} seconds`} onChange={e => { chapterJump.current = false; setTime(Number(e.target.value)); setPlaying(false); setDetails(false); }} /><span>{Math.floor(time).toString().padStart(2, '0')}s / {DURATION}s</span></label>
        <button className="step-button" disabled={chapter === 0} onClick={() => goTo(chapter - 1)} aria-label="Previous chapter">←</button>
        <button className="step-button" disabled={chapter === 4} onClick={() => goTo(chapter + 1)} aria-label="Next chapter">Next →</button>
      </footer>
    </section>
    <div className="source-row"><span>{scene.evidence}</span><button onClick={() => { setDetails(!details); setPlaying(false); }} aria-expanded={details} aria-controls="method-details">{details ? 'Hide details −' : 'Method & sources +'}</button></div>
    {details && <section id="method-details" className="details">
      <h2>One controller, a modular call path</h2>
      <p>BRACE determines when to replan and assigns token / time budgets. E-RECAP is one pluggable efficiency module; retrieval and caching can also occupy the call path. Audit records their costs.</p>
      <p>The 32 tokens, utility scores, gate inputs and phase durations are deterministic teaching examples. Platform metrics are the paper’s main-platform results (Table 1), reproduced without changing the reported outcomes.</p>
      <div className="detail-links"><a href="../static/main.pdf" target="_blank" rel="noreferrer">Read the paper ↗</a><a href="https://github.com/NEBULIS-Lab/BRACE/blob/main/docs/CONTROLLER.md" target="_blank" rel="noreferrer">Controller logic ↗</a><a href="https://github.com/NEBULIS-Lab/BRACE/tree/main/animation" target="_blank" rel="noreferrer">Animation source ↗</a></div>
    </section>}
  </main>;
}

function Problem({ reveal }: { reveal: number }) {
  const baseline = platforms.find(p => p.id === 'airsim')!;
  return <div className="problem-scene">
    <div className="context-flow"><div className="context-object"><span className="object-label">Live context</span><div className="mini-tokens">{tokens.slice(0, 24).map((t, i) => <i key={t.id} style={{ opacity: i < 8 + reveal * 16 ? 1 : .16 }} />)}</div><small>Observations + history + messages</small></div><span className="flow-arrow">→</span><div className="planner-object"><span className="object-label">Repeated replanning</span><strong>Planner</strong><div className="request-queue"><span>Request 1</span><span style={{ opacity: .3 + reveal * .7 }}>2</span><span style={{ opacity: .15 + reveal * .85 }}>3</span></div></div><span className="flow-arrow">→</span><div className="result-object"><span className="object-label">Task success</span><strong>100<span>%</span></strong><small>But is the plan on time?</small></div></div>
    <div className="latency-chart"><div className="chart-label"><span>AirSim · No BRACE · P95 call latency</span><strong>8.52 s</strong></div><div className="deadline-track"><div className="baseline-fill" style={{ width: `${94.67 * reveal}%` }} /><span className="deadline-marker" style={{ left: '27.78%' }}><b>2.50 s deadline</b></span></div><div className="axis"><span>0 s</span><span>9 s</span></div></div>
    <p className="scene-note"><span className="bad-dot" />100% deadline violations in this baseline setting.</p>
  </div>;
}

function Gate({ trigger, choose, reveal }: { trigger: Trigger; choose: (value: Trigger) => void; reveal: number }) {
  const decision = gateDecision(trigger);
  return <div className="gate-scene">
    <div className="choice-row" role="group" aria-label="Trigger scenario">{([['routine', 'Routine trigger'], ['unsafe', 'Unsafe state'], ['ready', 'Windows cleared']] as const).map(([id, label]) => <button key={id} className={trigger === id ? 'selected' : ''} aria-pressed={trigger === id} onClick={() => choose(id)}>{label}</button>)}</div>
    <div className="gate-flow"><div className="gate-input"><span className="object-label">Same live context</span><div className="mini-tokens">{tokens.slice(0, 24).map(t => <i key={t.id} />)}</div><p>{trigger === 'unsafe' ? 'Unsafe state detected' : trigger === 'ready' ? 'Ready for a new plan' : 'Another periodic request'}</p></div><div className="gate-controller"><span className="object-label">BRACE gate</span><div><span>Cooldown</span><b>{trigger === 'ready' ? 'clear' : 'active'}</b></div><div><span>Commit window</span><b>{trigger === 'ready' ? 'clear' : 'active'}</b></div><div className={trigger === 'unsafe' ? 'override' : ''}><span>Unsafe override</span><b>{trigger === 'unsafe' ? 'yes' : 'no'}</b></div></div><div className={`gate-output ${decision.calls ? 'admitted' : ''}`}><span className="object-label">Decision</span><strong>{decision.label}</strong><div className="call-count"><b>{decision.calls}</b><span>new planner call{decision.calls ? '' : 's'}</span></div></div></div>
    <p className="decision-reason" style={{ opacity: Math.max(.65, reveal) }}>{decision.reason}</p>
  </div>;
}

function Compression({ keptIds, selected, select }: { keptIds: string[]; selected: string | null; select: (id: string) => void }) {
  const kept = new Set(keptIds);
  const selectedToken = tokens.find(t => t.id === selected);
  return <div className="compression-scene">
    <div className="compression-header"><span>One context · stable token identities</span><strong>32 <span>→</span> {kept.size} <small>tokens</small></strong></div>
    <div className="token-grid" aria-label="Context tokens">{tokens.map((t, i) => <button key={t.id} data-token-id={t.id} aria-label={`Token ${i + 1}, ${t.protectedSlot ? `protected ${t.protectedSlot}` : `utility ${t.utility.toFixed(2)}`}, ${kept.has(t.id) ? 'kept' : 'pruned'}`} aria-pressed={selected === t.id} className={`token ${t.protectedSlot ? 'protected' : ''} ${kept.has(t.id) ? 'kept' : 'pruned'} ${selected === t.id ? 'inspected' : ''}`} onClick={() => select(t.id)}><span>{String(i + 1).padStart(2, '0')}</span><i style={{ height: `${t.utility * 65}%` }} /></button>)}</div>
    <div className="token-legend"><span><i className="protected-key" />Protected task / latest state</span><span><i className="kept-key" />Useful middle context</span><span><i className="pruned-key" />Pruned</span></div>
    <div className="packed-row"><span>Planner input →</span><div className="packed-tokens">{tokens.filter(t => kept.has(t.id)).map(t => <button key={t.id} className={`${t.protectedSlot ? 'protected' : ''} ${selected === t.id ? 'inspected' : ''}`} aria-label={`Inspect retained token ${tokens.indexOf(t) + 1}`} onClick={() => select(t.id)}>{tokens.indexOf(t) + 1}</button>)}</div></div>
    <p className="token-inspector" aria-live="polite">{selectedToken ? <>Token {tokens.indexOf(selectedToken) + 1} · {selectedToken.protectedSlot ? `protected ${selectedToken.protectedSlot === 'head' ? 'task anchor' : 'latest state'}` : `utility ${selectedToken.utility.toFixed(2)}`} · {kept.has(selectedToken.id) ? 'retained in its original order' : 'removed to meet the budget'}</> : 'Select a token to trace it into the planner input.'}</p>
  </div>;
}

function Audit({ reveal }: { reveal: number }) {
  const total = phases.reduce((sum, phase) => sum + phase.ms, 0);
  return <div className="audit-scene">
    <div className="audit-head"><div><span className="object-label">Total call cost</span><strong>{(total / 1000).toFixed(2)} <small>s</small></strong></div><span className="equation">= compression + retrieval + planner + update</span><div><span className="object-label">Deadline</span><strong>2.50 <small>s</small></strong></div></div>
    <div className="audit-track">{phases.map((phase, i) => <div key={phase.name} className={`phase phase-${i}`} style={{ width: `${phase.ms / 2500 * 100 * reveal}%` }} />)}<span className="headroom" style={{ left: `${total / 2500 * 100}%` }}>{((2500 - total) / 1000).toFixed(2)} s remaining</span></div>
    <div className="phase-labels">{phases.map((phase, i) => <div key={phase.name}><i className={`phase-${i}`} /><span>{phase.name}</span><strong>{phase.ms.toLocaleString()} ms</strong></div>)}</div>
    <p className="audit-result">Every enabled module is included in the same latency ledger.</p>
  </div>;
}

function Evidence({ platformId, choose }: { platformId: PlatformId; choose: (id: PlatformId) => void }) {
  const platform = platforms.find(p => p.id === platformId)!;
  const base = platform.baseline;
  const brace = platform.braceErecap;
  return <div className="evidence-scene">
    <div className="choice-row" role="group" aria-label="Evaluation platform">{platforms.map(p => <button key={p.id} className={p.id === platformId ? 'selected' : ''} aria-pressed={p.id === platformId} onClick={() => choose(p.id)}>{p.label.replace('Microsoft ', '').replace('Meta ', '')}</button>)}</div>
    <div className="evidence-context"><span>{platform.scenario}</span><span>Deadline {platform.sloMs.toLocaleString()} ms</span></div>
    <div className="violation-comparison"><div className="comparison-row"><span>No BRACE</span><div><i className="base-bar" style={{ width: `${base.sloViolationPct}%` }} /></div><strong>{base.sloViolationPct}%</strong></div><div className="comparison-row"><span>BRACE + E-RECAP</span><div><i className="brace-bar" style={{ width: `${brace.sloViolationPct}%` }} /></div><strong>{brace.sloViolationPct}%</strong></div><p>Deadline violations · lower is better</p></div>
    <div className="evidence-metrics"><div><span>Tokens / call</span><strong>{base.tokens.toLocaleString()} <i>→</i> {brace.tokens.toLocaleString()}</strong></div><div><span>P95 latency</span><strong>{(base.latencyP95Ms / 1000).toFixed(2)} <i>→</i> {(brace.latencyP95Ms / 1000).toFixed(2)} s</strong></div><div><span>Task success</span><strong>{base.successPct}% <i>→</i> {brace.successPct}%</strong></div></div>
  </div>;
}
