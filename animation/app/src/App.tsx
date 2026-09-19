import { useMemo, useState, type CSSProperties } from "react";
import { makeTokens, platforms, type PlatformId, type TokenSegment } from "./data";

type FocusArea = "context" | "pressure" | "gate" | "budget" | "compression" | "audit" | "evidence";
type LoopNode = "observe" | "trigger" | "gate" | "budget" | "compress" | "retrieve" | "planner" | "audit";
type TriggerType = "periodic" | "failure" | "hazard";
type GateState = "admit" | "cooldown" | "commit" | "override";
type PhaseName = "compression" | "retrieval" | "planner" | "update";
type LaneId = "task" | "history" | "messages" | "observations" | "latest";

type GateChecks = {
  trigger: boolean;
  cooldown: boolean;
  commit: boolean;
  override: boolean;
};

type Counters = {
  delta: number;
  omega: number;
  sinceReplan: number;
  sincePlanChange: number;
  failureWindow: number;
};

type GuidedStep = {
  id: string;
  label: string;
  title: string;
  narrative: string;
  takeaway: string;
  controlQuestion: string;
  controlAction: string;
  moduleRole: string;
  focus: FocusArea;
  node: LoopNode;
  platformId: PlatformId;
  triggerType: TriggerType;
  gate: GateState;
  checks: GateChecks;
  counters: Counters;
  pressure: number;
  rawTokens: number;
  budgetTokens: number | null;
  keepTarget: number;
  layerCounts: number[];
  phaseMs: Record<PhaseName, number>;
  auditLine: string;
  evidenceNote: string;
};

const loopNodes: { id: LoopNode; label: string; detail: string }[] = [
  { id: "observe", label: "Observe", detail: "C_t" },
  { id: "trigger", label: "Trigger", detail: "tau_t" },
  { id: "gate", label: "Gate", detail: "u_t" },
  { id: "budget", label: "Budget", detail: "B_t/SLO_t" },
  { id: "compress", label: "E-RECAP", detail: "pi_i" },
  { id: "retrieve", label: "Retrieve", detail: "reuse" },
  { id: "planner", label: "Planner", detail: "call" },
  { id: "audit", label: "Audit", detail: "l_t" },
];

const laneMeta: Record<LaneId, { label: string; short: string; description: string }> = {
  task: { label: "Task spec", short: "Head", description: "protected task anchors" },
  history: { label: "Plan history", short: "Hist", description: "previous plans and failures" },
  messages: { label: "Agent messages", short: "Msg", description: "coordination summaries" },
  observations: { label: "Observations", short: "Obs", description: "state and feedback" },
  latest: { label: "Latest state", short: "Tail", description: "protected recent context" },
};

const contributionFrame = [
  {
    label: "Reframe",
    value: "Planning ability -> control problem",
    detail: "Control when to call, how to call, and what cost the loop can afford.",
  },
  {
    label: "Call path",
    value: "Gate + budget + efficiency slot",
    detail: "Instrument the replanning call instead of treating it as a black-box planner.",
  },
  {
    label: "E-RECAP",
    value: "One pluggable efficiency module",
    detail: "Compression shows inserted modules can change latency and closed-loop behavior.",
  },
];

const callPathSlots = [
  { id: "gate", label: "Gate", detail: "admit, suppress, or override triggers" },
  { id: "budget", label: "Budget", detail: "assign token and latency budget" },
  { id: "compression", label: "Compress", detail: "E-RECAP prunes context before the planner" },
  { id: "retrieval", label: "Retrieve / cache", detail: "slot for reusable memory or cached context" },
  { id: "audit", label: "Audit", detail: "attribute latency and update the loop" },
];

const stepStates: GuidedStep[] = [
  {
    id: "context",
    label: "Context",
    title: "The loop starts by collecting a live replanning context.",
    narrative:
      "Task anchors, plan history, messages, observations, and the newest state all enter the same replanning buffer before the planner can act.",
    takeaway: "The page should make C_t feel like a growing runtime object, not a static prompt.",
    controlQuestion: "What state has accumulated enough to justify a replanning call?",
    controlAction: "Observe C_t as a system variable before spending a planner call.",
    moduleRole: "Compression and retrieval are idle until the controller decides the context should enter the call path.",
    focus: "context",
    node: "observe",
    platformId: "habitat",
    triggerType: "periodic",
    gate: "admit",
    checks: { trigger: true, cooldown: true, commit: true, override: false },
    counters: { delta: 3, omega: 2, sinceReplan: 6, sincePlanChange: 5, failureWindow: 0 },
    pressure: 0.42,
    rawTokens: 235,
    budgetTokens: null,
    keepTarget: 52,
    layerCounts: [52, 52, 52, 52],
    phaseMs: { compression: 0, retrieval: 0, planner: 2677, update: 0 },
    auditLine: "raw No-BRACE call path exceeds the Habitat SLO tail even with saturated success",
    evidenceNote: "Meta Habitat anchor: success is saturated, but tail latency is the hidden issue.",
  },
  {
    id: "pressure",
    label: "Pressure",
    title: "Coordination pressure turns context growth into repeated triggers.",
    narrative:
      "Hazard, failure, and periodic triggers compete for the same call path. The loop shows why repeated replanning can overload deadlines even when plans eventually work.",
    takeaway: "Trigger frequency is part of the bottleneck; a fast planner is not enough if the loop churns.",
    controlQuestion: "When should a fresh observation become an expensive replanning request?",
    controlAction: "Separate trigger detection from planner invocation so the loop can regulate call frequency.",
    moduleRole: "Efficiency modules matter only after trigger pressure is converted into an admitted call.",
    focus: "pressure",
    node: "trigger",
    platformId: "airsim",
    triggerType: "hazard",
    gate: "admit",
    checks: { trigger: true, cooldown: true, commit: true, override: false },
    counters: { delta: 3, omega: 2, sinceReplan: 4, sincePlanChange: 4, failureWindow: 1 },
    pressure: 0.86,
    rawTokens: 2934,
    budgetTokens: null,
    keepTarget: 52,
    layerCounts: [52, 52, 52, 52],
    phaseMs: { compression: 0, retrieval: 0, planner: 8520, update: 0 },
    auditLine: "hazard trigger fired; unbudgeted AirSim tail latency reaches the deadline-miss regime",
    evidenceNote: "AirSim K=8 baseline reaches 100% SLO violation despite 100% success.",
  },
  {
    id: "gate",
    label: "Gate",
    title: "The stability gate admits useful replans and suppresses churn.",
    narrative:
      "Cooldown and commit counters decide whether a trigger should become a planner call. Failure-aware override remains available when recovery is needed.",
    takeaway: "The controller is the anti-churn mechanism: it controls when computation is worth spending.",
    controlQuestion: "Should this trigger actually replace the current plan?",
    controlAction: "Use cooldown, commit stability, and failure override to suppress unstable call churn.",
    moduleRole: "The call path is protected before any compression, retrieval, or planner work is paid for.",
    focus: "gate",
    node: "gate",
    platformId: "robofactory",
    triggerType: "failure",
    gate: "commit",
    checks: { trigger: true, cooldown: true, commit: false, override: false },
    counters: { delta: 3, omega: 3, sinceReplan: 5, sincePlanChange: 1, failureWindow: 1 },
    pressure: 0.72,
    rawTokens: 1566,
    budgetTokens: null,
    keepTarget: 52,
    layerCounts: [52, 52, 52, 52],
    phaseMs: { compression: 0, retrieval: 0, planner: 0, update: 0 },
    auditLine: "commit window holds the current plan; no new planner call is spent",
    evidenceNote: "Proxy sweep: controller structure collapses calls/deadlocks compared with replanning every step.",
  },
  {
    id: "budget",
    label: "Budget",
    title: "An admitted trigger receives a token and latency budget.",
    narrative:
      "Budget selection connects C_t, SLO_t, and the enabled modules before retrieval, compression, planning, update, and audit run.",
    takeaway: "B_t is a controller output, not a manual keep-ratio knob.",
    controlQuestion: "How much context and latency can this admitted call spend?",
    controlAction: "Choose a budget for the whole replanning path rather than tuning a viewer-facing slider.",
    moduleRole: "Compression, retrieval, and caching can be swapped into the budgeted call path as efficiency modules.",
    focus: "budget",
    node: "budget",
    platformId: "habitat",
    triggerType: "periodic",
    gate: "admit",
    checks: { trigger: true, cooldown: true, commit: true, override: false },
    counters: { delta: 3, omega: 2, sinceReplan: 3, sincePlanChange: 5, failureWindow: 0 },
    pressure: 0.62,
    rawTokens: 235,
    budgetTokens: 20,
    keepTarget: 44,
    layerCounts: [52, 47, 44, 44],
    phaseMs: { compression: 35.63, retrieval: 0, planner: 2414.05, update: 0.24 },
    auditLine: "budget selected; Habitat call-path mean stays near the 2500 ms SLO after pruning",
    evidenceNote: "Matched-budget baselines show token count alone does not explain tail behavior.",
  },
  {
    id: "compression",
    label: "E-RECAP",
    title: "E-RECAP scores tokens, protects anchors, and prunes progressively.",
    narrative:
      "The utility predictor scores middle tokens from hidden states. Head and tail windows stay pinned while low-utility middle tokens are removed across layers.",
    takeaway: "This step must show actual token processing: score, select, prune, and pass kept tokens forward.",
    controlQuestion: "How should the admitted context be made planner-ready under the budget?",
    controlAction: "Run E-RECAP as a call-path module that protects anchors and prunes low-utility middle context.",
    moduleRole: "The contribution is the modular efficiency slot; E-RECAP is the concrete compression instance.",
    focus: "compression",
    node: "compress",
    platformId: "airsim",
    triggerType: "hazard",
    gate: "admit",
    checks: { trigger: true, cooldown: true, commit: true, override: false },
    counters: { delta: 3, omega: 2, sinceReplan: 5, sincePlanChange: 5, failureWindow: 1 },
    pressure: 0.7,
    rawTokens: 2934,
    budgetTokens: 1114,
    keepTarget: 27,
    layerCounts: [52, 44, 34, 27],
    phaseMs: { compression: 36, retrieval: 42, planner: 1536, update: 26 },
    auditLine: "head/tail pinned; top-utility middle tokens form the AirSim budgeted planner input",
    evidenceNote: "E-RECAP removes 71-76% of tokens in multi-agent Habitat context-growth tests.",
  },
  {
    id: "audit",
    label: "Audit",
    title: "The call returns through planner, execution, and phase accounting.",
    narrative:
      "The audit path records compression, retrieval, planner, and update time separately before checking l_t against SLO_t.",
    takeaway: "BRACE makes overhead attributable instead of hiding every cost inside one latency number.",
    controlQuestion: "Did the full replanning call stay within the real-time contract?",
    controlAction: "Audit each phase so future gating and budget decisions are informed by measured cost.",
    moduleRole: "Any inserted compression, retrieval, or cache module must be accounted for in l_t.",
    focus: "audit",
    node: "audit",
    platformId: "habitat",
    triggerType: "failure",
    gate: "override",
    checks: { trigger: true, cooldown: false, commit: false, override: true },
    counters: { delta: 3, omega: 2, sinceReplan: 1, sincePlanChange: 1, failureWindow: 3 },
    pressure: 0.58,
    rawTokens: 235,
    budgetTokens: 20,
    keepTarget: 22,
    layerCounts: [52, 38, 28, 22],
    phaseMs: { compression: 35.63, retrieval: 0, planner: 2414.05, update: 0.24 },
    auditLine: "failure override admits recovery; appendix phase audit reports 2486.31 ms P95",
    evidenceNote: "Habitat phase table: pruning overhead is small relative to planner time after compression.",
  },
  {
    id: "evidence",
    label: "Evidence",
    title: "The same loop explains navigation, manipulation, traffic, and robot results.",
    narrative:
      "The mechanism remains visible while evidence expands: token reduction maps to E-RECAP, deadline reduction maps to budgeting/accounting, and churn reduction maps to the gate.",
    takeaway: "Metrics should read as evidence for the loop, not as unrelated cards below the animation.",
    controlQuestion: "Does call-path control change closed-loop behavior across embodied domains?",
    controlAction: "Connect tokens, latency violations, and stability back to the same controlled replanning loop.",
    moduleRole: "E-RECAP is evidence that the framework can compose efficiency modules to alter real-time replanning.",
    focus: "evidence",
    node: "audit",
    platformId: "airsim",
    triggerType: "hazard",
    gate: "admit",
    checks: { trigger: true, cooldown: true, commit: true, override: false },
    counters: { delta: 3, omega: 2, sinceReplan: 5, sincePlanChange: 4, failureWindow: 0 },
    pressure: 0.64,
    rawTokens: 2934,
    budgetTokens: 1114,
    keepTarget: 26,
    layerCounts: [52, 42, 32, 26],
    phaseMs: { compression: 36, retrieval: 42, planner: 1536, update: 26 },
    auditLine: "cross-platform summary links token, SLO, and stability metrics back to loop modules",
    evidenceNote: "Across main platforms, baseline success can saturate while deadlines fail.",
  },
];

function metricWidth(value: number, max: number) {
  return `${Math.max(4, Math.min(100, (value / max) * 100))}%`;
}

function tokenSelection(tokens: TokenSegment[], keepTarget: number) {
  if (keepTarget >= tokens.length) {
    return new Set(tokens.map((token) => token.id));
  }

  const protectedTokens = tokens.filter((token) => token.protectedSlot);
  const middleTokens = tokens
    .filter((token) => !token.protectedSlot)
    .sort((left, right) => right.utility - left.utility);
  const remaining = Math.max(0, keepTarget - protectedTokens.length);
  return new Set([...protectedTokens, ...middleTokens.slice(0, remaining)].map((token) => token.id));
}

function tokenLane(token: TokenSegment, index: number): LaneId {
  if (token.protectedSlot === "head") return "task";
  if (token.protectedSlot === "tail") return "latest";
  if (token.kind === "message") return "messages";
  if (token.kind === "observation") return "observations";
  return index % 2 === 0 ? "history" : "observations";
}

function tokenClass(token: TokenSegment, keptTokens: Set<string>) {
  if (!keptTokens.has(token.id)) {
    return "token-chip pruned";
  }
  if (token.protectedSlot === "head") {
    return "token-chip head";
  }
  if (token.protectedSlot === "tail") {
    return "token-chip tail";
  }
  return token.utility > 0.68 ? "token-chip middle selected" : "token-chip middle";
}

function utilityBars(token: TokenSegment) {
  return [0, 1, 2, 3].map((index) => {
    const wave = Math.sin((index + 1) * (token.utility + 0.7)) * 0.5 + 0.5;
    return Math.max(18, Math.round((0.35 + wave * 0.65) * 100));
  });
}

function loopIndex(node: LoopNode) {
  return Math.max(
    0,
    loopNodes.findIndex((item) => item.id === node),
  );
}

function totalPhaseMs(step: GuidedStep) {
  return Object.values(step.phaseMs).reduce((sum, value) => sum + value, 0);
}

function gateLabel(gate: GateState) {
  return {
    admit: "Admit",
    cooldown: "Wait",
    commit: "Hold",
    override: "Override",
  }[gate];
}

function budgetLabel(step: GuidedStep) {
  if (step.gate === "commit" || step.gate === "cooldown") {
    return "held";
  }
  return step.budgetTokens === null ? "none" : `${step.budgetTokens.toLocaleString()} tok`;
}

function formatMs(value: number) {
  return Number.isInteger(value) ? `${value}` : value.toFixed(2);
}

function App() {
  const [stepIndex, setStepIndex] = useState(0);
  const step = stepStates[stepIndex];
  const platform = platforms.find((item) => item.id === step.platformId) ?? platforms[0];
  const tokens = useMemo(() => makeTokens(64, stepIndex + 5), [stepIndex]);
  const keptTokens = useMemo(() => tokenSelection(tokens, step.keepTarget), [tokens, step.keepTarget]);
  const isFinalStep = stepIndex === stepStates.length - 1;

  const advance = () => {
    setStepIndex((current) => (current === stepStates.length - 1 ? 0 : current + 1));
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="title-block">
          <div className="title-head">
            <div>
              <p className="eyebrow">Interactive method explainer</p>
              <h1>Budgeted Replanning for Embodied Agents</h1>
            </div>
            <a className="project-home-link" href="https://nebulis-lab.com/BRACE/" rel="noreferrer" target="_blank">
              Project site
            </a>
          </div>
          <div className="title-summary">
            <div>
              <span>When Replanning Becomes the Bottleneck</span>
              <strong>Budgeted controller, context compression, and SLO-aware audit signals.</strong>
            </div>
            <p>
              A loop-based walkthrough of how a replanning controller gates triggers, budgets context,
              compresses tokens with E-RECAP, and audits latency against real-time deadlines.
            </p>
          </div>
        </div>
      </header>

      <section className="guide-bar" aria-label="Guided steps">
        <div className="progress-cluster">
          <span>{`Step ${stepIndex + 1} of ${stepStates.length}`}</span>
          <strong>{step.label}</strong>
          <div className="step-dots">
            {stepStates.map((item, index) => (
              <button
                className={index === stepIndex ? "active" : index < stepIndex ? "visited" : ""}
                key={item.id}
                onClick={() => setStepIndex(index)}
                type="button"
                aria-label={`Go to ${item.label}`}
              />
            ))}
          </div>
        </div>
        <div className="guide-actions">
          <button
            className="secondary"
            disabled={stepIndex === 0}
            onClick={() => setStepIndex((current) => Math.max(0, current - 1))}
            type="button"
          >
            Back
          </button>
          <div className="next-action">
            <span className="next-cue" aria-hidden="true">
              {isFinalStep ? "replay" : "keep going"}
              <i />
            </span>
            <button className="primary next-primary" onClick={advance} type="button">
              {isFinalStep ? "Restart" : "Next"}
            </button>
          </div>
        </div>
      </section>

      <section className="story-layout" aria-label="Budgeted replanning walkthrough">
        <article className={`story-card focus-${step.focus}`} key={`story-${step.id}`}>
          <span className="section-label">{step.label}</span>
          <h2>{step.title}</h2>
          <p>{step.narrative}</p>
          <div className="thesis-strip">
            <span>Central contribution</span>
            <strong>Replanning becomes a control problem: when, how, and at what cost.</strong>
          </div>
          <div className="contribution-grid">
            {contributionFrame.map((item) => (
              <div key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <p>{item.detail}</p>
              </div>
            ))}
          </div>
          <div className="control-question">
            <span>Question in this step</span>
            <p>{step.controlQuestion}</p>
          </div>
          <div className="decision-stack">
            <div>
              <span>Control move</span>
              <strong>{step.controlAction}</strong>
            </div>
            <div>
              <span>Efficiency module role</span>
              <strong>{step.moduleRole}</strong>
            </div>
          </div>
          <div className="takeaway">
            <strong>Takeaway</strong>
            <p>{step.takeaway}</p>
          </div>
          <div className="step-facts">
            <span>{`trigger: ${step.triggerType}`}</span>
            <span>{`gate: ${gateLabel(step.gate)}`}</span>
            <span>{`B_t: ${budgetLabel(step)}`}</span>
          </div>
        </article>

        <LoopWorkbench key={`workbench-${step.id}`} keptTokens={keptTokens} platform={platform} step={step} tokens={tokens} />

        <aside className={`context-card focus-${step.focus}`} key={`scene-${step.id}`}>
          <span className="section-label">Current scene</span>
          <h2>{platform.label}</h2>
          <p>{platform.scenario}</p>
          <SceneState platform={platform} step={step} />
          <MetricRow
            label="P95 latency"
            base={platform.baseline.latencyP95Ms}
            brace={platform.braceErecap.latencyP95Ms}
            max={Math.max(platform.baseline.latencyP95Ms, platform.braceErecap.latencyP95Ms, platform.sloMs)}
            suffix=" ms"
          />
          <MetricRow
            label="SLO violation"
            base={platform.baseline.sloViolationPct}
            brace={platform.braceErecap.sloViolationPct}
            max={100}
            suffix="%"
          />
          <MetricRow
            label="Token load"
            base={platform.baseline.tokens}
            brace={platform.braceErecap.tokens}
            max={Math.max(platform.baseline.tokens, platform.braceErecap.tokens)}
            suffix=""
          />
          <MetricRow
            label="Task success"
            base={platform.baseline.successPct}
            brace={platform.braceErecap.successPct}
            max={100}
            suffix="%"
          />
          <CallPathSlots focus={step.focus} />
          <div className="slo-note">
            <span>SLO</span>
            <strong>{platform.sloMs} ms</strong>
          </div>
          <p className="evidence-note">{step.evidenceNote}</p>
          <div className="module-note">
            <span>Modular claim</span>
            <p>
              E-RECAP is not the only point: it shows that compression, retrieval, or caching modules can be inserted
              into the replanning call path to change real-time closed-loop behavior.
            </p>
          </div>
        </aside>
      </section>

      {step.focus === "evidence" ? <EvidenceBoard /> : null}
    </main>
  );
}

function SceneState({ platform, step }: { platform: (typeof platforms)[number]; step: GuidedStep }) {
  const tokenDrop = Math.round((1 - platform.braceErecap.tokens / platform.baseline.tokens) * 100);
  const sloDrop = Math.round((1 - platform.braceErecap.sloViolationPct / platform.baseline.sloViolationPct) * 100);

  return (
    <div className="scene-state">
      <div>
        <span>Agents</span>
        <strong>{`K=${platform.agents}`}</strong>
      </div>
      <div>
        <span>Raw C_t</span>
        <strong>{`${step.rawTokens.toLocaleString()} tok`}</strong>
      </div>
      <div>
        <span>Trigger</span>
        <strong>{step.triggerType}</strong>
      </div>
      <div>
        <span>Gate</span>
        <strong>{gateLabel(step.gate)}</strong>
      </div>
      <div>
        <span>Token drop</span>
        <strong>{`${tokenDrop}%`}</strong>
      </div>
      <div>
        <span>SLO drop</span>
        <strong>{`${sloDrop}%`}</strong>
      </div>
    </div>
  );
}

function CallPathSlots({ focus }: { focus: FocusArea }) {
  return (
    <div className="callpath-slots">
      <span className="mini-heading">Pluggable call path</span>
      {callPathSlots.map((slot) => {
        const active =
          slot.id === focus ||
          (slot.id === "compression" && focus === "budget") ||
          (slot.id === "audit" && focus === "evidence");
        return (
          <div className={active ? "active" : ""} key={slot.id}>
            <strong>{slot.label}</strong>
            <span>{slot.detail}</span>
          </div>
        );
      })}
    </div>
  );
}

function LoopWorkbench({
  keptTokens,
  platform,
  step,
  tokens,
}: {
  keptTokens: Set<string>;
  platform: (typeof platforms)[number];
  step: GuidedStep;
  tokens: TokenSegment[];
}) {
  const keptCount = keptTokens.size;
  const prunedCount = tokens.length - keptCount;

  return (
    <section className={`stage-card focus-${step.focus}`} aria-label="Replanning loop workbench">
      <div className="workbench-header">
        <div>
          <span>Live loop</span>
          <strong>{loopNodes[loopIndex(step.node)].label}</strong>
        </div>
        <div>
          <span>Raw context</span>
          <strong>{step.rawTokens.toLocaleString()} tok</strong>
        </div>
        <div>
          <span>Planner input</span>
          <strong>{budgetLabel(step)}</strong>
        </div>
        <div>
          <span>Call latency</span>
          <strong>{`${formatMs(totalPhaseMs(step))} ms`}</strong>
        </div>
      </div>

      <LoopRail activeNode={step.node} />

      <div className="workbench-grid">
        <TokenBuffer focus={step.focus} keptTokens={keptTokens} pressure={step.pressure} step={step} tokens={tokens} />
        <ControllerCard focus={step.focus} step={step} />
        <ErecapCard focus={step.focus} keptCount={keptCount} prunedCount={prunedCount} step={step} tokens={tokens} />
        <PlannerAuditCard focus={step.focus} keptTokens={keptTokens} platform={platform} step={step} tokens={tokens} />
      </div>

      <EvidenceRibbon focus={step.focus} />
    </section>
  );
}

function LoopRail({ activeNode }: { activeNode: LoopNode }) {
  const activeIndex = loopIndex(activeNode);

  return (
    <div className="loop-rail" style={{ "--pulse-index": activeIndex } as CSSProperties}>
      <span className="loop-pulse" />
      {loopNodes.map((node, index) => (
        <div className={index === activeIndex ? "active" : index < activeIndex ? "visited" : ""} key={node.id}>
          <strong>{node.label}</strong>
          <span>{node.detail}</span>
        </div>
      ))}
    </div>
  );
}

function TokenBuffer({
  focus,
  keptTokens,
  pressure,
  step,
  tokens,
}: {
  focus: FocusArea;
  keptTokens: Set<string>;
  pressure: number;
  step: GuidedStep;
  tokens: TokenSegment[];
}) {
  const laneTokens = useMemo(() => {
    const lanes: Record<LaneId, { token: TokenSegment; index: number }[]> = {
      task: [],
      history: [],
      messages: [],
      observations: [],
      latest: [],
    };
    tokens.forEach((token, index) => {
      const lane = tokenLane(token, index);
      if (lanes[lane].length < 9) lanes[lane].push({ token, index });
    });
    return lanes;
  }, [tokens]);

  return (
    <article className={focus === "context" || focus === "pressure" ? "work-module active" : "work-module"}>
      <div className="module-title">
        <span>Context buffer C_t</span>
        <strong>{Math.round(pressure * 100)}% pressure</strong>
      </div>
      <div className="pressure-meter">
        <div style={{ width: `${pressure * 100}%` }} />
      </div>
      <div className="context-lanes" aria-label="Context token lanes">
        {(Object.keys(laneMeta) as LaneId[]).map((lane) => (
          <div className={`context-lane lane-${lane}`} key={lane}>
            <div className="lane-label">
              <strong>{laneMeta[lane].short}</strong>
              <span>{laneMeta[lane].description}</span>
            </div>
            <div className="lane-tokens">
              {laneTokens[lane].map(({ token, index }) => (
                <div
                  className={tokenClass(token, keptTokens)}
                  key={token.id}
                  style={{ "--delay": `${index * 13}ms`, "--utility": token.utility.toFixed(2) } as CSSProperties}
                >
                  <div className="chip-bars">
                    {utilityBars(token).map((height, barIndex) => (
                      <i key={`${token.id}-${barIndex}`} style={{ height: `${height}%` }} />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="module-footer">
        <span>{`N_t ${step.rawTokens.toLocaleString()}`}</span>
        <span>{`K ${platformAgentCount(step.platformId)}`}</span>
        <span>{focus === "pressure" ? "new trigger queued" : "new observation appended"}</span>
      </div>
    </article>
  );
}

function platformAgentCount(platformId: PlatformId) {
  return platforms.find((item) => item.id === platformId)?.agents ?? 1;
}

function ControllerCard({ focus, step }: { focus: FocusArea; step: GuidedStep }) {
  const checks = [
    { id: "trigger", label: "tau_t", value: step.checks.trigger },
    { id: "cooldown", label: "Delta >= delta", value: step.checks.cooldown },
    { id: "commit", label: "kappa >= omega", value: step.checks.commit },
    { id: "override", label: "override", value: step.checks.override },
  ];

  return (
    <article className={focus === "gate" || focus === "budget" ? "work-module active" : "work-module"}>
      <div className="module-title">
        <span>Controller</span>
        <strong>{gateLabel(step.gate)}</strong>
      </div>
      <div className="controller-flow">
        <div className={`controller-node trigger ${step.triggerType}`}>{step.triggerType}</div>
        <div className={`controller-node gate ${step.gate}`}>{gateLabel(step.gate)}</div>
        <div className="controller-node budget">B_t {budgetLabel(step)}</div>
      </div>
      <div className="gate-checks">
        {checks.map((check) => (
          <div className={check.value ? "pass" : "fail"} key={check.id}>
            <span>{check.label}</span>
            <strong>{check.value ? "yes" : "no"}</strong>
          </div>
        ))}
      </div>
      <div className="counter-grid">
        <Counter label="cooldown" max={step.counters.delta} value={step.counters.sinceReplan} />
        <Counter label="commit" max={step.counters.omega} value={step.counters.sincePlanChange} />
        <Counter label="failure" max={3} value={step.counters.failureWindow} />
      </div>
    </article>
  );
}

function Counter({ label, max, value }: { label: string; max: number; value: number }) {
  const pct = Math.min(100, Math.round((value / Math.max(1, max)) * 100));

  return (
    <div className="counter">
      <span>{label}</span>
      <div>
        <i style={{ width: `${pct}%` }} />
      </div>
      <strong>{`${value}/${max}`}</strong>
    </div>
  );
}

function ErecapCard({
  focus,
  keptCount,
  prunedCount,
  step,
  tokens,
}: {
  focus: FocusArea;
  keptCount: number;
  prunedCount: number;
  step: GuidedStep;
  tokens: TokenSegment[];
}) {
  const heatTokens = tokens.slice(8, 48);

  return (
    <article className={focus === "compression" || focus === "budget" ? "work-module active" : "work-module"}>
      <div className="module-title">
        <span>E-RECAP processor</span>
        <strong>{`${step.layerCounts[0]} -> ${keptCount}`}</strong>
      </div>
      <div className="utility-heatmap" aria-label="Token utility heatmap">
        {heatTokens.map((token, index) => (
          <span
            className={token.utility > 0.68 ? "hot" : token.utility < 0.38 ? "cold" : ""}
            key={token.id}
            style={
              {
                "--utility": token.utility.toFixed(2),
                "--delay": `${index * 11}ms`,
              } as CSSProperties
            }
          />
        ))}
      </div>
      <div className="layer-stack">
        {step.layerCounts.map((count, index) => (
          <div className={index === step.layerCounts.length - 1 ? "layer-row final" : "layer-row"} key={`${count}-${index}`}>
            <span>{index === 0 ? "Raw" : `L${index}`}</span>
            <div>
              <i style={{ width: metricWidth(count, step.layerCounts[0]) }} />
            </div>
            <strong>{count}</strong>
          </div>
        ))}
      </div>
      <div className="anchor-strip">
        <span>head pinned</span>
        <span>{`${keptCount} kept`}</span>
        <span>{`${prunedCount} pruned`}</span>
      </div>
      <div className="pruned-bin">
        <span>pruned middle</span>
        <div>
          {Array.from({ length: Math.min(12, prunedCount) }, (_, index) => (
            <i key={index} />
          ))}
        </div>
      </div>
    </article>
  );
}

function PlannerAuditCard({
  focus,
  keptTokens,
  platform,
  step,
  tokens,
}: {
  focus: FocusArea;
  keptTokens: Set<string>;
  platform: (typeof platforms)[number];
  step: GuidedStep;
  tokens: TokenSegment[];
}) {
  const totalMs = totalPhaseMs(step);
  const kept = tokens.filter((token) => keptTokens.has(token.id)).slice(0, 16);

  return (
    <article className={focus === "audit" || focus === "evidence" ? "work-module active" : "work-module"}>
      <div className="module-title">
        <span>Planner + audit</span>
        <strong>{step.gate === "commit" || step.gate === "cooldown" ? "no call" : `${formatMs(totalMs)} ms`}</strong>
      </div>
      <div className="planner-input-strip">
        {kept.map((token, index) => (
          <span className={token.protectedSlot ? token.protectedSlot : "kept"} key={token.id} style={{ "--delay": `${index * 18}ms` } as CSSProperties} />
        ))}
      </div>
      <div className="phase-lanes">
        {(Object.keys(step.phaseMs) as PhaseName[]).map((phase, index) => (
          <div className={step.phaseMs[phase] > 0 ? "active" : ""} key={phase}>
            <span>{phase}</span>
            <div>
              <i style={{ width: metricWidth(step.phaseMs[phase], Math.max(...Object.values(step.phaseMs), 1)), "--delay": `${index * 90}ms` } as CSSProperties} />
            </div>
            <strong>{formatMs(step.phaseMs[phase])} ms</strong>
          </div>
        ))}
      </div>
      <div className={totalMs <= platform.sloMs ? "slo-gauge pass" : "slo-gauge fail"}>
        <div>
          <span>l_t</span>
          <strong>{step.gate === "commit" || step.gate === "cooldown" ? "held" : `${formatMs(totalMs)} ms`}</strong>
        </div>
        <div>
          <span>SLO_t</span>
          <strong>{platform.sloMs} ms</strong>
        </div>
      </div>
      <div className="audit-log">{step.auditLine}</div>
    </article>
  );
}

function EvidenceRibbon({ focus }: { focus: FocusArea }) {
  return (
    <div className={focus === "evidence" ? "evidence-ribbon active" : "evidence-ribbon"}>
      {platforms.map((item) => (
        <div key={item.id}>
          <span>{item.label}</span>
          <strong>{`${item.baseline.sloViolationPct}% to ${item.braceErecap.sloViolationPct}%`}</strong>
        </div>
      ))}
    </div>
  );
}

function EvidenceBoard() {
  return (
    <section className="evidence-board">
      <div className="panel-header">
        <span>Cross-platform evidence</span>
        <h2>Baseline replanning can succeed while missing deadlines.</h2>
      </div>
      <div className="platform-grid">
        {platforms.map((item) => (
          <article className="platform-card" key={item.id}>
            <div>
              <strong>{item.label}</strong>
              <span>{item.scenario}</span>
            </div>
            <MetricRow
              label="Tokens"
              base={item.baseline.tokens}
              brace={item.braceErecap.tokens}
              max={Math.max(item.baseline.tokens, item.braceErecap.tokens)}
              suffix=""
            />
            <MetricRow
              label="SLO violation"
              base={item.baseline.sloViolationPct}
              brace={item.braceErecap.sloViolationPct}
              max={100}
              suffix="%"
            />
          </article>
        ))}
      </div>
    </section>
  );
}

function MetricRow({
  base,
  brace,
  label,
  max,
  suffix,
}: {
  base: number;
  brace: number;
  label: string;
  max: number;
  suffix: string;
}) {
  return (
    <div className="metric-row">
      <div className="metric-label">
        <span>{label}</span>
        <strong>{`${brace}${suffix}`}</strong>
      </div>
      <div className="bar-pair">
        <div>
          <span style={{ width: metricWidth(base, max) }} />
          <em>{`Baseline ${base}${suffix}`}</em>
        </div>
        <div>
          <span style={{ width: metricWidth(brace, max) }} />
          <em>{`BRACE+E-RECAP ${brace}${suffix}`}</em>
        </div>
      </div>
    </div>
  );
}

export default App;
