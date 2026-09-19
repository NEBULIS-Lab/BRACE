import { makeTokens, type TokenSegment } from './data.js';

export const CHAPTER_SECONDS = 12;
export const DURATION = CHAPTER_SECONDS * 5;
// Deterministic teaching context. IDs and scores stay fixed throughout the tour.
export const tokens = makeTokens(32, 7).map((token, i) => ({
  ...token,
  protectedSlot: i < 3 ? 'head' as const : i >= 29 ? 'tail' as const : null,
  kind: i < 3 ? 'task' as const : i >= 29 ? 'tail' as const : token.kind,
}));

export function selectTokens(context: TokenSegment[], budget: number) {
  const anchors = context.filter(t => t.protectedSlot);
  const count = Math.max(anchors.length, Math.min(context.length, Math.floor(budget)));
  const middle = context.filter(t => !t.protectedSlot).sort((a, b) => b.utility - a.utility);
  const ids = new Set([...anchors, ...middle.slice(0, count - anchors.length)].map(t => t.id));
  return context.filter(t => ids.has(t.id));
}

export type Trigger = 'routine' | 'unsafe' | 'ready';
// Three teaching cases from docs/CONTROLLER.md, not a replacement controller.
export function gateDecision(trigger: Trigger) {
  if (trigger === 'unsafe') return { mode: 'full_replan', calls: 1, label: 'Admit recovery', reason: 'Unsafe state overrides the stability windows.' };
  if (trigger === 'ready') return { mode: 'partial_replan', calls: 1, label: 'Admit + budget', reason: 'Windows have cleared. Assign a budget to the next call.' };
  return { mode: 'defer_replan', calls: 0, label: 'Keep executing', reason: 'Cooldown is active. This routine trigger can wait.' };
}

export function frameAt(seconds: number) {
  const time = Math.min(DURATION, Math.max(0, seconds));
  const chapter = Math.min(4, Math.floor(time / CHAPTER_SECONDS));
  return { chapter, progress: (time - chapter * CHAPTER_SECONDS) / CHAPTER_SECONDS };
}

export const phases = [
  { name: 'Compress', ms: 40 },
  { name: 'Retrieve', ms: 60 },
  { name: 'Planner', ms: 1400 },
  { name: 'Update', ms: 30 },
];
