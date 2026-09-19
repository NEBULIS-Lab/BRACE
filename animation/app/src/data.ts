export type Mode = "baseline" | "brace" | "braceErecap";
export type PlatformId = "habitat" | "robofactory" | "airsim";

export type PlatformMetric = {
  id: PlatformId;
  label: string;
  scenario: string;
  sloMs: number;
  agents: number;
  baseline: {
    tokens: number;
    latencyP95Ms: number;
    sloViolationPct: number;
    successPct: number;
  };
  braceErecap: {
    tokens: number;
    latencyP95Ms: number;
    sloViolationPct: number;
    successPct: number;
  };
};

export const platforms: PlatformMetric[] = [
  {
    id: "habitat",
    label: "Meta Habitat",
    scenario: "Navigation",
    sloMs: 2500,
    agents: 1,
    baseline: {
      tokens: 235,
      latencyP95Ms: 2677,
      sloViolationPct: 85.5,
      successPct: 100,
    },
    braceErecap: {
      tokens: 20,
      latencyP95Ms: 2500,
      sloViolationPct: 4.7,
      successPct: 100,
    },
  },
  {
    id: "robofactory",
    label: "RoboFactory",
    scenario: "Pass-Shoe manipulation",
    sloMs: 250,
    agents: 2,
    baseline: {
      tokens: 1566,
      latencyP95Ms: 1604,
      sloViolationPct: 100,
      successPct: 100,
    },
    braceErecap: {
      tokens: 319,
      latencyP95Ms: 1213,
      sloViolationPct: 50,
      successPct: 100,
    },
  },
  {
    id: "airsim",
    label: "Microsoft AirSim",
    scenario: "K=8 intersection",
    sloMs: 2500,
    agents: 8,
    baseline: {
      tokens: 2934,
      latencyP95Ms: 8520,
      sloViolationPct: 100,
      successPct: 100,
    },
    braceErecap: {
      tokens: 1114,
      latencyP95Ms: 1640,
      sloViolationPct: 4.7,
      successPct: 100,
    },
  },
];

export const sceneLabels = [
  "Context Growth",
  "BRACE Gate",
  "E-RECAP Pruning",
  "SLO Evidence",
] as const;

export type TokenKind = "task" | "history" | "message" | "observation" | "tail";

export type TokenSegment = {
  id: string;
  kind: TokenKind;
  utility: number;
  protectedSlot: "head" | "tail" | null;
};

const kinds: TokenKind[] = ["task", "history", "message", "observation", "tail"];

export function makeTokens(count: number, seed: number): TokenSegment[] {
  return Array.from({ length: count }, (_, index) => {
    const head = index < 5;
    const tail = index > count - 8;
    const wave = Math.sin((index + 1) * (seed + 2.7)) * 0.5 + 0.5;
    const recencyBoost = tail ? 0.35 : 0;
    const taskBoost = head ? 0.28 : 0;
    const utility = Math.min(1, 0.12 + wave * 0.65 + recencyBoost + taskBoost);
    return {
      id: `tok-${seed}-${index}`,
      kind: head ? "task" : tail ? "tail" : kinds[(index + seed) % kinds.length],
      utility,
      protectedSlot: head ? "head" : tail ? "tail" : null,
    };
  });
}
