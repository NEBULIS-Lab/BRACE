import test from 'node:test';
import assert from 'node:assert/strict';
import { tokens, selectTokens, gateDecision, frameAt, DURATION } from '../.test-build/model.js';

test('compression preserves anchors, honors the budget, and retains original order', () => {
  for (const budget of [12, 20, 32]) {
    const kept = selectTokens(tokens, budget);
    assert.equal(kept.length, budget);
    assert.ok(tokens.filter(t => t.protectedSlot).every(t => kept.includes(t)));
    assert.deepEqual(kept.map(t => tokens.indexOf(t)), kept.map(t => tokens.indexOf(t)).sort((a,b) => a-b));
    const middle = kept.filter(t => !t.protectedSlot);
    const dropped = tokens.filter(t => !kept.includes(t));
    if (dropped.length && middle.length) assert.ok(Math.min(...middle.map(t => t.utility)) >= Math.max(...dropped.map(t => t.utility)));
  }
});

test('unsafe recovery overrides stability windows; routine triggers wait', () => {
  assert.equal(gateDecision('routine').mode, 'defer_replan');
  assert.equal(gateDecision('routine').calls, 0);
  assert.equal(gateDecision('unsafe').mode, 'full_replan');
  assert.equal(gateDecision('unsafe').calls, 1);
  assert.equal(gateDecision('ready').mode, 'partial_replan');
});

test('arbitrary seek is deterministic, including the final frame and reverse seek', () => {
  const before = frameAt(18);
  frameAt(49);
  assert.deepEqual(frameAt(18), before);
  assert.equal(frameAt(-1).chapter, 0);
  assert.equal(frameAt(DURATION).chapter, 4);
  assert.equal(frameAt(DURATION).progress, 1);
  assert.deepEqual(frameAt(DURATION + 100), frameAt(DURATION));
});
