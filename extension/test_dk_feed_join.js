// Offline validation of the DK-feed join invariant used by syncFromDKApi().
// Uses the REAL draftStatus board (picks 1-48) captured from draft 191317972 on
// 2026-06-13. Asserts: (1) pickToTeamSlot maps every userKey to exactly ONE slot
// across all rounds (the snake invariant), (2) there are exactly 12 teams, and
// (3) our entrant (dropdown userKey 312fc683…) lands in slot 10.
//
// Run: node extension/test_dk_feed_join.js

// ── the function under test (copied verbatim from content.js) ────────────────
function pickToTeamSlot(overallPick, totalTeams) {
  const roundIdx   = Math.floor((overallPick - 1) / totalTeams);
  const posInRound = (overallPick - 1) % totalTeams;
  return roundIdx % 2 === 0 ? posInRound + 1 : totalTeams - posInRound;
}

// ── real board: [overallSelectionNumber, userKey-prefix] ─────────────────────
const SELF = "312fc683";
const board = [
  [1,"e5f3f3fa"],[2,"70c70fc6"],[3,"1b8dbebb"],[4,"3640d46f"],[5,"39b8feaa"],
  [6,"247a81bf"],[7,"a66752d6"],[8,"1f667bd4"],[9,"a06f3a59"],[10,"312fc683"],
  [11,"60df08fb"],[12,"c97e140a"],[13,"c97e140a"],[14,"60df08fb"],[15,"312fc683"],
  [16,"a06f3a59"],[17,"1f667bd4"],[18,"a66752d6"],[19,"247a81bf"],[20,"39b8feaa"],
  [21,"3640d46f"],[22,"1b8dbebb"],[23,"70c70fc6"],[24,"e5f3f3fa"],[25,"e5f3f3fa"],
  [26,"70c70fc6"],[27,"1b8dbebb"],[28,"3640d46f"],[29,"39b8feaa"],[30,"247a81bf"],
  [31,"a66752d6"],[32,"1f667bd4"],[33,"a06f3a59"],[34,"312fc683"],[35,"60df08fb"],
  [36,"c97e140a"],[37,"c97e140a"],[38,"60df08fb"],[39,"312fc683"],[40,"a06f3a59"],
  [41,"1f667bd4"],[42,"a66752d6"],[43,"247a81bf"],[44,"39b8feaa"],[45,"3640d46f"],
  [46,"1b8dbebb"],[47,"70c70fc6"],[48,"e5f3f3fa"],
];

const TEAMS = 12;
let failures = 0;
const fail = (m) => { console.log("[FAIL] " + m); failures++; };

// (1) each userKey maps to exactly one slot across all its picks
const keyToSlots = {};
for (const [overall, key] of board) {
  const slot = pickToTeamSlot(overall, TEAMS);
  (keyToSlots[key] = keyToSlots[key] || new Set()).add(slot);
}
for (const [key, slots] of Object.entries(keyToSlots)) {
  if (slots.size !== 1) fail(`${key} mapped to multiple slots: ${[...slots]}`);
}
if (failures === 0) console.log("[PASS] every userKey maps to exactly one snake slot");

// (2) exactly 12 distinct teams
const nTeams = Object.keys(keyToSlots).length;
if (nTeams !== 12) fail(`expected 12 teams, got ${nTeams}`);
else console.log(`[PASS] exactly 12 distinct teams`);

// (3) self lands in slot 10
const selfSlot = [...keyToSlots[SELF]][0];
if (selfSlot !== 10) fail(`self ${SELF} expected slot 10, got ${selfSlot}`);
else console.log(`[PASS] self (${SELF}…) → slot 10`);

// (4) slots 1..12 all present and unique
const allSlots = new Set(Object.values(keyToSlots).map((s) => [...s][0]));
if (allSlots.size !== 12 || [...allSlots].some((s) => s < 1 || s > 12))
  fail(`slots not a clean 1..12 set: ${[...allSlots].sort((a,b)=>a-b)}`);
else console.log(`[PASS] slots are a clean 1..12 set`);

console.log(failures === 0 ? "\nALL CHECKS PASSED" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
