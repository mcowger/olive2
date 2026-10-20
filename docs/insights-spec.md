# Olive Insights — cross-meeting meta layer (spec)

Status: DRAFT for build. Owner: Phil. Target: fork `philmcneely/olive2`, new branch
`feat/insights`. Local-only; nothing leaves the box.

## 1. Goal

A layer on top of Olive's existing per-meeting transcripts that reads the **whole
corpus** and surfaces, with citations back to the exact call + moment:

- Trends over time (by week / month).
- Breakdowns by person and by meeting type (tag / source).
- Emotional tone: am I angry / stressed in a given month or week?
- Interpersonal dynamics: was I too aggressive in specific meetings; did someone else
  get hostile or insulting — and in which calls.
- Outliers: a meeting or a person-week that deviates from that person's own baseline.
- **Good things / bad things:** wins to reinforce (recognition, strong collaboration,
  de-escalation, a clean repair after tension) and issues to work on (you escalated, an
  accusation, contempt/dismissiveness), each with the call + moment.
- **Coaching guidance:** grounded, constructive takeaways — what went well to keep doing,
  and what to try differently — per person and overall.

Every number drills to the meeting(s) and the transcript timestamp where it happened
(house rule: analytics lead with the finding, every figure drills down).

## 1a. The questions a normal person actually wants answered

These are the top-line questions the UI must answer directly (each drills to the calls):

- **Me over time:** Am I getting nicer / calmer / more patient? Where did I regress?
- **Reactions to me:** Do certain attendees consistently react badly to me? Who do I bring
  out the best / worst in?
- **Relationship trajectory:** Do people warm up to me over time? Is a specific relationship
  degrading (cooling, more friction, shorter/colder exchanges)?
- **Wins & flags:** Where did I handle something well; where was I too aggressive; where did
  someone cross a line toward me.
- **The needles:** surface the specific pivotal calls (see §12) on demand.

The analysis unit is therefore not just the person but the **dyad over time** (me↔each
attendee, and attendee→me reactions), bucketed by week/month.

## 2. Non-goals (v1)

- No external LLMs or cloud calls — sensitive relational/emotional data stays local.
- No real-time/live scoring; signals are computed from finalized transcripts.
- No clinical/psychological **diagnosis**. We apply communication- and behavioral-science
  *lenses* for self-reflection and coaching, always backed by evidence quotes — never a
  mental-health label about a person.
- No new ML models; reuse the existing diarization + speaker-ID + the LLM service.

## 3. Why Olive can do this already (enablers in-repo)

- Diarized transcripts with per-turn timestamps and a **canonical speaker id across
  meetings** (voiceprint library) → per-person aggregation across the corpus is possible.
- `meetings.start_time` (epoch ms) → week/month bucketing.
- `tags` + `source` → "by type of meeting".
- Existing LLM service (`services/llm.py`, used by summaries/chat) → tone extraction is a
  prompt, not new infra. Route via the fleet proxy for attribution; cheapest passing model.
- Artifact model + transcript search → provenance + linking.

## 4. Design decisions

1. **LLM-driven, not hardcoded.** Dimensions, scoring guidance, and moment-detection live
   in editable prompt templates (config), never as hardcoded thresholds/keyword lists.
   Code is plumbing only. (Matches: LLM-driven-not-hardcoded, no-numerical-prompt-guidance.)
2. **Two granularities, stored normalized + raw:**
   - `meeting_speaker_signals` — one row per (meeting, canonical speaker): aggregated
     per-dimension scores + talk-time + interruption count for that speaker in that call.
   - `meeting_moments` — the citable events: (meeting, speaker, start_ms, end_ms,
     dimension, severity, quote, rationale). This is "the call where it happened".
   - The raw LLM JSON is also persisted as an `artifact` (kind=`signals`) for provenance
     and re-aggregation without re-running the LLM.
3. **Rollups computed at query time** first (by month/person/tag, with per-person baseline
   + z-score outliers). Materialize later only if measured too slow.
4. **Pipeline reuses DBOS** (the sole enqueue path): a new stage job `analyze_signals`
   enqueued when a transcript finalizes, plus a **backfill driver** over already-transcribed
   meetings. Dedup by meeting; idempotent (re-run replaces that meeting's rows).
5. **Privacy gate:** feature + backfill behind a setting (`OLIVE_INSIGHTS_ENABLED`),
   default on but local-only; no egress path.
6. **Scores are relative to the person's own baseline**, not an absolute "aggression score"
   — the useful signal is deviation ("March is well above your own norm"), not a label.

## 5. Data model (SQLite, epoch-ms ints, UUID text ids — repo invariants)

New tables (Alembic migration, following `db/migrations.py` + `migrations/` pattern):

```
meeting_speaker_signals(
  id TEXT PK, meeting_id TEXT FK, speaker_id TEXT NULL,   -- null = unmatched speaker
  dimensions TEXT,            -- JSON: {emotion, valence, hostility, insult, dominance, stress...} each -1..1 or 0..1
  talk_ms INTEGER, turn_count INTEGER, interrupt_count INTEGER,
  model TEXT, provider TEXT, created_at INTEGER
)
meeting_moments(
  id TEXT PK, meeting_id TEXT FK, speaker_id TEXT NULL, toward_speaker_id TEXT NULL,
  start_ms INTEGER, end_ms INTEGER,
  event_type TEXT,            -- LLM-chosen open label: accusation, termination, praise, repair...
  valence TEXT,               -- 'good' | 'bad' | 'neutral'
  framework TEXT NULL,        -- e.g. 'gottman:contempt', 'nvc:demand' (citation lens, not hardcoded)
  dimension TEXT, severity REAL, quote TEXT, rationale TEXT, guidance TEXT NULL,
  created_at INTEGER
)
```
Plus a **dyad** rollup for relationship trajectories (computed from the per-meeting rows,
keyed by the unordered pair + direction):
```
-- derived at query time first; materialize if slow
relationship(pair = {speakerA, speakerB}, direction):
  per-month warmth/hostility/respect series, trend (warming|cooling|stable|volatile),
  reaction profile (how B responds to A), linked meetings per point
```
Artifact: `kind='signals'` holds the raw per-meeting LLM JSON.

## 6. Signal-extraction (per meeting)

Input: the diarized, speaker-attributed transcript (already produced). Prompt → structured
JSON (validated by a Pydantic schema): for each speaker, per-dimension scores + counts; a
list of notable **moments** with turn index/timestamp, dimension, severity, a short verbatim
quote, and one-line rationale. Long transcripts chunked; scores aggregated across chunks.
Model via proxy, cheapest passing; prompt template in config, user-editable.

Moments cover not just tone but **significant interpersonal events** — the model labels the
event type itself (LLM-driven, open vocabulary; not a hardcoded enum), e.g. accusation /
blame, termination or firing, threat/ultimatum, apology, praise/recognition, conflict
escalation. Each moment stores a type label + verbatim quote + who→whom + timestamp, so a
single high-signal exchange is recoverable even when the meeting's aggregate tone is mild.

## 6a. Event retrieval ("find the call where…")

A natural-language lookup over `meeting_moments` + transcript search: given a description
("the meeting where the other person accused me of sabotage", "the one where I got fired"),
rank candidate moments/meetings and return the call(s) + timestamp + quote. Implementation:
filter moments by type/severity, then an LLM re-rank over the candidate quotes/snippets
(cheapest passing model, via proxy). This is a *query* over already-extracted signals — it
does not re-scan every transcript per query.

## 6b. Psychological / communication-science guidance (the lens)

The extraction and the coaching output are guided by **named, established frameworks**
carried in the (editable) prompt template, so results are principled and explainable rather
than ad-hoc vibes. Seed set:

- **Gottman conflict markers** — criticism, contempt, defensiveness, stonewalling (the "Four
  Horsemen"), plus repair attempts and bids for connection.
- **Nonviolent Communication** (observation vs evaluation, feelings/needs, requests vs
  demands) — distinguishes assertive from aggressive.
- **Assertive / aggressive / passive / passive-aggressive** stance classification.
- Active-listening & validation signals; interruption/dominance; accountability vs blame.

These are the LLM's *reference lens*, stated behaviorally in the prompt — **no hardcoded
keyword lists, no numeric thresholds in code** (house rules: LLM-driven-not-hardcoded,
no-numerical-prompt-guidance). Each flagged item cites the framework it maps to + the quote,
so a "bad thing" reads as "this looked like contempt (eye-rolling/dismissive) — here's the
line," and a "good thing" as "a clean repair attempt that de-escalated — here's the line."
Coaching guidance is phrased as constructive, non-judgmental suggestions grounded in the same
frameworks. The whole rubric lives in config and Phil can edit/extend it without a code change.

## 6c. Per-person communication playbook (actionable)

The highest-value output: for each frequent attendee, a short **"how to communicate with
them"** guide, derived — not guessed — by correlating *Phil's* style features in each
exchange with *their* reaction in the same call:

- **Directness vs context:** does this person respond better when I get to the point fast,
  or when I give more background? (Correlate Phil's brevity/directness with their reaction.)
- **Language that lands vs backfires:** word choices / framing that drew warmth or
  cooperation vs ones that drew friction, with example lines from real calls.
- **Tone & assertiveness fit:** where assertive worked, where it read as aggressive to them.
- **Timing/format:** long explanation vs TL;DR; ask vs tell.

Method: extract Phil's per-turn style features (directness, verbosity, warmth, jargon, ask vs
tell) alongside the counterpart's reaction valence; aggregate per dyad; LLM synthesizes the
playbook with evidence quotes. Presented as "do more of / ease off on," always citing the
calls. Still LLM-driven and evidence-backed — never a bare prescriptive rule with no example.

## 6d. Transcript import — no audio (Zoom etc.)

Insights must work on meetings that never had audio in Olive: import an existing transcript
(Zoom `.vtt`/`.txt`, SRT, or speaker-labeled text) as a first-class meeting.

- New ingest path `POST /api/transcripts/import` (multipart or JSON): parse the file into the
  same internal `Transcript` shape (speaker-labeled, timestamped segments) used by the ASR
  pipeline, create a meeting with `source='import'` (subtype e.g. `zoom`), write a transcript
  artifact directly, and set `start_time` from the file/metadata (fallback: user-supplied).
- **No recording row / no audio**; it SKIPS the DBOS transcription pipeline entirely and goes
  straight to the insights `analyze_signals` step.
- **Speaker identity:** imported transcripts carry display names, not voiceprints. Map names
  to canonical speakers by name (and let Phil confirm/merge), so dyad/per-person analysis
  still works. Record identity provenance = `name` vs `voiceprint` so mixed-source analysis
  is honest about confidence. "Me" is mapped from Phil's known display names + voiceprint.
- Parsers: start with Zoom VTT and plain `Speaker: text` text; SRT/others extensible.
- Dedup by content hash like audio ingest. Backward-compatible with existing `/api/ingest`.

This widens the corpus (more of the 500+) and is a clean, self-contained module.

## 7. API (FastAPI, camelCase; update openapi + regen TS types — repo rule 4)

- `GET /api/insights/overview?from&to&speaker&tag` → series by month, top outliers, per-dim
  corpus trend.
- `GET /api/insights/person/{speakerId}?from&to` → that person over time, baseline, outlier
  weeks/meetings, each linking to meetings.
- `GET /api/insights/meeting/{meetingId}` → that meeting's speaker signals + moments.
- Every response row carries `meetingId` + `startMs` for drill-down.

## 8. MCP (read tools, always-on; writes gated by OLIVE_MCP_WRITE)

- `insights_overview`, `person_insights`, `meeting_signals` (read).
- Backfill trigger as a write tool.

## 9. UI — Insights tab (Solid Glass, internal-tool house style)

- Timeline: per-dimension by month, switch person / all / tag.
- Per-person cards with baseline + flagged outlier weeks.
- Outliers feed: "You were notably sharp in these 4 calls" / "[X] turned hostile here" —
  each a chip that opens the meeting at that `startMs`.
- Drill: figure → meeting → transcript scrolled to the moment, quote highlighted.

## 10. Phases & delegation (Opus orchestrates/reviews/integrates)

- **P0 schema + migration** — Haiku (mechanical, follow existing migration pattern). TDD: a
  migration-applies test.
- **P1 extraction service + prompt template + Pydantic schema + DBOS `analyze_signals` stage
  + backfill driver** — Sonnet (bounded). TDD with an injected fake LLM returning canned
  JSON; no real model in tests (repo rule: no ad-hoc ML imports in tests).
- **P2 aggregation/query service + API endpoints + openapi/TS regen** — Sonnet. TDD on
  bucketing, baseline, outlier math with fixture rows.
- **P3 MCP read tools + backfill write tool** — Haiku/Sonnet. TDD.
- **P4 Insights frontend (Solid Glass) + drill-down** — Sonnet. e2e-runner for the flow.
- **P5 E2E + security-reviewer pass + CHANGELOG** — Opus + security-reviewer.

Shared checkout: each phase owns distinct files (house rule) to avoid write conflicts.
Each phase is reviewed by Opus before the next starts; failing tests block progress
(never weaken a gate to converge).

## 11. Cost / run

Backfill = one LLM pass per transcribed meeting (~680). Piggybacks on the local models via
the proxy; runs as a bounded DBOS queue like transcription. Each new meeting scores once as
its transcript finalizes. All attributed through the proxy (Switchboard).

## 12. Acceptance / validation cases ("find the needle in 500+")

Validation anchors — the system must surface these from the full corpus without being told
where. Do NOT over-fit the design to them; they prove moment-extraction + event-retrieval
work end to end:

- **Accusation:** the call where another person accused Phil of sabotage (or similar) →
  returns meeting + timestamp + verbatim quote, labeled as an accusation toward Phil.
- **Termination:** the call where Phil got fired → returns the meeting + the moment.

Trajectory sanity checks on real data:
- ≥1 dyad classified "warming" and ≥1 "cooling", each with supporting calls.
- A month where Phil's hostility/stress is flagged above his own baseline, with the calls.
- ≥1 per-person playbook recommendation backed by example lines (e.g. "X responds better to
  TL;DR — these 3 calls").

Method: run backfill, then query via natural-language description only; Phil confirms the
returned call is right. Confirmed cases become an eval fixture.

## 13. Open questions for Phil (non-blocking; defaults chosen)

- Anchor "me" on Phil's enrolled voiceprint so "me over time" + playbooks center on him? (yes)
- Attendee pages Phil-relative, plus that attendee's own cross-meeting view? (both)
- Backfill all ~680 now, or JPR/date-range first? (all, low-priority queue behind transcription)
