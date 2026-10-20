import { useCallback, useEffect, useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { AlertTriangle, Quote, RefreshCw, Search, Sparkles } from "lucide-react";
import { Badge, Button, Card, CardBody, Input, notify } from "./ui";
import {
  InsightsApiError,
  insightsApi,
  meetingHref,
  type DyadTrend,
  type InsightsBucket,
  type InsightsCoaching,
  type InsightsDyad,
  type InsightsDyads,
  type InsightsEvidence,
  type InsightsFindResult,
  type InsightsGuidanceItem,
  type InsightsMe,
  type InsightsOutlier,
  type InsightsOverview,
  type InsightsPerson,
  type InsightsPlaybook,
  type InsightsSeriesPoint,
  type InsightsStatus,
  type InsightsUnattributed,
} from "../api/insights";

export interface InsightsSpeakerOption {
  id: string;
  name: string;
}

interface InsightsViewProps {
  /** Navigate the SPA to an app path (e.g. `/meetings/{id}?t=1234&q=quote`). */
  onNavigate: (path: string) => void;
  speakers: InsightsSpeakerOption[];
}

type Open = (meetingId: string, startMs?: number | null, quote?: string | null) => void;

function errorMessage(exc: unknown, fallback: string): string {
  return exc instanceof Error ? exc.message : fallback;
}

function fmtNum(n: number): string {
  return (Math.round(n * 100) / 100).toFixed(2);
}

function fmtDimension(name: string): string {
  return name.replace(/[_-]+/g, " ");
}

function trendVariant(t: DyadTrend): "success" | "warning" | "info" | "ghost" | "outline" {
  switch (t) {
    case "warming":
      return "success";
    case "cooling":
      return "warning";
    case "volatile":
      return "info";
    case "stable":
      return "ghost";
    default:
      return "outline";
  }
}

/** Shared "model-inferred" label: every judgment on this page is an LLM read of the transcript. */
function InferredBadge(): ReactElement {
  return (
    <span title="Scored by a language model from the transcript text. Treat as a lead to check, not a measurement.">
      <Badge variant="outline">model-inferred</Badge>
    </span>
  );
}

function SectionHeader({ title, hint, right }: { title: string; hint?: ReactNode; right?: ReactNode }): ReactElement {
  return (
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div className="min-w-0">
        <h2 className="text-lg font-semibold tracking-tight text-base-content">{title}</h2>
        {hint && <p className="text-xs text-base-content/70">{hint}</p>}
      </div>
      {right}
    </div>
  );
}

function UnattributedWarning({ unattributed, onOpen }: { unattributed: InsightsUnattributed; onOpen: Open }): ReactElement | null {
  if (!unattributed || unattributed.rows <= 0) return null;
  return (
    <div role="alert" className="alert alert-warning alert-soft text-sm">
      <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
      <div className="min-w-0">
        <span className="font-semibold">{unattributed.rows} speaker reads have no confident identity.</span>{" "}
        They are left out of per-person figures. Name the speakers in these meetings to include them:
        <span className="mt-1 flex flex-wrap gap-1.5">
          {unattributed.meetingIds.slice(0, 8).map((id) => (
            <button key={id} type="button" onClick={() => onOpen(id)} className="badge badge-sm badge-warning badge-outline cursor-pointer font-mono">
              {id.slice(0, 8)}
            </button>
          ))}
          {unattributed.meetingIds.length > 8 && <span className="text-xs">+{unattributed.meetingIds.length - 8} more</span>}
        </span>
      </div>
    </div>
  );
}

/** A clickable verbatim quote that opens its meeting at the quote's moment. */
function QuoteButton({
  quote,
  meetingId,
  startMs,
  onOpen,
  meta,
}: {
  quote: string;
  meetingId: string;
  startMs?: number | null;
  onOpen: Open;
  meta?: ReactNode;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={() => onOpen(meetingId, startMs, quote)}
      title="Open this moment in the transcript"
      className="group flex w-full min-w-0 items-start gap-2 rounded-lg border border-base-300 bg-base-100/60 px-3 py-2 text-left transition hover:border-primary/50"
    >
      <Quote className="mt-0.5 h-3.5 w-3.5 shrink-0 text-base-content/50 group-hover:text-primary" aria-hidden="true" />
      <span className="min-w-0 flex-1">
        <span className="block break-words text-sm italic text-base-content">&ldquo;{quote}&rdquo;</span>
        {meta && <span className="mt-1 block text-xs text-base-content/70">{meta}</span>}
      </span>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Trend chart                                                         */
/* ------------------------------------------------------------------ */

function TrendChart({ points, onOpen, label }: { points: InsightsSeriesPoint[]; onOpen: Open; label: string }): ReactElement {
  const W = 640;
  const H = 150;
  const PAD_X = 28;
  const PAD_Y = 18;
  if (points.length === 0) {
    return <p className="text-sm text-base-content/70">No data for this dimension yet.</p>;
  }
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const x = (i: number) => (points.length === 1 ? W / 2 : PAD_X + (i * (W - PAD_X * 2)) / (points.length - 1));
  const y = (v: number) => H - PAD_Y - ((v - min) / span) * (H - PAD_Y * 2);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.value).toFixed(1)}`).join(" ");
  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-40 w-full min-w-[420px] text-primary" role="img" aria-label={`${label} over time`}>
        <line x1={PAD_X} x2={W - PAD_X} y1={H - PAD_Y} y2={H - PAD_Y} className="stroke-base-300" strokeWidth="1" />
        <path d={path} fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
        {points.map((p, i) => (
          <g
            key={`${p.bucket}-${i}`}
            role="button"
            tabIndex={0}
            aria-label={`${p.bucket}: ${fmtNum(p.value)} across ${p.n} reads. Open meeting.`}
            className="cursor-pointer outline-none [&:focus-visible>circle]:stroke-base-content"
            onClick={() => p.meetingIds[0] && onOpen(p.meetingIds[0], p.startMs)}
            onKeyDown={(e) => {
              if ((e.key === "Enter" || e.key === " ") && p.meetingIds[0]) {
                e.preventDefault();
                onOpen(p.meetingIds[0], p.startMs);
              }
            }}
          >
            <title>{`${p.bucket} · ${fmtNum(p.value)} · n=${p.n}`}</title>
            <circle cx={x(i)} cy={y(p.value)} r={Math.min(9, 4 + Math.log2(p.n + 1))} className="fill-primary stroke-base-100" strokeWidth="2" />
            {(i === 0 || i === points.length - 1 || points.length <= 8) && (
              <text x={x(i)} y={H - 3} textAnchor="middle" className="fill-base-content/60 text-[10px]">
                {p.bucket}
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
}

function DimensionPicker({ dimensions, value, onChange }: { dimensions: string[]; value: string; onChange: (d: string) => void }): ReactElement | null {
  if (dimensions.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label="Dimension">
      {dimensions.map((d) => (
        <button
          key={d}
          type="button"
          aria-pressed={d === value}
          onClick={() => onChange(d)}
          className={`btn btn-xs min-h-8 capitalize ${d === value ? "btn-primary" : "btn-ghost border-base-300"}`}
        >
          {fmtDimension(d)}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Status bar + backfill                                               */
/* ------------------------------------------------------------------ */

function StatusBar({ status, onBackfill, backfilling }: { status: InsightsStatus | null; onBackfill: () => void; backfilling: boolean }): ReactElement {
  const pct = status && status.transcribedMeetings > 0 ? Math.round((status.analyzedMeetings / status.transcribedMeetings) * 100) : 0;
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-base-300 bg-base-200/60 p-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
          <span className="font-semibold text-base-content">Analysis coverage</span>
          {status ? (
            <span className="text-base-content/70">
              {status.analyzedMeetings} of {status.transcribedMeetings} transcribed meetings analyzed
              {status.pendingMeetings > 0 ? ` · ${status.pendingMeetings} pending` : ""}
            </span>
          ) : (
            <span className="text-base-content/70">Loading…</span>
          )}
        </div>
        <progress className="progress progress-primary w-full" value={status?.analyzedMeetings ?? 0} max={Math.max(status?.transcribedMeetings ?? 0, 1)} aria-label={`${pct}% analyzed`} />
      </div>
      <Button variant="tonal" onClick={onBackfill} disabled={backfilling || status?.pendingMeetings === 0} title="Queue analysis for meetings that have a transcript but no insights yet">
        <RefreshCw className={`h-4 w-4 shrink-0 ${backfilling ? "animate-spin" : ""}`} aria-hidden="true" />
        {backfilling ? "Queuing…" : "Backfill"}
      </Button>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Find the call                                                       */
/* ------------------------------------------------------------------ */

function FindCall({ onOpen }: { onOpen: Open }): ReactElement {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<InsightsFindResult[] | null>(null);

  const run = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      const data = await insightsApi.find(q);
      setResults(data.results);
    } catch (exc) {
      setResults(null);
      setError(errorMessage(exc, "Search failed"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardBody className="space-y-3">
        <SectionHeader title="Find the call" hint="Describe the moment in plain words, e.g. the time someone accused me of missing a deadline." right={<InferredBadge />} />
        <form onSubmit={run} className="flex flex-col gap-2 sm:flex-row">
          <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="When did I get blamed for…" aria-label="Find a call" className="flex-1" />
          <Button type="submit" variant="primary" disabled={loading || !query.trim()}>
            <Search className="h-4 w-4 shrink-0" aria-hidden="true" />
            {loading ? "Searching…" : "Find"}
          </Button>
        </form>
        {error && <p className="text-sm text-error">{error}</p>}
        {results && results.length === 0 && <p className="text-sm text-base-content/70">No matching moments found.</p>}
        {results && results.length > 0 && (
          <ul className="space-y-2">
            {results.map((r) => (
              <li key={r.momentId}>
                <QuoteButton
                  quote={r.quote}
                  meetingId={r.meetingId}
                  startMs={r.startMs}
                  onOpen={onOpen}
                  meta={
                    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                      <span className="font-medium text-base-content">{r.meetingTitle || r.meetingId.slice(0, 8)}</span>
                      {r.speaker && <span>{r.speaker}{r.toward ? ` → ${r.toward}` : ""}</span>}
                      <Badge variant={r.valence === "good" ? "success" : r.valence === "bad" ? "error" : "ghost"} soft>{r.eventType}</Badge>
                      <span>match {Math.round(r.score * 100)}%</span>
                      {r.reason && <span className="basis-full">{r.reason}</span>}
                    </span>
                  }
                />
              </li>
            ))}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Findings: outliers + wins & flags                                   */
/* ------------------------------------------------------------------ */

function OutlierList({ outliers, nameOf, onOpen }: { outliers: InsightsOutlier[]; nameOf: (id: string) => string; onOpen: Open }): ReactElement {
  if (outliers.length === 0) return <p className="text-sm text-base-content/70">No meetings stand out from anyone&apos;s own baseline.</p>;
  return (
    <ul className="grid grid-cols-1 gap-2 md:grid-cols-2">
      {outliers.slice(0, 8).map((o, i) => (
        <li key={`${o.meetingId}-${o.dimension}-${i}`}>
          <button
            type="button"
            onClick={() => onOpen(o.meetingId, o.startMs)}
            className="flex w-full items-center justify-between gap-3 rounded-lg border border-base-300 bg-base-100/60 px-3 py-2 text-left transition hover:border-primary/50"
          >
            <span className="min-w-0 text-sm">
              <span className="font-medium text-base-content">{nameOf(o.speakerId)}</span>{" "}
              <span className="capitalize text-base-content/70">{fmtDimension(o.dimension)}</span>{" "}
              <span className="text-base-content/70">{o.direction === "above" ? "above" : "below"} usual</span>
            </span>
            <span className="shrink-0 text-right text-xs text-base-content/70">
              <Badge variant={o.direction === "above" ? "warning" : "info"} soft>z {o.z > 0 ? "+" : ""}{fmtNum(o.z)}</Badge>
              <span className="block pt-0.5">{fmtNum(o.value)} vs {fmtNum(o.baselineMean)}</span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function WinsAndFlags({ overview, onOpen }: { overview: InsightsOverview; onOpen: Open }): ReactElement {
  // The overview carries the strongest moments already, so there is no per-meeting fetch.
  const pick = (valence: "good" | "bad") => (valence === "good" ? overview.topGoodMoments : overview.topBadMoments);

  const column = (valence: "good" | "bad", title: string, total: number) => {
    const rows = pick(valence);
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-base-content">{title}</h3>
          <Badge variant={valence === "good" ? "success" : "error"} soft>{total} total</Badge>
        </div>
        {rows.length === 0 ? (
          <p className="text-sm text-base-content/70">None in this range.</p>
        ) : (
          rows.map((m) => (
            <QuoteButton
              key={m.id}
              quote={m.quote}
              meetingId={m.meetingId}
              startMs={m.startMs}
              onOpen={onOpen}
              meta={
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <Badge variant={valence === "good" ? "success" : "error"} soft>{m.eventType}</Badge>
                  <span>{m.speakerLabel ?? "Unknown"}{m.towardLabel ? ` → ${m.towardLabel}` : ""}</span>
                  {m.rationale && <span className="basis-full">{m.rationale}</span>}
                </span>
              }
            />
          ))
        )}
      </div>
    );
  };

  return (
    <Card>
      <CardBody className="space-y-3">
        <SectionHeader
          title="Wins & flags"
          hint="Strongest cited moments in this range. Counts cover every moment."
          right={<InferredBadge />}
        />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {column("good", "Wins", overview.moments.good)}
          {column("bad", "Flags", overview.moments.bad)}
        </div>
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Relationship trajectory                                             */
/* ------------------------------------------------------------------ */

function DyadsPanel({ dyads, onOpen, selectedId, onSelect }: { dyads: InsightsDyads; onOpen: Open; selectedId: string | null; onSelect: (id: string) => void }): ReactElement {
  const [expanded, setExpanded] = useState<string | null>(null);
  const label = (d: InsightsDyad) => {
    const other = d.direction === "meToThem" ? d.toName : d.fromName;
    return `${d.direction === "meToThem" ? "Me → " : ""}${other ?? d.otherSpeakerId.slice(0, 8)}${d.direction === "themToMe" ? " → me" : ""}`;
  };
  const sorted = [...dyads.dyads].sort((a, b) => b.n - a.n);
  return (
    <Card>
      <CardBody className="space-y-3">
        <SectionHeader title="Relationship trajectory" hint="How the tone between me and each person is moving, per direction." right={<InferredBadge />} />
        <UnattributedWarning unattributed={dyads.unattributed} onOpen={onOpen} />
        {!dyads.meId ? (
          <p className="text-sm text-base-content/70">Pick who &ldquo;me&rdquo; is above to see relationship trajectories.</p>
        ) : sorted.length === 0 ? (
          <p className="text-sm text-base-content/70">No attributed conversations with other people yet.</p>
        ) : (
          <ul className="space-y-2">
            {sorted.map((d) => {
              const key = `${d.fromSpeakerId}>${d.toSpeakerId}`;
              const open = expanded === key;
              return (
                <li key={key} className="rounded-lg border border-base-300 bg-base-100/60">
                  <div className="flex flex-wrap items-center gap-2 px-3 py-2">
                    <button type="button" onClick={() => onSelect(d.otherSpeakerId)} className={`link-hover text-sm font-medium ${selectedId === d.otherSpeakerId ? "text-primary" : "text-base-content"}`} title="Show this person's card">
                      {label(d)}
                    </button>
                    <Badge variant={trendVariant(d.trend)}>{d.trend}</Badge>
                    <span className="text-xs text-base-content/70">n={d.n}</span>
                    <span className="text-xs text-success">+{d.moments.good}</span>
                    <span className="text-xs text-error">&minus;{d.moments.bad}</span>
                    <button type="button" onClick={() => setExpanded(open ? null : key)} aria-expanded={open} className="btn btn-ghost btn-xs ml-auto">
                      {open ? "Hide" : "Over time"}
                    </button>
                  </div>
                  {open && (
                    <div className="flex flex-wrap gap-1.5 border-t border-base-300 px-3 py-2">
                      {d.series.map((p, i) => (
                        <button
                          key={`${p.bucket}-${i}`}
                          type="button"
                          onClick={() => p.meetingIds[0] && onOpen(p.meetingIds[0], p.startMs)}
                          title={`Open the ${p.bucket} meeting`}
                          className={`badge badge-sm cursor-pointer ${p.tone >= 0 ? "badge-success" : "badge-error"} badge-soft`}
                        >
                          {p.bucket} · {p.tone > 0 ? "+" : ""}{fmtNum(p.tone)} (n={p.n})
                        </button>
                      ))}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Guidance (playbook + coaching share a shape)                        */
/* ------------------------------------------------------------------ */

function GuidanceList({ title, items, tone, onOpen }: { title: string; items: InsightsGuidanceItem[]; tone: "success" | "warning"; onOpen: Open }): ReactElement {
  const ev = (e: InsightsEvidence, key: string, prefix?: string) => (
    <QuoteButton key={key} quote={e.quote} meetingId={e.meetingId} startMs={e.startMs} onOpen={onOpen} meta={prefix} />
  );
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-base-content">
        <Badge variant={tone} soft>{title}</Badge>
      </h3>
      {items.length === 0 ? (
        <p className="text-sm text-base-content/70">Nothing yet.</p>
      ) : (
        <ul className="space-y-3">
          {items.map((it, i) => (
            <li key={i} className="space-y-1.5">
              <p className="text-sm font-medium text-base-content">{it.point}</p>
              {it.suggestion && <p className="text-sm text-base-content/70">{it.suggestion}</p>}
              {it.evidence.map((e, j) => ev(e, `e${j}`))}
              {it.counterExample && ev(it.counterExample, "c", "Counter-example")}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function SampleNote({ sampleSize, cached }: { sampleSize?: Record<string, number> | null; cached: boolean }): ReactElement {
  const parts = sampleSize ? Object.entries(sampleSize).map(([k, v]) => `${v} ${fmtDimension(k)}`) : [];
  return (
    <p className="text-xs text-base-content/70">
      {parts.length > 0 ? `Based on ${parts.join(", ")}. ` : ""}
      {cached ? "Cached result." : "Freshly generated."}
    </p>
  );
}

function PlaybookPanel({ speakerId, name, onOpen }: { speakerId: string; name: string; onOpen: Open }): ReactElement {
  const [data, setData] = useState<InsightsPlaybook | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
  }, [speakerId]);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await insightsApi.playbook(speakerId));
    } catch (exc) {
      setError(errorMessage(exc, "Failed to build playbook"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3 rounded-xl border border-base-300 bg-base-100/60 p-3">
      <SectionHeader
        title={`Playbook for ${name}`}
        hint="What has landed well and badly with this person, from the cited moments."
        right={
          <div className="flex items-center gap-2">
            <InferredBadge />
            <Button size="sm" variant="tonal" onClick={() => void load()} disabled={loading}>
              <Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" />
              {loading ? "Building…" : data ? "Rebuild" : "Build playbook"}
            </Button>
          </div>
        }
      />
      {error && <p className="text-sm text-error">{error}</p>}
      {data?.insufficientData && <p className="text-sm text-warning">Not enough analyzed conversation with {name} to build a reliable playbook.</p>}
      {data && !data.insufficientData && (
        <div className="space-y-3">
          {data.summary && <p className="text-sm text-base-content">{data.summary}</p>}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <GuidanceList title="Do more" items={data.doMore} tone="success" onOpen={onOpen} />
            <GuidanceList title="Ease off" items={data.easeOff} tone="warning" onOpen={onOpen} />
          </div>
          <SampleNote sampleSize={data.sampleSize} cached={data.cached} />
        </div>
      )}
    </div>
  );
}

function CoachingPanel({ speakerId, name, onOpen }: { speakerId: string | null; name: string | null; onOpen: Open }): ReactElement {
  const [data, setData] = useState<InsightsCoaching | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
  }, [speakerId]);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await insightsApi.coaching(speakerId));
    } catch (exc) {
      setError(errorMessage(exc, "Failed to build coaching"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardBody className="space-y-3">
        <SectionHeader
          title="Coaching"
          hint={speakerId ? `Coaching drawn from conversations with ${name ?? "this person"}.` : "Coaching drawn from all analyzed conversations."}
          right={
            <div className="flex items-center gap-2">
              <InferredBadge />
              <Button size="sm" variant="tonal" onClick={() => void load()} disabled={loading}>
                <Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" />
                {loading ? "Generating…" : data ? "Regenerate" : "Generate"}
              </Button>
            </div>
          }
        />
        {error && <p className="text-sm text-error">{error}</p>}
        {data?.insufficientData && <p className="text-sm text-warning">Not enough analyzed conversation to coach from yet.</p>}
        {data && !data.insufficientData && (
          <div className="space-y-3">
            {data.summary && <p className="text-sm text-base-content">{data.summary}</p>}
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <GuidanceList title="Strengths" items={data.strengths} tone="success" onOpen={onOpen} />
              <GuidanceList title="Growth areas" items={data.growthAreas} tone="warning" onOpen={onOpen} />
            </div>
            <SampleNote sampleSize={data.sampleSize} cached={data.cached} />
          </div>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Per-person card                                                     */
/* ------------------------------------------------------------------ */

function PersonCard({ speakerId, name, bucket, onOpen }: { speakerId: string; name: string; bucket: InsightsBucket; onOpen: Open }): ReactElement {
  const [person, setPerson] = useState<InsightsPerson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dimension, setDimension] = useState("");

  useEffect(() => {
    let cancelled = false;
    setPerson(null);
    setError(null);
    insightsApi
      .person(speakerId, bucket)
      .then((data) => {
        if (cancelled) return;
        setPerson(data);
        setDimension((prev) => (prev && data.series[prev] ? prev : Object.keys(data.series)[0] ?? ""));
      })
      .catch((exc) => {
        if (!cancelled) setError(errorMessage(exc, "Failed to load person"));
      });
    return () => {
      cancelled = true;
    };
  }, [speakerId, bucket]);

  const dimensions = person ? Object.keys(person.series) : [];
  const base = dimension && person ? person.baseline[dimension] : undefined;
  const conf = person?.identityConfidence;

  return (
    <Card>
      <CardBody className="space-y-4">
        <SectionHeader title={person?.name ?? name} hint="Baseline and flagged weeks for this person, measured against their own history." right={<InferredBadge />} />
        {error && <p className="text-sm text-error">{error}</p>}
        {!person && !error && <p className="text-sm text-base-content/70">Loading…</p>}
        {person && (
          <>
            <UnattributedWarning unattributed={person.unattributed} onOpen={onOpen} />
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-base-300 bg-base-100/60 p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Meetings</div>
                <div className="mt-1 text-xl font-bold text-base-content">{person.meetings.length}</div>
              </div>
              <div className="rounded-xl border border-base-300 bg-base-100/60 p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Wins</div>
                <div className="mt-1 text-xl font-bold text-success">{person.moments.good}</div>
              </div>
              <div className="rounded-xl border border-base-300 bg-base-100/60 p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Flags</div>
                <div className="mt-1 text-xl font-bold text-error">{person.moments.bad}</div>
              </div>
              <div className="rounded-xl border border-base-300 bg-base-100/60 p-3" title="Share of this person's reads whose identity came from a spoken name, a voiceprint, or something weaker">
                <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Identity basis</div>
                <div className="mt-1 text-sm text-base-content">
                  {conf && conf.rows > 0
                    ? `${Math.round(conf.nameShare * 100)}% name · ${Math.round(conf.voiceprintShare * 100)}% voice · ${Math.round(conf.otherShare * 100)}% other`
                    : "No reads"}
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <DimensionPicker dimensions={dimensions} value={dimension} onChange={setDimension} />
              {base && (
                <p className="text-xs text-base-content/70">
                  Baseline for <span className="capitalize">{fmtDimension(dimension)}</span>: {fmtNum(base.mean)} &plusmn; {fmtNum(base.std)} over {base.n} meetings.
                </p>
              )}
              <TrendChart points={person.series[dimension] ?? []} onOpen={onOpen} label={`${name} ${fmtDimension(dimension)}`} />
            </div>

            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-base-content">Flagged weeks</h3>
              {person.outlierWeeks.length === 0 ? (
                <p className="text-sm text-base-content/70">No week stands out from this person&apos;s baseline.</p>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {person.outlierWeeks.map((w, i) => (
                    <button
                      key={`${w.bucket}-${w.dimension}-${i}`}
                      type="button"
                      onClick={() => w.meetingIds[0] && onOpen(w.meetingIds[0], w.startMs)}
                      title={`Open the ${w.bucket} meeting`}
                      className={`badge badge-soft cursor-pointer ${w.direction === "above" ? "badge-warning" : "badge-info"}`}
                    >
                      {w.bucket} · <span className="capitalize">{fmtDimension(w.dimension)}</span> {w.direction} (z {w.z > 0 ? "+" : ""}{fmtNum(w.z)}, n={w.n})
                    </button>
                  ))}
                </div>
              )}
            </div>

            {person.outlierMeetings.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-base-content">Outlier meetings</h3>
                <OutlierList outliers={person.outlierMeetings} nameOf={() => person.name ?? name} onOpen={onOpen} />
              </div>
            )}

            <PlaybookPanel speakerId={speakerId} name={person.name ?? name} onOpen={onOpen} />
          </>
        )}
      </CardBody>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* "Me" selector                                                       */
/* ------------------------------------------------------------------ */

function MePicker({ me, speakers, onChange }: { me: InsightsMe | null; speakers: InsightsSpeakerOption[]; onChange: (me: InsightsMe) => void }): ReactElement | null {
  const [saving, setSaving] = useState(false);
  if (!me) return null;
  const save = async (speakerId: string) => {
    setSaving(true);
    try {
      onChange(await insightsApi.setMe({ speakerId: speakerId || null, aliases: me.aliases }));
    } catch (exc) {
      notify("error", errorMessage(exc, "Failed to save"));
    } finally {
      setSaving(false);
    }
  };
  return (
    <label className="flex flex-wrap items-center gap-2 text-sm">
      <span className="font-medium text-base-content">I am</span>
      <select
        className="select select-sm w-full sm:w-56"
        value={me.speakerId ?? ""}
        disabled={saving}
        onChange={(e) => void save(e.target.value)}
        aria-label="Which speaker is me"
      >
        <option value="">Not set</option>
        {speakers.map((s) => (
          <option key={s.id} value={s.id}>{s.name}</option>
        ))}
      </select>
      {!me.speakerId && <span className="text-xs text-warning">Required for relationship and coaching views.</span>}
    </label>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export function InsightsView({ onNavigate, speakers }: InsightsViewProps): ReactElement {
  const [bucket, setBucket] = useState<InsightsBucket>("month");
  const [status, setStatus] = useState<InsightsStatus | null>(null);
  const [overview, setOverview] = useState<InsightsOverview | null>(null);
  const [dyads, setDyads] = useState<InsightsDyads | null>(null);
  const [me, setMe] = useState<InsightsMe | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [disabled, setDisabled] = useState(false);
  const [loading, setLoading] = useState(true);
  const [dimension, setDimension] = useState("");
  const [personId, setPersonId] = useState<string | null>(null);
  const [backfilling, setBackfilling] = useState(false);

  const open: Open = useCallback((meetingId, startMs, quote) => onNavigate(meetingHref(meetingId, startMs, quote)), [onNavigate]);

  const nameOf = useCallback((id: string) => speakers.find((s) => s.id === id)?.name ?? id.slice(0, 8), [speakers]);

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await insightsApi.status());
    } catch {
      // status is advisory; the main load surfaces real errors
    }
  }, []);

  const loadAll = useCallback(
    async (isCancelled: () => boolean = () => false) => {
      setLoading(true);
      setError(null);
      try {
        const [s, o, d, m] = await Promise.all([
          insightsApi.status(),
          insightsApi.overview({ bucket }),
          insightsApi.dyads(bucket),
          insightsApi.getMe(),
        ]);
        if (isCancelled()) return;
        setDisabled(false);
        setStatus(s);
        setOverview(o);
        setDyads(d);
        setMe(m);
        setDimension((prev) => (prev && o.series[prev] ? prev : Object.keys(o.series)[0] ?? ""));
      } catch (exc) {
        if (isCancelled()) return;
        if (exc instanceof InsightsApiError && exc.disabled) setDisabled(true);
        else setError(errorMessage(exc, "Failed to load insights"));
      } finally {
        if (!isCancelled()) setLoading(false);
      }
    },
    [bucket]
  );

  useEffect(() => {
    let cancelled = false;
    void loadAll(() => cancelled);
    return () => {
      cancelled = true;
    };
  }, [loadAll]);

  const handleBackfill = async () => {
    setBackfilling(true);
    try {
      const res = await insightsApi.backfill();
      notify("success", `Queued ${res.queued} meeting${res.queued === 1 ? "" : "s"} for analysis (${res.skipped} skipped).`);
      await refreshStatus();
    } catch (exc) {
      notify("error", errorMessage(exc, "Backfill failed"));
    } finally {
      setBackfilling(false);
    }
  };

  const dimensions = overview ? Object.keys(overview.series) : [];
  const selectedName = personId ? nameOf(personId) : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-base-content">Insights</h1>
          <p className="text-sm text-base-content/70">
            How conversations are going over time. Every figure is a link to the meeting and the moment behind it.
          </p>
        </div>
        <div className="join" role="group" aria-label="Group by">
          {(["week", "month"] as const).map((b) => (
            <button key={b} type="button" aria-pressed={bucket === b} onClick={() => setBucket(b)} className={`btn btn-sm join-item min-h-9 capitalize ${bucket === b ? "btn-primary" : ""}`}>
              {b}
            </button>
          ))}
        </div>
      </div>

      {disabled ? (
        <div role="alert" className="alert alert-info alert-soft text-sm">
          Insights is turned off on this server. Enable it in the backend settings, then reload.
        </div>
      ) : (
        <>
          <StatusBar status={status} onBackfill={() => void handleBackfill()} backfilling={backfilling} />
          <MePicker me={me} speakers={speakers} onChange={(next) => { setMe(next); void loadAll(); }} />
          {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-sm text-error">{error}</div>}
          <FindCall onOpen={open} />

          {loading && !overview ? (
            <p className="text-sm text-base-content/70">Loading insights…</p>
          ) : overview ? (
            <>
              <Card>
                <CardBody className="space-y-3">
                  <SectionHeader
                    title="Me over time"
                    hint={`Based on ${overview.coverage.analyzedMeetings} of ${overview.coverage.totalMeetings} meetings in range. Points sized by number of reads (n).`}
                    right={<InferredBadge />}
                  />
                  {overview.coverage.analyzedMeetings < overview.coverage.totalMeetings && (
                    <p className="text-xs text-warning">
                      {overview.coverage.totalMeetings - overview.coverage.analyzedMeetings} meetings are not analyzed yet, so trends may shift after a backfill.
                    </p>
                  )}
                  <UnattributedWarning unattributed={overview.unattributed} onOpen={open} />
                  <DimensionPicker dimensions={dimensions} value={dimension} onChange={setDimension} />
                  <TrendChart points={overview.series[dimension] ?? []} onOpen={open} label={fmtDimension(dimension)} />
                </CardBody>
              </Card>

              <Card>
                <CardBody className="space-y-3">
                  <SectionHeader title="Who reacts badly" hint="Biggest departures from each person's own baseline, strongest first." right={<InferredBadge />} />
                  <OutlierList outliers={overview.outliers} nameOf={nameOf} onOpen={open} />
                </CardBody>
              </Card>

              {dyads && <DyadsPanel dyads={dyads} onOpen={open} selectedId={personId} onSelect={setPersonId} />}

              <WinsAndFlags overview={overview} onOpen={open} />
            </>
          ) : null}

          <Card>
            <CardBody className="space-y-3">
              <SectionHeader title="Person" hint="Pick someone for their baseline, flagged weeks, and playbook." />
              <select
                className="select select-md w-full sm:w-72"
                aria-label="Person"
                value={personId ?? ""}
                onChange={(e) => setPersonId(e.target.value || null)}
              >
                <option value="">Choose a person…</option>
                {speakers.map((s) => (
                  <option key={s.id} value={s.id}>{s.name}</option>
                ))}
              </select>
            </CardBody>
          </Card>
          {personId && selectedName && <PersonCard speakerId={personId} name={selectedName} bucket={bucket} onOpen={open} />}

          <CoachingPanel speakerId={personId} name={selectedName} onOpen={open} />
        </>
      )}
    </div>
  );
}
