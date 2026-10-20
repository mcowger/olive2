// Typed fetch helpers for /api/insights/* and POST /api/transcripts/import.
//
// The generated schema.d.ts types these routes' 2xx bodies as `{}` (the FastAPI
// routes return a bare JSONResponse, so no response_model reaches openapi.json).
// The shapes below mirror the Pydantic models in backend/src/olive/schemas.py
// ("Olive Insights" section) until the routes declare response_model.

export type InsightsBucket = "week" | "month";
export type InsightsDirection = "above" | "below";
export type DyadTrend = "warming" | "cooling" | "stable" | "volatile" | "insufficient";

export interface InsightsMomentCounts {
  good: number;
  bad: number;
}

export interface InsightsCoverage {
  analyzedMeetings: number;
  totalMeetings: number;
}

export interface InsightsUnattributed {
  rows: number;
  meetingIds: string[];
}

/** `startMs` on series points / outliers is the meeting's start (epoch ms), not a transcript offset. */
export interface InsightsSeriesPoint {
  bucket: string;
  value: number;
  n: number;
  meetingIds: string[];
  startMs: number;
}

export interface InsightsDyadPoint extends InsightsSeriesPoint {
  tone: number;
  warmth?: number | null;
  hostility?: number | null;
  moments: InsightsMomentCounts;
}

export interface InsightsOutlier {
  speakerId: string;
  meetingId: string;
  startMs: number;
  dimension: string;
  value: number;
  z: number;
  direction: InsightsDirection;
  baselineMean: number;
  baselineStd: number;
}

export interface InsightsWeekOutlier {
  dimension: string;
  bucket: string;
  value: number;
  z: number;
  direction: InsightsDirection;
  n: number;
  meetingIds: string[];
  startMs: number;
}

export interface InsightsBaseline {
  mean: number;
  std: number;
  n: number;
}

export interface InsightsOverview {
  from?: number | null;
  to?: number | null;
  speakerId?: string | null;
  tag?: string | null;
  bucket: InsightsBucket;
  series: Record<string, InsightsSeriesPoint[]>;
  outliers: InsightsOutlier[];
  moments: InsightsMomentCounts;
  /** Strongest verified moments in scope, so the page can cite them without fetching each meeting. */
  topGoodMoments: InsightsMoment[];
  topBadMoments: InsightsMoment[];
  coverage: InsightsCoverage;
  unattributed: InsightsUnattributed;
}

export interface InsightsPerson {
  speakerId: string;
  name?: string | null;
  bucket: InsightsBucket;
  series: Record<string, InsightsSeriesPoint[]>;
  baseline: Record<string, InsightsBaseline>;
  outlierMeetings: InsightsOutlier[];
  outlierWeeks: InsightsWeekOutlier[];
  meetings: { meetingId: string; startMs: number; title?: string | null }[];
  moments: InsightsMomentCounts;
  identityConfidence: { rows: number; nameShare: number; voiceprintShare: number; otherShare: number };
  unattributed: InsightsUnattributed;
}

/** `startMs` on moments is the offset into the meeting transcript; `meetingStartMs` is the epoch start. */
export interface InsightsMoment {
  id: string;
  meetingId: string;
  startMs: number;
  endMs?: number | null;
  meetingStartMs: number;
  speakerId?: string | null;
  speakerLabel?: string | null;
  towardSpeakerId?: string | null;
  towardLabel?: string | null;
  eventType: string;
  valence: string;
  framework?: string | null;
  dimension?: string | null;
  severity?: number | null;
  quote: string;
  rationale?: string | null;
  guidance?: string | null;
  identitySource?: string | null;
}

export interface InsightsMeetingSignals {
  meetingId: string;
  title?: string | null;
  startMs: number;
  speakers: {
    speakerId?: string | null;
    speakerLabel: string;
    attributed: boolean;
    dimensions: Record<string, number>;
  }[];
  moments: InsightsMoment[];
}

export interface InsightsDyad {
  fromSpeakerId: string;
  toSpeakerId: string;
  fromName?: string | null;
  toName?: string | null;
  otherSpeakerId: string;
  direction: "meToThem" | "themToMe";
  series: InsightsDyadPoint[];
  trend: DyadTrend;
  n: number;
  moments: InsightsMomentCounts;
  meetingIds: string[];
  startMs: number;
}

export interface InsightsDyads {
  meId?: string | null;
  bucket: InsightsBucket;
  dyads: InsightsDyad[];
  unattributed: InsightsUnattributed;
}

export interface InsightsFindResult {
  momentId: string;
  meetingId: string;
  meetingTitle?: string | null;
  startMs: number;
  endMs?: number | null;
  quote: string;
  eventType: string;
  valence: string;
  speaker?: string | null;
  speakerId?: string | null;
  toward?: string | null;
  towardSpeakerId?: string | null;
  score: number;
  enacted?: boolean | null;
  reason?: string | null;
}

export interface InsightsEvidence {
  quote: string;
  meetingId: string;
  startMs?: number | null;
}

export interface InsightsGuidanceItem {
  point: string;
  suggestion?: string | null;
  evidence: InsightsEvidence[];
  counterExample?: InsightsEvidence | null;
}

export interface InsightsPlaybook {
  speakerId: string;
  person?: string | null;
  summary?: string | null;
  doMore: InsightsGuidanceItem[];
  easeOff: InsightsGuidanceItem[];
  sampleSize?: Record<string, number> | null;
  generatedAt?: number | null;
  cached: boolean;
  insufficientData: boolean;
}

export interface InsightsCoaching {
  scope: "all" | "speaker";
  speakerId?: string | null;
  summary?: string | null;
  strengths: InsightsGuidanceItem[];
  growthAreas: InsightsGuidanceItem[];
  sampleSize?: Record<string, number> | null;
  generatedAt?: number | null;
  cached: boolean;
  insufficientData: boolean;
}

export interface InsightsStatus {
  transcribedMeetings: number;
  analyzedMeetings: number;
  pendingMeetings: number;
}

export interface InsightsMe {
  speakerId?: string | null;
  aliases: string[];
}

export interface TranscriptImportResult {
  meetingId: string;
  deduped: boolean;
  format?: string | null;
  subtype?: string | null;
  segmentCount?: number | null;
  startTime?: number | null;
  speakers: { label: string; speakerId: string; identitySource: string }[];
  unmappedNames: string[];
  warnings: string[];
  timingSynthetic?: boolean | null;
}

/** Every /api/insights/* route answers 404 {"error":"insights disabled"} while the feature is off. */
export class InsightsApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "InsightsApiError";
    this.status = status;
  }
  get disabled(): boolean {
    return this.status === 404 && /insights disabled/i.test(this.message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const data = (await res.json().catch(() => ({}))) as { error?: string };
    throw new InsightsApiError(data.error || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

function qs(params: Record<string, string | number | null | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== null && v !== undefined && v !== "") sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export const insightsApi = {
  overview: (p: { bucket?: InsightsBucket; speaker?: string | null; tag?: string | null } = {}) =>
    request<InsightsOverview>(`/api/insights/overview${qs(p)}`),
  person: (speakerId: string, bucket?: InsightsBucket) =>
    request<InsightsPerson>(`/api/insights/person/${encodeURIComponent(speakerId)}${qs({ bucket })}`),
  meeting: (meetingId: string) =>
    request<InsightsMeetingSignals>(`/api/insights/meeting/${encodeURIComponent(meetingId)}`),
  dyads: (bucket?: InsightsBucket) => request<InsightsDyads>(`/api/insights/dyads${qs({ bucket })}`),
  find: (q: string, limit = 5) =>
    request<{ query: string; results: InsightsFindResult[] }>(`/api/insights/find${qs({ q, limit })}`),
  playbook: (speakerId: string) =>
    request<InsightsPlaybook>(`/api/insights/playbook/${encodeURIComponent(speakerId)}`),
  coaching: (speaker?: string | null) => request<InsightsCoaching>(`/api/insights/coaching${qs({ speaker })}`),
  getMe: () => request<InsightsMe>("/api/insights/me"),
  setMe: (body: { speakerId: string | null; aliases: string[] }) =>
    request<InsightsMe>("/api/insights/me", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  backfill: (body: { force?: boolean; limit?: number } = {}) =>
    request<{ queued: number; skipped: number }>("/api/insights/backfill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  status: () => request<InsightsStatus>("/api/insights/status"),
};

export function importTranscript(form: FormData): Promise<TranscriptImportResult> {
  return request<TranscriptImportResult>("/api/transcripts/import", { method: "POST", body: form });
}

/**
 * Deep link to a meeting at a transcript moment. Moment `startMs` is a transcript offset; series points and
 * outliers carry the meeting's epoch start instead, which has no transcript position, so those open the
 * meeting from the top.
 */
const EPOCH_MS_FLOOR = 1e11;
export function meetingHref(meetingId: string, startMs?: number | null, quote?: string | null): string {
  const base = `/meetings/${meetingId}`;
  if (startMs === null || startMs === undefined || startMs >= EPOCH_MS_FLOOR) return base;
  const q = quote?.trim() ? `&q=${encodeURIComponent(quote.trim().slice(0, 160))}` : "";
  return `${base}?t=${Math.max(0, Math.round(startMs))}${q}`;
}
