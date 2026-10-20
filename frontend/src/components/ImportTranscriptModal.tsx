import { useState } from "react";
import type { ReactElement } from "react";
import { Button, Field, FileInput, Input, Modal } from "./ui";
import { InsightsApiError, importTranscript, type TranscriptImportResult } from "../api/insights";

/** Import an existing text transcript (no audio) as a meeting. POSTs /api/transcripts/import. */
export function ImportTranscriptModal({
  onClose,
  onImported,
}: {
  onClose: () => void;
  onImported: (meetingId: string) => void;
}): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [startTime, setStartTime] = useState("");
  const [meAliases, setMeAliases] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsStartTime, setNeedsStartTime] = useState(false);
  const [result, setResult] = useState<TranscriptImportResult | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      if (title.trim()) form.append("title", title.trim());
      if (startTime) {
        const parsed = new Date(startTime);
        if (!Number.isNaN(parsed.getTime())) form.append("startTime", parsed.toISOString());
      }
      for (const alias of meAliases.split(",").map((a) => a.trim()).filter(Boolean)) {
        form.append("meAliases", alias);
      }
      const res = await importTranscript(form);
      if (res.warnings.length > 0 || res.unmappedNames.length > 0 || res.deduped) {
        setResult(res);
      } else {
        onImported(res.meetingId);
      }
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : String(exc);
      setError(message);
      // The backend answers 422 when the transcript carries no timestamp of its own.
      setNeedsStartTime(exc instanceof InsightsApiError && exc.status === 422);
    } finally {
      setSubmitting(false);
    }
  };

  if (result) {
    return (
      <Modal onClose={() => onImported(result.meetingId)} title="Transcript imported" ariaLabel="Transcript import result" maxWidth="max-w-md">
        <div className="space-y-3 text-sm">
          {result.deduped && <p className="text-base-content">This transcript was already imported; opening the existing meeting.</p>}
          <p className="text-base-content/70">
            {result.segmentCount ?? 0} turns{result.format ? ` · ${result.format}` : ""}
            {result.timingSynthetic ? " · timings estimated (the file had none)" : ""}
          </p>
          {result.unmappedNames.length > 0 && (
            <div className="alert alert-warning alert-soft">
              <span>No known person matched: {result.unmappedNames.join(", ")}. Name them from the meeting so insights can attribute them.</span>
            </div>
          )}
          {result.warnings.map((w, i) => (
            <div key={i} className="alert alert-warning alert-soft">
              <span>{w}</span>
            </div>
          ))}
          <div className="flex justify-end">
            <Button variant="primary" onClick={() => onImported(result.meetingId)}>Open meeting</Button>
          </div>
        </div>
      </Modal>
    );
  }

  return (
    <Modal onClose={onClose} title="Import transcript" ariaLabel="Import transcript" maxWidth="max-w-md">
      {error && <div className="mb-3 rounded-xl border border-error bg-error/10 p-3 text-xs text-error">{error}</div>}
      <form onSubmit={handleSubmit} className="space-y-4 text-sm">
        <Field label="Transcript file">
          <FileInput
            required
            accept=".txt,.md,.srt,.vtt,.json,.csv,text/*,application/json"
            onChange={(e) => {
              const selected = e.target.files?.[0] ?? null;
              setFile(selected);
              if (selected && !title.trim()) setTitle(selected.name.replace(/\.[^/.]+$/, "").replace(/[_-]/g, " "));
            }}
          />
        </Field>
        <Field label="Title (optional)">
          <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Weekly sync with Dana" />
        </Field>
        <Field label={needsStartTime ? "Start time (required for this file)" : "Start time (optional)"}>
          <Input type="datetime-local" value={startTime} onChange={(e) => setStartTime(e.target.value)} required={needsStartTime} />
        </Field>
        <Field label="This is me (optional)">
          <Input value={meAliases} onChange={(e) => setMeAliases(e.target.value)} placeholder="Names I appear under, comma separated" />
        </Field>
        <p className="text-xs text-base-content/70">
          Imported meetings have no recording, so there is no audio player and speakers are matched by name only.
        </p>
        <div className="flex justify-end gap-3 pt-2">
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary" disabled={submitting || !file}>
            {submitting ? "Importing…" : "Import"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
