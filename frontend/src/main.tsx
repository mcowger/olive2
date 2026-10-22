import React, { Fragment, StrictMode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import type { ReactElement, ReactNode } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";
import { Check, CheckCircle2, ChevronDown, ChevronUp, ClipboardList, Download, Ellipsis, Eye, FileText, Info, Link2, Mic, Pause, Pencil, Play, RefreshCw, RotateCw, Save, Scissors, Settings, Sparkles, Star, Trash2, VolumeX, X, XCircle, Zap } from "lucide-react";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessageLike
} from "@assistant-ui/react";
import "./styles.css";
import { Combobox } from "./components/Combobox";
import { ConfirmHostRoot, Modal as UiModal, ThemeToggle, ToastProvider, confirmAction, notify } from "./components/ui";
import { SpeakerCombobox, SpeakerStatusBadges } from "./components/SpeakerCombobox";
import {
  ENGINE_OPTIONS,
  LOG_LEVEL_OPTIONS,
  PLAUD_RETRANSCRIBE_OPTIONS,
  TRANSCRIBE_PROVIDER_OPTIONS,
  TRANSCRIPT_PROVIDER_FILTER_OPTIONS,
  ThinkingLabel,
  llmProviderOptions,
  logCategoryOptions,
  modelOptions,
  templateOptions,
  thinkingLevelOptions,
} from "./components/OptionComboboxes";
import type {
  Artifact,
  BackupInfo,
  ChatMessage,
  LlmModelCatalogItem,
  LlmProviderSummary,
  LogItem,
  LogLevel,
  LogListResponse,
  MeetingDetailAggregate,
  MeetingListItem,
  MeetingSummaryItem,
  Recording,
  RestoreResult,
  Speaker,
  StageRunRow,
  SummaryGenerationStatus,
  SpeakerReidentificationStatus,
  MeetingUsageResponse,
  RapidEnrollAssignResponse,
  RapidEnrollCandidateGroup,
  Template,
  TranscriptionProgressUpdate,
  UsageSummary
} from "./api/types.ts";

interface MeetingsResponse {
  meetings: MeetingListItem[];
  pagination: {
    limit: number;
    offset: number;
    total: number;
    hasMore?: boolean;
    nextOffset?: number | null;
  };
}

const MEETINGS_PAGE_SIZE = 100;

type MeetingDetailResponse = MeetingDetailAggregate;

interface SpeakerWithStats extends Speaker {
  meetingCount: number;
}

// Unidentified diarization labels ("Speaker 1", "S1", "Unknown", ...) are
// transcript labels, not people: never list them in People or as assignable
// persons. Mirrors backend olive/speaker_labels.py (defense in depth for
// speaker rows persisted before that rule existed).
const GENERIC_SPEAKER_RE = /^(?:(?:speaker|speakers)(?:[\s_-]*#?(?:\d+|[a-z]{1,2}))?|(?:spk|participant|person|guest|unknown|uu)(?:[\s_-]*#?\d+)?|unknown[\s_-]+speaker(?:[\s_-]*#?\d+)?|s[\s_-]*\d+)$/i;
function isGenericSpeakerLabel(name: string | null | undefined): boolean {
  return !name?.trim() || GENERIC_SPEAKER_RE.test(name.trim());
}

interface PlaudStatus {
  connected: boolean;
  lastPollAt: number | null;
  pcsPending: number;
}

function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short"
  }).format(new Date(timestamp));
}

function toDatetimeLocalValue(timestamp: number): string {
  const d = new Date(timestamp);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor((ms / 1000) % 60);
  const minutes = Math.floor((ms / (1000 * 60)) % 60);
  const hours = Math.floor(ms / (1000 * 60 * 60));

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m ${seconds}s`;
}

function formatTimeOffset(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

marked.setOptions({
  gfm: true,
  breaks: true
});

function MarkdownRenderer({ content, className = "" }: { content: string; className?: string }): ReactElement {
  const renderedHtml = useMemo(() => {
    if (!content) return "";
    try {
      const raw = marked.parse(content) as string;
      return DOMPurify.sanitize(raw, {
        USE_PROFILES: { html: true }
      });
    } catch {
      return content;
    }
  }, [content]);

  return (
    <div
      className={`prose prose-invert prose-stone max-w-none text-base-content prose-headings:font-semibold prose-headings:tracking-tight prose-headings:text-base-content prose-h1:text-xl prose-h1:border-b prose-h1:border-base-300 prose-h1:pb-2 prose-h2:text-lg prose-h2:border-b prose-h2:border-base-300/60 prose-h2:pb-1.5 prose-h3:text-base prose-h4:text-sm prose-p:leading-relaxed prose-p:my-3 prose-ul:my-2 prose-ul:list-disc prose-ol:my-2 prose-ol:list-decimal prose-li:my-1 prose-a:text-primary hover:prose-a:text-primary prose-strong:text-base-content prose-blockquote:border-l-2 prose-blockquote:border-primary/40 prose-blockquote:bg-base-200/60 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:rounded-r prose-blockquote:text-base-content prose-blockquote:not-italic prose-code:text-primary prose-code:bg-base-300/90 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:font-mono prose-code:text-xs prose-code:before:content-none prose-code:after:content-none prose-pre:bg-base-100 prose-pre:border prose-pre:border-base-300 prose-pre:rounded-xl prose-table:my-4 prose-table:w-full prose-table:text-left prose-th:border-b prose-th:border-base-300 prose-th:py-2 prose-th:px-3 prose-th:text-base-content prose-th:font-semibold prose-td:border-b prose-td:border-base-300 prose-td:py-2 prose-td:px-3 prose-hr:border-base-300 ${className}`}
      dangerouslySetInnerHTML={{ __html: renderedHtml }}
    />
  );
}

function chatMessageContent(message: ChatMessage): ThreadMessageLike {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: new Date(message.createdAt)
  };
}

function chatTextFromMessage(message: { content: unknown }): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return "";
  return message.content
    .filter((part): part is { type: "text"; text: string } =>
      typeof part === "object" && part !== null && "type" in part && part.type === "text" && "text" in part && typeof part.text === "string"
    )
    .map((part) => part.text)
    .join("");
}

function parseChatStreamBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  let data = "";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return data ? { event, data } : null;
}

function ChatTitleReporter({ title, onChatTitle }: { title: string; onChatTitle: (title: string) => void }): ReactElement | null {
  useEffect(() => {
    onChatTitle(title);
  }, [title, onChatTitle]);
  return null;
}

function MeetingChatPanel({ meetingId, meetingTitle, onClose }: { meetingId: string; meetingTitle: string; onClose: () => void }): ReactElement {
  const [initialMessages, setInitialMessages] = useState<readonly ThreadMessageLike[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    void fetch(`/api/meetings/${meetingId}/chat/messages`)
      .then(async (response) => {
        if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `HTTP ${response.status}`);
        return response.json() as Promise<{ messages: ChatMessage[] }>;
      })
      .then((data) => {
        if (!cancelled) setInitialMessages(data.messages.map(chatMessageContent));
      })
      .catch((error) => {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  return (
    <aside className="flex h-full w-full max-w-xl flex-col bg-base-100" role="dialog" aria-modal="true" aria-label={`Chat with ${meetingTitle}`}>
      <div className="flex items-center justify-between border-b border-base-300 bg-base-200 px-4 py-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wider text-primary">Chat with Meeting</p>
          <h2 className="truncate text-sm font-medium">{meetingTitle}</h2>
        </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={async () => {
                if (!(await confirmAction({ title: "Clear chat history", message: "Clear this meeting's chat history?", confirmLabel: "Clear", danger: true }))) return;
                void fetch(`/api/meetings/${meetingId}/chat/messages`, { method: "DELETE" }).then((response) => {
                  if (!response.ok) throw new Error("Failed to clear chat history");
                  onClose();
                }).catch((error) => notify("error", error instanceof Error ? error.message : String(error)));
              }}
              className="btn btn-ghost btn-sm min-h-[44px]"
            >
              Clear
            </button>
            <button type="button" onClick={onClose} className="btn btn-ghost inline-flex min-h-[44px] min-w-[44px] items-center justify-center text-lg leading-none hover:text-white" aria-label="Close chat"><X className="h-4 w-4 shrink-0" aria-hidden="true" /></button>
          </div>
        </div>

        {loading ? <div className="flex flex-1 items-center justify-center text-sm text-base-content/50">Loading conversation…</div> : loadError ? <div className="m-4 alert alert-soft alert-error p-3 text-sm">{loadError}</div> : initialMessages ? (
          <MeetingChatRuntime
            meetingId={meetingId}
            initialMessages={initialMessages}
          />
        ) : null}
      </aside>
  );
}

function MeetingChatRuntime({ meetingId, initialMessages }: { meetingId: string; initialMessages: readonly ThreadMessageLike[] }): ReactElement {
  const adapter: ChatModelAdapter = useMemo(() => ({
    async *run({ messages, abortSignal }) {
      const latest = messages[messages.length - 1];
      const content = latest ? chatTextFromMessage(latest) : "";
      const response = await fetch(`/api/meetings/${meetingId}/chat/messages`, {
        method: "POST",
        headers: { "content-type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify({ content }),
        signal: abortSignal
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({})) as { error?: string };
        throw new Error(error.error || `HTTP ${response.status}`);
      }
      if (!response.body) throw new Error("Chat response did not include a stream");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.replace(/\r\n/g, "\n").split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          const parsed = parseChatStreamBlock(block);
          if (!parsed) continue;
          const data = JSON.parse(parsed.data) as { delta?: string; error?: string };
          if (parsed.event === "error") throw new Error(data.error || "Chat request failed");
          if (parsed.event === "delta" && data.delta) {
            fullText += data.delta;
            yield { content: [{ type: "text", text: fullText }] };
          }
        }
      }
    }
  }), [meetingId]);

  const runtime = useLocalRuntime(adapter, { initialMessages });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
        <ThreadPrimitive.Viewport autoScroll className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <ThreadPrimitive.Messages components={{
            UserMessage: () => (
              <MessagePrimitive.Root className="mb-4 flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2.5 text-sm text-primary-content">
                  <MessagePrimitive.Content />
                </div>
              </MessagePrimitive.Root>
            ),
            AssistantMessage: () => (
              <MessagePrimitive.Root className="mb-4 flex justify-start">
                <div className="max-w-[90%] rounded-2xl rounded-bl-md border border-base-300 bg-base-200 px-3.5 py-2.5 text-sm text-base-content prose prose-invert prose-sm max-w-none">
                  <MessagePrimitive.Content />
                </div>
              </MessagePrimitive.Root>
            )
          }} />
        </ThreadPrimitive.Viewport>
        <div className="border-t border-base-300 bg-base-200/80 pb-safe">
          <ComposerPrimitive.Root className="p-3">
            <ComposerPrimitive.Input placeholder="Ask about this meeting…" className="min-h-20 w-full resize-none rounded-xl border border-base-300 bg-base-100 px-3 py-2 text-base text-base-content outline-none placeholder:text-base-content/50 focus:border-primary sm:text-sm" />
            <div className="mt-2 flex justify-end">
              <ComposerPrimitive.Send className="btn btn-primary min-h-[44px] text-sm disabled:cursor-not-allowed">Send</ComposerPrimitive.Send>
            </div>
          </ComposerPrimitive.Root>
        </div>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  );
}

type TabKey = "meetings" | "speakers" | "templates" | "logs" | "backup" | "usage";

function parseLocation(pathname: string): { tab: TabKey; meetingId: string | null } {
  const clean = pathname.replace(/^\/+|\/+$/g, "");
  if (!clean || clean === "meetings") {
    return { tab: "meetings", meetingId: null };
  }

  if (clean.startsWith("meetings/")) {
    const id = clean.slice("meetings/".length).trim();
    return { tab: "meetings", meetingId: id || null };
  }

  if (clean === "people" || clean === "speakers") {
    return { tab: "speakers", meetingId: null };
  }

  if (clean === "templates") {
    return { tab: "templates", meetingId: null };
  }

  if (clean === "logs") {
    return { tab: "logs", meetingId: null };
  }

  if (clean === "backup" || clean === "backups") {
    return { tab: "backup", meetingId: null };
  }

  if (clean === "usage") {
    return { tab: "usage", meetingId: null };
  }

  return { tab: "meetings", meetingId: null };
}

const NAV_TABS: { key: TabKey; path: string; label: string }[] = [
  { key: "meetings", path: "/meetings", label: "Meetings" },
  { key: "speakers", path: "/people", label: "People" },
  { key: "templates", path: "/templates", label: "Templates" },
  { key: "logs", path: "/logs", label: "Logs" },
  { key: "backup", path: "/backup", label: "Backup" },
  { key: "usage", path: "/usage", label: "Usage" },
];

const TAB_ICON_CONTENT: Record<TabKey, ReactNode> = {
  meetings: (
    <>
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <path d="M16 2v4M8 2v4M3 10h18" />
    </>
  ),
  speakers: (
    <>
      <circle cx="9" cy="7" r="4" />
      <path d="M2 21v-2a5 5 0 0 1 5-5h4a5 5 0 0 1 5 5v2" />
    </>
  ),
  templates: (
    <>
      <rect x="4" y="3" width="16" height="18" rx="2" />
      <path d="M8 8h8M8 12h8M8 16h5" />
    </>
  ),
  logs: <path d="M4 6h16M4 12h16M4 18h10" />,
  backup: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </>
  ),
  usage: <path d="M4 20V10M10 20V4M16 20v-7M2 20h20" />,
};

function TabIcon({ name }: { name: TabKey }): ReactElement {
  return (
    <svg className="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {TAB_ICON_CONTENT[name]}
    </svg>
  );
}

export function App(): ReactElement {
  const initialRoute = parseLocation(typeof window !== "undefined" ? window.location.pathname : "/");
  const [activeTab, setActiveTab] = useState<TabKey>(initialRoute.tab);
  const [selectedMeetingId, setSelectedMeetingId] = useState<string | null>(initialRoute.meetingId);

  // Global speakers cache
  const [knownSpeakers, setKnownSpeakers] = useState<SpeakerWithStats[]>([]);

  // Plaud status
  const [plaudStatus, setPlaudStatus] = useState<PlaudStatus | null>(null);
  const [showPlaudModal, setShowPlaudModal] = useState(false);

  // LLM status & settings
  const [llmConfig, setLlmConfig] = useState<{ defaultProvider: string; defaultModel: string } | null>(null);
  const [showLlmModal, setShowLlmModal] = useState(false);

  // Changelog
  const [showChangelogModal, setShowChangelogModal] = useState(false);

  // Meeting chat drawer (App-level so it overlays nav + content via drawer-side)
  const [chatOpen, setChatOpen] = useState(false);
  const [chatTitle, setChatTitle] = useState("");
  const [showMoreNav, setShowMoreNav] = useState(false);

  const isTabActive = (key: TabKey) => (key === "meetings" ? activeTab === "meetings" && !selectedMeetingId : activeTab === key);

  const navigate = (path: string) => {
    const route = parseLocation(path);
    setActiveTab(route.tab);
    setSelectedMeetingId(route.meetingId);
    setChatOpen(false);
    setShowMoreNav(false);
    if (typeof window !== "undefined" && window.location.pathname !== path) {
      window.history.pushState({}, "", path);
    }
  };

  useEffect(() => {
    const handlePopState = () => {
      const route = parseLocation(window.location.pathname);
      setActiveTab(route.tab);
      setSelectedMeetingId(route.meetingId);
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const refreshSpeakers = async () => {
    try {
      const res = await fetch("/api/speakers");
      if (res.ok) {
        const data = (await res.json()) as { speakers: SpeakerWithStats[] };
        setKnownSpeakers(data.speakers.filter((s) => !isGenericSpeakerLabel(s.name)));
      }
    } catch {
      // ignore
    }
  };

  const refreshPlaudStatus = async () => {
    try {
      const res = await fetch("/api/plaud/status");
      if (res.ok) {
        const data = (await res.json()) as PlaudStatus;
        setPlaudStatus(data);
      }
    } catch {
      // ignore
    }
  };

  const refreshLlmConfig = async () => {
    try {
      const res = await fetch("/api/llm/config");
      if (res.ok) {
        const data = await res.json();
        setLlmConfig(data);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    void refreshSpeakers();
    void refreshPlaudStatus();
    void refreshLlmConfig();
  }, []);

  return (
    <div className="drawer drawer-end w-full">
      <input
        id="olive-chat-drawer"
        type="checkbox"
        className="drawer-toggle"
        checked={chatOpen}
        onChange={(e) => setChatOpen(e.target.checked)}
      />
      <div className="drawer-content min-h-screen bg-base-100 text-base-content selection:bg-primary selection:text-primary-content w-full overflow-x-clip">
      <nav className="border-b border-base-300 bg-base-200/80 backdrop-blur sticky top-0 z-30 px-3 sm:px-6 py-2.5 sm:py-3.5 w-full">
        <div className="mx-auto flex max-w-6xl flex-nowrap items-center justify-between gap-2 sm:gap-4">
          <div className="flex min-w-0 items-center gap-3 sm:gap-4 lg:gap-6">
            <a
              href="/meetings"
              onClick={(e) => {
                e.preventDefault();
                navigate("/meetings");
              }}
              className="flex shrink-0 items-center gap-2 cursor-pointer hover:opacity-90 transition"
            >
              <span className="flex h-7 w-7 sm:h-8 sm:w-8 items-center justify-center rounded-lg bg-primary font-bold text-primary-content text-sm sm:text-base">
                O
              </span>
              <span className="text-lg sm:text-xl font-semibold tracking-tight">Olive</span>
            </a>
            <div className="hidden sm:flex min-w-0 flex-nowrap items-center gap-0.5 overflow-x-auto rounded-lg bg-base-200 p-1 text-xs lg:text-sm font-medium">
              {NAV_TABS.map((tab) => (
                <a
                  key={tab.key}
                  href={tab.path}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(tab.path);
                    if (tab.key === "speakers") void refreshSpeakers();
                  }}
                  className={`shrink-0 whitespace-nowrap rounded-field px-2 lg:px-3 py-1.5 transition ${
                    isTabActive(tab.key) ? "bg-base-100 text-primary shadow" : "text-base-content/60 hover:text-base-content"
                  }`}
                >
                  {tab.key === "speakers" ? `People (${knownSpeakers.length})` : tab.label}
                </a>
              ))}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setShowChangelogModal(true)}
              title="What's new in Olive"
              className="btn btn-ghost btn-sm shrink-0"
            >
              <span aria-hidden="true" className="inline-flex"><Sparkles className="h-3.5 w-3.5" aria-hidden="true" /></span>
              <span className="hidden md:inline">What's New</span>
            </button>
            <button
              type="button"
              onClick={() => setShowLlmModal(true)}
              className="btn btn-ghost btn-sm max-w-[110px] lg:max-w-[150px] xl:max-w-none"
            >
              <span className="h-2 w-2 rounded-full bg-accent shrink-0" />
              <span className="truncate">AI: {llmConfig ? `${llmConfig.defaultProvider}/${llmConfig.defaultModel}` : "Config"}</span>
            </button>
            <button
              type="button"
              onClick={() => {
                setShowPlaudModal(true);
                void refreshPlaudStatus();
              }}
              className={`btn btn-sm shrink-0 ${
                plaudStatus?.connected
                  ? "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20"
                  : "border-warning/30 bg-warning/10 text-warning hover:bg-warning/20"
              }`}
            >
              <span className={`h-2 w-2 rounded-full shrink-0 ${plaudStatus?.connected ? "bg-primary" : "bg-warning"}`} />
              <span>{plaudStatus?.connected ? "Plaud" : "Connect"}</span>
            </button>
            <ThemeToggle />
          </div>
        </div>
      </nav>

      <nav
        aria-label="Primary"
        className="dock dock-md z-30 pb-safe sm:hidden"
      >
        {(NAV_TABS.filter((t) => t.key === "meetings" || t.key === "speakers" || t.key === "templates" || t.key === "usage")).map((tab) => {
          const active = isTabActive(tab.key);
          return (
            <button
              key={tab.key}
              type="button"
              aria-current={active ? "page" : undefined}
              onClick={() => {
                setShowMoreNav(false);
                navigate(tab.path);
                if (tab.key === "speakers") void refreshSpeakers();
              }}
              className={active ? "dock-active" : ""}
            >
              <TabIcon name={tab.key} />
              <span className="dock-label">{tab.key === "speakers" ? `People${knownSpeakers.length > 0 ? ` (${knownSpeakers.length})` : ""}` : tab.label}</span>
            </button>
          );
        })}
        <button
          type="button"
          aria-expanded={showMoreNav}
          aria-label="More sections"
          onClick={() => setShowMoreNav((v) => !v)}
          className={activeTab === "logs" || activeTab === "backup" ? "dock-active" : ""}
        >
          <Ellipsis className="h-5 w-5" aria-hidden="true" />
          <span className="dock-label">More</span>
        </button>
      </nav>
      {showMoreNav && (
        <div className="fixed inset-x-4 bottom-20 z-40 sm:hidden">
          <ul className="menu rounded-box border border-base-300 bg-base-100 p-2 shadow-lg">
            {(NAV_TABS.filter((t) => t.key === "logs" || t.key === "backup")).map((tab) => (
              <li key={tab.key}>
                <a
                  href={tab.path}
                  aria-current={isTabActive(tab.key) ? "page" : undefined}
                  onClick={(e) => {
                    e.preventDefault();
                    setShowMoreNav(false);
                    navigate(tab.path);
                  }}
                  className={isTabActive(tab.key) ? "menu-active" : ""}
                >
                  <TabIcon name={tab.key} />
                  {tab.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}

      <main className="mx-auto max-w-6xl px-3.5 sm:px-6 pt-4 sm:pt-8 pb-[calc(6rem+env(safe-area-inset-bottom))] sm:pb-8 w-full overflow-x-clip min-w-0">
        {selectedMeetingId ? (
          <MeetingDetailView
            meetingId={selectedMeetingId}
            knownSpeakers={knownSpeakers}
            onBack={() => navigate("/meetings")}
            onDeleted={() => navigate("/meetings")}
            onSpeakerUpdated={() => void refreshSpeakers()}
            onChatOpenChange={setChatOpen}
            onChatTitle={setChatTitle}
          />
        ) : activeTab === "meetings" ? (
          <MeetingsListView onSelectMeeting={(id) => navigate(`/meetings/${id}`)} />
        ) : activeTab === "speakers" ? (
          <SpeakersListView
            speakers={knownSpeakers}
            onRefresh={() => void refreshSpeakers()}
            onSelectMeeting={(id) => navigate(`/meetings/${id}`)}
          />
        ) : activeTab === "templates" ? (
          <TemplatesListView />
        ) : activeTab === "logs" ? (
          <LogsView
            onSelectMeeting={(id) => navigate(`/meetings/${id}`)}
          />
        ) : activeTab === "usage" ? (
          <UsageView onSelectMeeting={(id) => navigate(`/meetings/${id}`)} />
        ) : (
          <BackupRestoreView
            onRestoreComplete={() => {
              void refreshSpeakers();
              void refreshLlmConfig();
            }}
          />
        )}
      </main>

      {/* Changelog Modal */}
      {showChangelogModal && (
        <ChangelogModal onClose={() => setShowChangelogModal(false)} />
      )}

      {/* Plaud Auth & Sync Modal */}
      {showPlaudModal && (
        <PlaudModal
          status={plaudStatus}
          onClose={() => setShowPlaudModal(false)}
          onRefreshStatus={() => void refreshPlaudStatus()}
        />
      )}

      {/* LLM Provider & Model Settings Modal */}
      {showLlmModal && (
        <LlmSettingsModal
          onClose={() => setShowLlmModal(false)}
          onSaved={() => {
            setShowLlmModal(false);
            void refreshLlmConfig();
          }}
        />
      )}
      </div>
      <div className="drawer-side z-50">
        <label htmlFor="olive-chat-drawer" aria-label="Close chat" className="drawer-overlay" />
        <div className="min-h-full w-full max-w-xl bg-base-100 shadow-2xl">
          {chatOpen && selectedMeetingId && (
            <MeetingChatPanel
              meetingId={selectedMeetingId}
              meetingTitle={chatTitle || "Meeting"}
              onClose={() => setChatOpen(false)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function Modal({
  onClose,
  title,
  ariaLabel,
  maxWidth = "max-w-lg",
  children
}: {
  onClose: () => void;
  title: ReactNode;
  ariaLabel: string;
  maxWidth?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <UiModal onClose={onClose} title={title} ariaLabel={ariaLabel} maxWidth={maxWidth}>
      {children}
    </UiModal>
  );
}

function ChangelogModal({ onClose }: { onClose: () => void }): ReactElement {
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [version, setVersion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetch("/api/changelog")
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<{ version: string; markdown: string }>;
      })
      .then((data) => {
        if (cancelled) return;
        setVersion(data.version);
        setMarkdown(data.markdown);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Modal
      onClose={onClose}
      title={version ? `What's New — ${version}` : "What's New"}
      ariaLabel="Olive changelog"
      maxWidth="max-w-2xl"
    >
      {error ? (
        <div className="alert alert-soft alert-error p-3 text-sm text-error">
          Failed to load changelog: {error}
        </div>
      ) : markdown === null ? (
        <div className="py-8 text-center text-sm text-base-content/50">Loading changelog…</div>
      ) : markdown ? (
        <div className="max-h-[60vh] overflow-y-auto pr-1">
          <MarkdownRenderer content={markdown} />
        </div>
      ) : (
        <div className="py-8 text-center text-sm text-base-content/50">No changelog available.</div>
      )}
    </Modal>
  );
}

function PlaudModal({
  status,
  onClose,
  onRefreshStatus
}: {
  status: PlaudStatus | null;
  onClose: () => void;
  onRefreshStatus: () => void;
}): ReactElement {
  const [authStep, setAuthStep] = useState<"idle" | "authorizing" | "success" | "error">("idle");
  const [authUrl, setAuthUrl] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [callbackInput, setCallbackInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);

  const handleStartAuth = async () => {
    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/plaud/auth/start", { method: "POST" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { authUrl: string; sessionId: string };
      setAuthUrl(data.authUrl);
      setSessionId(data.sessionId);
      setAuthStep("authorizing");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleCompleteAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !callbackInput.trim()) {
      notify("warning", "Please paste the callback URL or code from your browser");
      return;
    }

    setLoading(true);
    setErrorMessage(null);
    try {
      const res = await fetch("/api/plaud/auth/complete", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          sessionId,
          pastedUrlOrCode: callbackInput.trim()
        })
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      setAuthStep("success");
      onRefreshStatus();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerSync = async () => {
    setSyncing(true);
    setSyncMessage(null);
    try {
      const res = await fetch("/api/plaud/sync/trigger", { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      setSyncMessage(
        data.alreadyRunning
          ? "A Plaud sync is already running in the background."
          : "Plaud sync started in the background. New recordings will appear as they finish."
      );
      onRefreshStatus();
    } catch (err) {
      setSyncMessage(`Sync failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSyncing(false);
    }
  };

  return (
    <Modal onClose={onClose} title={<span className="flex flex-wrap items-center gap-2"><span>Plaud Integration</span><span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${status?.connected ? "bg-primary/10 text-primary" : "bg-warning/10 text-warning"}`}>{status?.connected ? "Connected" : "Disconnected"}</span></span>} ariaLabel="Plaud integration" maxWidth="max-w-lg">
      {errorMessage && (
        <div className="rounded-xl border border-error bg-error/10 p-4 text-xs text-error">
          {errorMessage}
        </div>
      )}

      {syncMessage && (
        <div className="rounded-xl border border-primary bg-primary/10 p-4 text-xs text-primary">
          {syncMessage}
        </div>
      )}

      {status?.connected && authStep === "idle" ? (
        <div className="space-y-4">
          <div className="rounded-xl border border-base-300 bg-base-100/60 p-4 space-y-2 text-xs text-base-content">
            <div className="flex justify-between">
              <span className="text-base-content/70">Connection Status:</span>
              <span className="font-semibold text-primary">Active (OAuth tokens valid)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-base-content/70">Last Synced:</span>
              <span>{status.lastPollAt ? formatDate(status.lastPollAt) : "Never"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-base-content/70">Pending PCS Items:</span>
              <span>{status.pcsPending}</span>
            </div>
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              disabled={syncing}
              onClick={() => void handleTriggerSync()}
              className="btn btn-primary flex-1 min-h-11"
            >
              {syncing ? "Syncing with Plaud…" : "Sync Now"}
            </button>
            <button
              type="button"
              onClick={() => void handleStartAuth()}
              className="btn bg-base-300 min-h-11"
            >
              Re-authenticate
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-5">
          {authStep === "idle" && (
            <div className="space-y-4 text-sm text-base-content">
              <p>
                Connect your Plaud account using OAuth 2.0 PKCE to automatically ingest recordings, transcripts, and summaries into Olive.
              </p>
              <button
                type="button"
                disabled={loading}
                onClick={() => void handleStartAuth()}
                className="btn btn-primary w-full py-2.5 min-h-11"
              >
                {loading ? "Generating login link…" : "Start Plaud OAuth Login"}
              </button>
            </div>
          )}

          {authStep === "authorizing" && authUrl && (
            <form onSubmit={handleCompleteAuth} className="space-y-4 text-sm">
              <div className="rounded-xl border border-base-300 bg-base-100/80 p-4 space-y-3">
                <p className="font-semibold text-base-content">Step 1: Authorize with Plaud</p>
                <p className="text-xs text-base-content/70 leading-relaxed">
                  Click the button below to open the official Plaud login in a new tab. Log in and grant permission.
                </p>
                <a
                  href={authUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-primary btn-sm min-h-11"
                >
                  Open Plaud Login Page ↗
                </a>
              </div>

              <div className="rounded-xl border border-base-300 bg-base-100/80 p-4 space-y-3">
                <p className="font-semibold text-base-content">Step 2: Paste the Callback URL</p>
                <p className="text-xs text-base-content/70 leading-relaxed">
                  After authorizing, Plaud will redirect your browser to a callback address starting with{" "}
                  <code className="rounded bg-base-200 px-1 py-0.5 text-primary">http://localhost:8199/auth/callback?code=...</code>{" "}
                  (your browser might show "Site can't be reached"—this is normal). Copy the <strong>entire URL</strong> from your address bar and paste it below:
                </p>
                <input
                  type="text"
                  required
                  placeholder="http://localhost:8199/auth/callback?code=...&state=..."
                  value={callbackInput}
                  onChange={(e) => setCallbackInput(e.target.value)}
                  className="input input-md w-full"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setAuthStep("idle")}
                  className="rounded-lg px-4 py-2 text-sm text-base-content/70 hover:text-base-content"
                >
                  Back
                </button>
                <button
                  type="submit"
                  disabled={loading || !callbackInput.trim()}
                  className="btn btn-primary min-h-11"
                >
                  {loading ? "Verifying…" : "Complete Connection"}
                </button>
              </div>
            </form>
          )}

          {authStep === "success" && (
            <div className="space-y-4 py-4 text-center">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-primary/20 text-2xl text-primary">
                <Check className="h-4 w-4 shrink-0" aria-hidden="true" />
              </div>
              <h4 className="text-lg font-bold text-base-content">Plaud Connected Successfully!</h4>
              <p className="text-xs text-base-content/70">
                Olive is now authenticated with your Plaud account. Tokens will refresh automatically.
              </p>
              <button
                type="button"
                onClick={() => {
                  setAuthStep("idle");
                  onClose();
                }}
                className="btn btn-primary min-h-11"
              >
                Done
              </button>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}

function MeetingsListView({ onSelectMeeting }: { onSelectMeeting: (id: string) => void }): ReactElement {
  const [meetings, setMeetings] = useState<MeetingListItem[]>([]);
  const [activeProgressMap, setActiveProgressMap] = useState<Record<string, TranscriptionProgressUpdate>>({});
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [nextOffset, setNextOffset] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showBulkModal, setShowBulkModal] = useState(false);
  const [mergingMeetings, setMergingMeetings] = useState(false);
  const [bulkNotice, setBulkNotice] = useState<string | null>(null);
  const [selectMode, setSelectMode] = useState(false);
  const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);
  const meetingsRequestId = useRef(0);
  const meetingsRef = useRef<MeetingListItem[]>([]);
  const debouncedSearchRef = useRef("");
  const paginationRef = useRef<{ hasMore: boolean; nextOffset: number | null }>({ hasMore: false, nextOffset: null });

  useEffect(() => {
    meetingsRef.current = meetings;
  }, [meetings]);
  useEffect(() => {
    debouncedSearchRef.current = debouncedSearch;
  }, [debouncedSearch]);
  useEffect(() => {
    paginationRef.current = { hasMore, nextOffset };
  }, [hasMore, nextOffset]);

  const buildMeetingsUrl = (query: string, offset: number) => {
    const params = new URLSearchParams({
      limit: String(MEETINGS_PAGE_SIZE),
      offset: String(offset),
      // Full detail: the list view needs status, audioStatus, and artifact ids.
      includeDetails: "true",
    });
    const q = query.trim();
    if (q) params.set("search", q);
    return `/api/meetings?${params.toString()}`;
  };

  const applyPage = (data: MeetingsResponse, mode: "reset" | "append") => {
    const incoming = data.meetings || [];
    const pagination = data.pagination;
    const totalCount = pagination ? pagination.total : incoming.length;
    if (mode === "reset") {
      setMeetings(incoming);
      setTotal(totalCount);
      const more = pagination?.hasMore ?? incoming.length < totalCount;
      setHasMore(more);
      setNextOffset(pagination?.nextOffset ?? (more ? incoming.length : null));
    } else {
      const seen = new Set(meetingsRef.current.map((m) => m.id));
      const deduped = incoming.filter((m) => !seen.has(m.id));
      const merged = [...meetingsRef.current, ...deduped];
      setMeetings(merged);
      setTotal(totalCount);
      const more = pagination?.hasMore ?? merged.length < totalCount;
      setHasMore(more);
      setNextOffset(pagination?.nextOffset ?? (more ? merged.length : null));
    }
  };

  const fetchMeetings = async (opts?: { silent?: boolean }) => {
    const requestId = ++meetingsRequestId.current;
    if (!opts?.silent) {
      setLoading(true);
    }
    setLoadingMore(false);
    setError(null);
    try {
      const query = debouncedSearchRef.current;
      const [res, activeRes] = await Promise.all([
        fetch(buildMeetingsUrl(query, 0)),
        fetch("/api/transcriptions/active").catch(() => null)
      ]);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as MeetingsResponse;
      if (requestId !== meetingsRequestId.current) return;
      applyPage(data, "reset");
      if (activeRes && activeRes.ok) {
        const activeData = await activeRes.json();
        setActiveProgressMap(activeData.active || {});
      }
    } catch (err) {
      if (requestId !== meetingsRequestId.current) return;
      setError(err instanceof Error ? err.message : "Failed to load meetings");
    } finally {
      if (requestId === meetingsRequestId.current) {
        setLoading(false);
      }
    }
  };

  const loadMore = async () => {
    if (loadingMore || loading || !paginationRef.current.hasMore) return;
    const offset = paginationRef.current.nextOffset ?? meetingsRef.current.length;
    if (offset == null) return;
    const requestId = ++meetingsRequestId.current;
    setLoadingMore(true);
    try {
      const res = await fetch(buildMeetingsUrl(debouncedSearchRef.current, offset));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as MeetingsResponse;
      if (requestId !== meetingsRequestId.current) return;
      applyPage(data, "append");
    } catch (err) {
      if (requestId !== meetingsRequestId.current) return;
      setError(err instanceof Error ? err.message : "Failed to load more meetings");
    } finally {
      if (requestId === meetingsRequestId.current) {
        setLoadingMore(false);
      }
    }
  };

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    void fetchMeetings();
  }, [debouncedSearch]);

  useEffect(() => {
    if (!hasMore || loading || loadingMore) return;
    const el = loadMoreSentinelRef.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) void loadMore();
      },
      { rootMargin: "600px 0px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loading, loadingMore, nextOffset, debouncedSearch, meetings.length]);

  const toggleSelectMeeting = (id: string) => {
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]));
  };

  const exitSelectMode = () => {
    setSelectMode(false);
    setSelectedIds([]);
  };

  const handleSelectVisible = () => {
    setSelectedIds(meetings.map((m) => m.id));
  };

  const handleClearSelection = () => {
    setSelectedIds([]);
  };

  const handleMergeSelected = async () => {
    if (mergingMeetings) return;
    const ordered = meetings
      .filter((m) => selectedIds.includes(m.id))
      .slice()
      .sort((a, b) => a.startTime - b.startTime);
    if (ordered.length < 2) return;
    const target = ordered[0];
    const titles = ordered.map((m) => `“${m.title}”`).join(", ");
    const others = ordered.length - 1;
    if (
      !(await confirmAction({
        title: `Merge ${ordered.length} meetings`,
        message: `Merge ${titles} into “${target.title}” in chronological order? Audio and transcripts are joined without re-transcription, and the other ${others} meeting${others === 1 ? "" : "s"} will be permanently deleted. Summaries are dropped because they only describe the separate clips.`,
        confirmLabel: "Merge & delete originals",
        danger: true
      }))
    ) {
      return;
    }
    setMergingMeetings(true);
    setError(null);
    try {
      const res = await fetch("/api/meetings/merge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ meetingIds: ordered.map((m) => m.id) })
      });
      const data = (await res.json().catch(() => ({}))) as { meeting?: { id?: string }; error?: string };
      if (!res.ok) throw new Error(data.error || `Failed to merge meetings (HTTP ${res.status})`);
      const mergedId = data.meeting?.id;
      setSelectedIds([]);
      setSelectMode(false);
      setMeetings((prev) => prev.filter((m) => m.id === mergedId || !ordered.some((o) => o.id === m.id)));
      setTotal((t) => Math.max(0, t - others));
      await fetchMeetings({ silent: true });
      if (mergedId) onSelectMeeting(mergedId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to merge meetings");
    } finally {
      setMergingMeetings(false);
    }
  };

  const handleDeleteMeeting = async (id: string, title: string) => {
    if (!(await confirmAction({ title: "Delete meeting", message: `Are you sure you want to delete meeting "${title}"? This will permanently remove its recordings, transcripts, and summaries.`, confirmLabel: "Delete", danger: true }))) {
      return;
    }
    try {
      const res = await fetch(`/api/meetings/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error || `Failed to delete meeting (HTTP ${res.status})`);
      }
      setMeetings((prev) => prev.filter((m) => m.id !== id));
      setTotal((t) => Math.max(0, t - 1));
      setSelectedIds((prev) => prev.filter((sid) => sid !== id));
      if (paginationRef.current.hasMore) {
        setNextOffset(Math.max(0, meetingsRef.current.length - 1));
      }
      setActiveProgressMap((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  // Poll active transcriptions every 2.5s if any meeting is processing.
  // Refresh the first page and merge into the loaded list so infinite
  // scroll position is preserved (the backend caps pages at 100).
  useEffect(() => {
    const hasProcessing = meetings.some((m) => m.status === "processing") || Object.keys(activeProgressMap).length > 0;
    if (!hasProcessing) return;

    const interval = setInterval(async () => {
      try {
        const [activeRes, meetingsRes] = await Promise.all([
          fetch("/api/transcriptions/active"),
          fetch(buildMeetingsUrl(debouncedSearchRef.current, 0))
        ]);
        if (activeRes.ok) {
          const activeData = await activeRes.json();
          setActiveProgressMap(activeData.active || {});
        }
        if (meetingsRes.ok) {
          const mData = (await meetingsRes.json()) as MeetingsResponse;
          const fresh = mData.meetings || [];
          const freshTotal = mData.pagination ? mData.pagination.total : fresh.length;
          const prev = meetingsRef.current;
          if (prev.length === 0) {
            applyPage(mData, "reset");
          } else {
            const freshById = new Map(fresh.map((m) => [m.id, m]));
            const prevIds = new Set(prev.map((m) => m.id));
            const newcomers = fresh.filter((m) => !prevIds.has(m.id));
            const combined = [...newcomers, ...prev.map((m) => freshById.get(m.id) ?? m)];
            setMeetings(combined);
            setTotal(freshTotal);
            const more = combined.length < freshTotal;
            setHasMore(more);
            setNextOffset(more ? combined.length : null);
          }
        }
      } catch {}
    }, 2500);

    return () => clearInterval(interval);
  }, [meetings, activeProgressMap, debouncedSearch]);

  return (
    <div className="space-y-5 sm:space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Meetings</h1>
          <p className="mt-1 text-sm text-base-content/70">Recorded and ingested audio archive</p>
        </div>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:gap-3">
          <input
            type="search"
            placeholder="Search meetings…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input input-md w-full sm:w-64"
          />
          <button
            type="button"
            onClick={() => {
              if (selectMode) {
                exitSelectMode();
              } else {
                setSelectMode(true);
              }
            }}
            aria-pressed={selectMode}
            className={`flex min-h-[44px] items-center justify-center rounded-lg border px-4 py-2 text-sm font-semibold transition shrink-0 ${
              selectMode
                ? "border-primary bg-primary/10 text-primary"
                : "border-base-300 text-base-content hover:border-base-300 hover:text-white"
            }`}
          >
            {selectMode ? "Done" : "Select"}
          </button>
          <button
            type="button"
            onClick={() => {
              setBulkNotice(null);
              setShowBulkModal(true);
            }}
            className={`btn min-h-[44px] flex-1 sm:flex-none gap-2 shrink-0 ${
              selectMode && selectedIds.length > 0
                ? "btn-primary"
                : "btn-primary btn-soft"
            }`}
            title="Re-transcribe many meetings with one engine"
          >
            <span><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /></span> {selectMode && selectedIds.length > 0 ? `Re-transcribe ${selectedIds.length}` : "Bulk re-transcribe"}
          </button>
          <button
            type="button"
            onClick={() => setShowUploadModal(true)}
            className="btn btn-primary flex min-h-[44px] flex-1 sm:flex-none items-center justify-center gap-2 shrink-0"
          >
            <span>+</span> Upload Audio
          </button>
        </div>
      </div>

      {bulkNotice && (
        <div className="rounded-xl border border-primary bg-primary/10 p-4 text-sm text-primary">{bulkNotice}</div>
      )}

      {selectMode && !loading && !error && meetings.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-xl border border-base-300 bg-base-200/60 px-3 py-2 text-xs text-base-content/70 sm:gap-3">
          <span className={`mr-auto ${selectedIds.length > 0 ? "font-semibold text-primary" : ""}`}>
            {selectedIds.length > 0 ? `${selectedIds.length} selected` : "Tap meetings to select"}
          </span>
          <button type="button" onClick={handleSelectVisible} className="btn min-h-[36px] hover:border-primary/60 hover:text-primary">
            Select visible ({meetings.length})
          </button>
          {selectedIds.length > 0 && (
            <button type="button" onClick={handleClearSelection} className="btn min-h-[36px] hover:border-error/60 hover:text-error">
              Clear
            </button>
          )}
          {selectedIds.length >= 2 && (
            <button
              type="button"
              onClick={() => void handleMergeSelected()}
              disabled={mergingMeetings}
              className="btn min-h-[36px] gap-1.5 border-primary/60 text-primary hover:bg-primary/10"
              title="Join the selected meetings (chronological) into one and delete the rest"
            >
              <Link2 className="h-4 w-4 shrink-0" aria-hidden="true" />
              {mergingMeetings ? "Merging…" : `Merge ${selectedIds.length}`}
            </button>
          )}
        </div>
      )}

      {loading && <p className="text-base-content/70">Loading meetings…</p>}
      {error && <div className="rounded-xl border border-error bg-error/10 p-4 text-error">{error}</div>}

      {!loading && !error && meetings.length === 0 && (
        <div className="rounded-2xl border border-dashed border-base-300 bg-base-200/40 py-16 text-center">
          <p className="text-base-content/70">No meetings found.</p>
        </div>
      )}

      {!loading && !error && meetings.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2">
          {meetings.map((meeting) => {
            const selected = selectedIds.includes(meeting.id);
            const progress = activeProgressMap[meeting.id];
            const audioPending = meeting.audioStatus === "retrieving" || meeting.audioStatus === "missing";
            return (
              <div
                key={meeting.id}
                onClick={() => (selectMode ? toggleSelectMeeting(meeting.id) : onSelectMeeting(meeting.id))}
                aria-selected={selectMode ? selected : undefined}
                className={`group flex h-full cursor-pointer flex-col gap-2.5 rounded-xl border p-4 sm:p-5 shadow-sm transition ${
                  selectMode && selected
                    ? "border-primary bg-primary/5"
                    : audioPending
                    ? "border-base-300 bg-base-200/60 opacity-60 saturate-50 hover:border-info/50 hover:opacity-80"
                    : "border-base-300 bg-base-200/90 hover:border-primary/60 hover:bg-base-200"
                }`}
              >
                <div className="flex items-start gap-2">
                  {selectMode && (
                    <span className="-ml-2 -mt-1.5 flex h-11 w-11 shrink-0 items-center justify-center">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleSelectMeeting(meeting.id)}
                        onClick={(e) => e.stopPropagation()}
                        className="h-5 w-5 accent-primary"
                        aria-label={`Select meeting ${meeting.title}`}
                      />
                    </span>
                  )}
                  <div className="min-w-0 flex-1">
                    <h2 className="line-clamp-2 break-words text-lg font-semibold leading-snug text-base-content group-hover:text-primary transition">
                      {meeting.title}
                    </h2>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-base-content/70">
                      <span>{formatDate(meeting.startTime)}</span>
                      <span>•</span>
                      <span>{formatDuration(meeting.endTime - meeting.startTime)}</span>
                      <span>•</span>
                      <span className="capitalize">{meeting.source}</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      void handleDeleteMeeting(meeting.id, meeting.title);
                    }}
                    className="btn btn-circle btn-ghost -mr-2 -mt-1.5 flex shrink-0 items-center justify-center text-base-content/50 hover:bg-error/10 hover:text-error min-h-11 min-w-11"
                    title="Delete meeting"
                    aria-label={`Delete meeting ${meeting.title}`}
                  >
                    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>

                <div className="mt-auto flex flex-wrap items-center gap-1.5">
                  {meeting.audioStatus === "retrieving" && !progress ? (
                    <span className="badge badge-info badge-soft gap-1.5 px-3 py-1 animate-pulse">
                      <span className="status status-info status-xs animate-ping" />
                      Retrieving audio…
                    </span>
                  ) : meeting.audioStatus === "missing" && !progress ? (
                    <span className="badge badge-error badge-soft gap-1.5 px-3 py-1">
                      Audio missing
                    </span>
                  ) : progress ? (
                    <div className="badge badge-warning badge-soft gap-2 px-3 py-1">
                      <span className="status status-warning status-xs animate-ping" />
                      <span className="font-semibold">{progress.stage} ({Math.round(progress.percent)}%)</span>
                    </div>
                  ) : (
                    <span
                      className={`badge badge-soft capitalize ${
                        meeting.status === "ready"
                          ? "badge-success"
                          : meeting.status === "processing"
                          ? "badge-warning animate-pulse"
                          : "badge-neutral"
                      }`}
                    >
                      {meeting.status}
                    </span>
                  )}
                  {meeting.primaryTranscriptArtifactId && (
                    <span className="rounded-full bg-base-300 px-2.5 py-1 text-xs text-base-content">
                      Transcript
                    </span>
                  )}
                  {meeting.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-full border border-base-300 bg-base-300/80 px-2 py-0.5 text-[11px] text-base-content"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {progress && (
                  <div className="space-y-1">
                    <div className="flex justify-between gap-2 text-[11px] text-base-content/70">
                      <span className="truncate max-w-[80%]">{progress.message}</span>
                      <span className="font-mono text-warning font-bold">{Math.round(progress.percent)}%</span>
                    </div>
                    <progress className="progress progress-warning w-full" value={Math.max(5, progress.percent)} max={100} aria-label="Meeting progress" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {!loading && !error && meetings.length > 0 && (
        <div className="flex flex-col items-center gap-3 pt-1">
          <p className="text-xs text-base-content/50" aria-live="polite">
            Showing {meetings.length} of {total} meeting{total === 1 ? "" : "s"}
          </p>
          {hasMore && !loadingMore && (
            <button
              type="button"
              onClick={() => void loadMore()}
              className="btn min-h-[44px] font-semibold hover:border-primary/60 hover:text-primary"
            >
              Load more
            </button>
          )}
          {loadingMore && <p className="text-xs text-base-content/50">Loading more meetings…</p>}
          {!hasMore && total > 0 && (
            <p className="text-xs text-base-content/50">You&apos;ve reached the end of the list.</p>
          )}
          <div ref={loadMoreSentinelRef} aria-hidden="true" className="h-1 w-full" />
        </div>
      )}

      {showBulkModal && (
        <BulkTranscribeModal
          selectedIds={selectedIds}
          listSearch={search}
          onClose={() => setShowBulkModal(false)}
          onQueued={(summary, queuedIds) => {
            setShowBulkModal(false);
            setBulkNotice(summary);
            if (queuedIds.length > 0) {
              setSelectedIds([]);
            }
            void fetchMeetings({ silent: true });
          }}
        />
      )}

      {showUploadModal && (
        <UploadAudioModal
          onClose={() => setShowUploadModal(false)}
          onUploaded={(meetingId) => {
            setShowUploadModal(false);
            onSelectMeeting(meetingId);
          }}
        />
      )}
    </div>
  );
}

interface BulkQueueResult {
  bulkId: string;
  provider: string;
  total: number;
  queued: number;
  alreadyRunning: number;
  skipped: number;
  results: Array<{ meetingId: string; status: string; jobId?: string | null; alreadyRunning?: boolean; error?: string | null }>;
}

function BulkTranscribeModal({
  selectedIds,
  listSearch,
  onClose,
  onQueued
}: {
  selectedIds: string[];
  listSearch: string;
  onClose: () => void;
  onQueued: (summary: string, queuedIds: string[]) => void;
}): ReactElement {
  const [scope, setScope] = useState<"selected" | "filtered">(selectedIds.length > 0 ? "selected" : "filtered");
  const [provider, setProvider] = useState<string>("assemblyai");
  const [force, setForce] = useState(true);
  const [search, setSearch] = useState(listSearch);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [currentProvider, setCurrentProvider] = useState("any");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BulkQueueResult | null>(null);

  useEffect(() => {
    if (selectedIds.length > 0) setScope("selected");
  }, [selectedIds.length]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const filters: Record<string, unknown> = {};
      if (scope === "filtered") {
        if (search.trim()) filters.search = search.trim();
        if (startDate) {
          const ms = new Date(`${startDate}T00:00:00`).getTime();
          if (!Number.isNaN(ms)) filters.startTime = ms;
        }
        if (endDate) {
          const ms = new Date(`${endDate}T23:59:59.999`).getTime();
          if (!Number.isNaN(ms)) filters.endTime = ms;
        }
        if (currentProvider && currentProvider !== "any") filters.currentProvider = currentProvider;
      }
      const body: Record<string, unknown> = { provider, force };
      if (scope === "selected") {
        body.meetingIds = selectedIds;
      } else {
        body.filters = filters;
      }
      const res = await fetch("/api/meetings/bulk-transcribe", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data as { error?: string }).error || `HTTP ${res.status}`);
      const queued = data as BulkQueueResult;
      setResult(queued);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const failures = (result?.results || []).filter((r) => r.status === "skipped").slice(0, 8);

  return (
    <Modal onClose={onClose} title={"Bulk re-transcribe"} ariaLabel="Bulk re-transcribe meetings" maxWidth="max-w-lg">
      {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-xs text-error">{error}</div>}

      {result ? (
        <div className="space-y-4 text-sm">
          <div className="rounded-xl border border-primary bg-primary/10 p-4 text-primary">
            Queued {result.queued} of {result.total} meeting{result.total === 1 ? "" : "s"} with <strong>{result.provider}</strong>
            {result.alreadyRunning > 0 && <span> ({result.alreadyRunning} already running)</span>}
            {result.skipped > 0 && <span> ({result.skipped} skipped)</span>}. Watch progress on the meetings list.
          </div>
          {failures.length > 0 && (
            <ul className="space-y-1 text-xs text-base-content/70">
              {failures.map((f) => (
                <li key={f.meetingId} className="rounded-lg border border-base-300 bg-base-100/60 px-3 py-2">
                  <span className="font-mono text-base-content">{f.meetingId.slice(0, 8)}…</span>: {f.error || "skipped"}
                </li>
              ))}
            </ul>
          )}
          <div className="flex justify-end gap-3">
            <button
              type="button"
              onClick={() => onQueued(
                `Bulk job ${result.bulkId}: queued ${result.queued}/${result.total} with ${result.provider}.`,
                (result.results || []).filter((r) => r.status === "running").map((r) => r.meetingId)
              )}
              className="btn btn-primary min-h-11"
            >
              Done
            </button>
          </div>
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4 text-sm">
          <div className="grid gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-base-content/70">Which meetings?</span>
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-base-300 bg-base-100/60 px-3 py-2.5 hover:border-base-300 transition">
              <input type="radio" name="bulk-scope" checked={scope === "selected"} onChange={() => setScope("selected")} disabled={selectedIds.length === 0} className="accent-primary" />
              <span>Selected meetings ({selectedIds.length}){selectedIds.length === 0 && " — tick meetings in the list first"}</span>
            </label>
            <label className="flex cursor-pointer items-center gap-2 rounded-xl border border-base-300 bg-base-100/60 px-3 py-2.5 hover:border-base-300 transition">
              <input type="radio" name="bulk-scope" checked={scope === "filtered"} onChange={() => setScope("filtered")} className="accent-primary" />
              <span>All meetings matching filters (whole library, server-side)</span>
            </label>
          </div>

          {scope === "filtered" && (
            <div className="space-y-3 rounded-xl border border-base-300 bg-base-100/60 p-4">
              <input
                type="text"
                placeholder="Search title, tags, transcript…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="input input-md w-full"
              />
              <div className="grid grid-cols-2 gap-3">
                <label className="space-y-1 text-xs text-base-content/70">
                  <span>From date</span>
                  <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="input input-md w-full" />
                </label>
                <label className="space-y-1 text-xs text-base-content/70">
                  <span>To date</span>
                  <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="input input-md w-full" />
                </label>
              </div>
              <label className="space-y-1 text-xs text-base-content/70">
                <span>Current transcript provider</span>
                <Combobox
                  value={currentProvider}
                  onChange={setCurrentProvider}
                  placeholder="Select provider filter…"
                  ariaLabel="Current transcript provider"
                  searchPlaceholder="Search providers…"
                  options={TRANSCRIPT_PROVIDER_FILTER_OPTIONS}
                />
              </label>
            </div>
          )}

          <label className="space-y-1 text-xs text-base-content/70">
            <span className="text-xs font-semibold uppercase tracking-wider">Transcribe with</span>
            <Combobox
              value={provider}
              onChange={setProvider}
              placeholder="Select engine…"
              ariaLabel="Transcribe with"
              searchPlaceholder="Search engines…"
              options={TRANSCRIBE_PROVIDER_OPTIONS}
            />
          </label>

          <label className="flex cursor-pointer items-center gap-2 text-xs text-base-content">
            <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} className="checkbox checkbox-sm" />
            Force re-transcribe even if already done (recommended when switching engines)
          </label>

          <div className="flex justify-end gap-3 pt-1">
            <button type="button" onClick={onClose} className="rounded-lg px-4 py-2 text-sm text-base-content/70 hover:text-base-content">Cancel</button>
            <button
              type="submit"
              disabled={submitting || (scope === "selected" && selectedIds.length === 0)}
              className="btn btn-primary min-h-11"
            >
              {submitting ? "Queueing…" : scope === "selected" ? `Queue ${selectedIds.length} meeting${selectedIds.length === 1 ? "" : "s"}` : "Queue matching meetings"}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}

function SegmentActionsMenu({
  showConfirm,
  showMerge,
  showWrong,
  onConfirm,
  onChange,
  onSplit,
  onMerge,
  onWrong
}: {
  showConfirm: boolean;
  showMerge: boolean;
  showWrong: boolean;
  onConfirm: () => void;
  onChange: () => void;
  onSplit: () => void;
  onMerge: () => void;
  onWrong: () => void;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const pick = (action: () => void) => {
    setOpen(false);
    action();
  };
  const itemClass = "flex w-full min-h-[44px] items-center gap-2 rounded-field px-3 text-left text-sm transition hover:bg-base-200 focus-visible:bg-base-200";

  return (
    <div className="relative shrink-0" onKeyDown={(event) => {
      if (event.key === "Escape") setOpen(false);
    }}>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Turn actions"
        onClick={() => setOpen((prev) => !prev)}
        className="btn btn-ghost btn-square min-h-[44px] min-w-[44px] text-lg leading-none"
      >
        <Ellipsis className="h-4 w-4 shrink-0" aria-hidden="true" />
      </button>
      {open && (
        <>
          <button
            type="button"
            aria-label="Close turn actions"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 cursor-default"
          />
          <div role="menu" className="menu absolute right-0 top-full z-50 mt-1 min-w-[190px] rounded-box border border-base-300 bg-base-100 p-1 shadow-lg">
            {showConfirm && (
              <button type="button" role="menuitem" className={`${itemClass} font-semibold text-primary`} onClick={() => pick(onConfirm)} title="Confirm this speaker is correct & update voiceprint">
                <Check className="h-4 w-4 shrink-0" aria-hidden="true" /> Confirm
              </button>
            )}
            <button type="button" role="menuitem" className={`${itemClass} text-base-content`} onClick={() => pick(onChange)} title="Change speaker for this turn or all turns">
              <Pencil className="h-4 w-4 shrink-0" aria-hidden="true" /> Change
            </button>
            <button type="button" role="menuitem" className={`${itemClass} text-warning`} onClick={() => pick(onSplit)} title="Split this turn into two separate segments">
              <Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split
            </button>
            {showMerge && (
              <button type="button" role="menuitem" className={`${itemClass} text-base-content`} onClick={() => pick(onMerge)} title="Merge this turn with the following turn">
                <Link2 className="h-4 w-4 shrink-0" aria-hidden="true" /> Merge
              </button>
            )}
            {showWrong && (
              <button type="button" role="menuitem" className={`${itemClass} text-error`} onClick={() => pick(onWrong)} title="Mark as incorrect (reset to Unknown speaker)">
                <X className="h-4 w-4 shrink-0" aria-hidden="true" /> Wrong
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function MeetingDetailView({
  meetingId,
  knownSpeakers,
  onBack,
  onDeleted,
  onSpeakerUpdated,
  onChatOpenChange,
  onChatTitle
}: {
  meetingId: string;
  knownSpeakers: SpeakerWithStats[];
  onBack: () => void;
  onDeleted: () => void;
  onSpeakerUpdated: () => void;
  onChatOpenChange: (open: boolean) => void;
  onChatTitle: (title: string) => void;
}): ReactElement {
  const [detail, setDetail] = useState<MeetingDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [cancellingTranscription, setCancellingTranscription] = useState(false);
  const [transcriptionProgress, setTranscriptionProgress] = useState<TranscriptionProgressUpdate | null>(null);
  const transcriptionAbortRef = useRef<AbortController | null>(null);
  const [selectedEngine, setSelectedEngine] = useState<string>("local:cohere");
  const [showDiarizationSettings, setShowDiarizationSettings] = useState(false);
  const [customClusteringThreshold, setCustomClusteringThreshold] = useState(0.50);
  const [customSimilarityThreshold, setCustomSimilarityThreshold] = useState(0.45);
  const [enhancedAudio, setEnhancedAudio] = useState(true);
  const [reassigningSpeaker, setReassigningSpeaker] = useState<{ speaker: string; segmentIndex?: number } | null>(null);
  const [reidentifyingSpeakers, setReidentifyingSpeakers] = useState(false);
  const [reidentifyStatus, setReidentifyStatus] = useState<SpeakerReidentificationStatus | null>(null);
  const [showRapidEnroll, setShowRapidEnroll] = useState(false);
  const [splittingSegment, setSplittingSegment] = useState<{ segmentIndex: number; speaker: string; text: string; startMs: number; endMs: number } | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [selectedSummaryId, setSelectedSummaryId] = useState<string | null>(null);
  const [summaryViewMode, setSummaryViewMode] = useState<"rendered" | "raw">("rendered");
  const [regenerating, setRegenerating] = useState(false);
  const [summaryGeneration, setSummaryGeneration] = useState<SummaryGenerationStatus | null>(null);
  const [copied, setCopied] = useState(false);
  const [confirmationNotice, setConfirmationNotice] = useState<string | null>(null);
  const [playingSegmentIndex, setPlayingSegmentIndex] = useState<number | null>(null);
  const playbackStopMsRef = useRef<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleInput, setTitleInput] = useState("");
  const [savingTitle, setSavingTitle] = useState(false);
  const [autoTitling, setAutoTitling] = useState(false);
  const [editingStartTime, setEditingStartTime] = useState(false);
  const [startTimeInput, setStartTimeInput] = useState("");
  const [savingStartTime, setSavingStartTime] = useState(false);
  const [newTagInput, setNewTagInput] = useState("");
  const [savingTags, setSavingTags] = useState(false);
  const [mobileSection, setMobileSection] = useState<"summary" | "transcript" | null>(null);
  const [retryingAudio, setRetryingAudio] = useState(false);
  const [audioRetryNotice, setAudioRetryNotice] = useState<string | null>(null);
  const [audioDurationMs, setAudioDurationMs] = useState<number | null>(null);
  const [splitMs, setSplitMs] = useState<number | null>(null);
  const [showSplitCard, setShowSplitCard] = useState(false);
  const [showSplitConfirm, setShowSplitConfirm] = useState(false);
  const [splittingMeeting, setSplittingMeeting] = useState(false);
  const [splitResult, setSplitResult] = useState<{ newMeetingId: string; newTitle: string } | null>(null);

  const audioStatus = detail?.meeting.audioStatus ?? (detail && detail.recordings.length === 0 ? "retrieving" : "ready");
  const hasNoAudio = !!detail && audioStatus !== "ready";

  const handleRetryAudio = async () => {
    if (retryingAudio) return;
    setRetryingAudio(true);
    setAudioRetryNotice(null);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/audio/retry`, { method: "POST" });
      const data = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setAudioRetryNotice("Refetch queued — the audio will appear automatically once Plaud Cloud serves it.");
      await fetchDetail();
    } catch (err) {
      setAudioRetryNotice(err instanceof Error ? err.message : "Failed to queue audio refetch");
    } finally {
      setRetryingAudio(false);
    }
  };

  const splitTotalMs =
    audioDurationMs ?? detail?.recordings[0]?.durationMs ?? null;
  const SPLIT_EDGE_MS = 5000;
  const splitValid =
    splitMs !== null &&
    splitTotalMs !== null &&
    splitMs >= SPLIT_EDGE_MS &&
    splitMs <= splitTotalMs - SPLIT_EDGE_MS;
  const splitAllowedAt = (ms: number) =>
    splitTotalMs === null || (ms >= SPLIT_EDGE_MS && ms <= splitTotalMs - SPLIT_EDGE_MS);
  const clampSplitMs = (ms: number) =>
    splitTotalMs === null
      ? ms
      : Math.min(Math.max(ms, SPLIT_EDGE_MS), splitTotalMs - SPLIT_EDGE_MS);

  const confirmSplitMeeting = async () => {
    if (splitMs === null || splittingMeeting) return;
    setSplittingMeeting(true);
    setError(null);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/split`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ splitMs })
      });
      const data = (await res.json().catch(() => ({}))) as {
        error?: string;
        newMeeting?: { id: string; title: string };
      };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (!data.newMeeting) throw new Error("Split returned no new meeting");
      setShowSplitConfirm(false);
      setShowSplitCard(false);
      setSplitMs(null);
      setSplitResult({ newMeetingId: data.newMeeting.id, newTitle: data.newMeeting.title });
      await fetchDetail();
      // The audio file changed on disk: drop the cached duration (which
      // otherwise keeps splitTotalMs pinned to the pre-split length) and
      // force the <audio> element to re-request instead of replaying its
      // buffered/stale source.
      setAudioDurationMs(null);
      if (audioRef.current) audioRef.current.load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to split meeting");
    } finally {
      setSplittingMeeting(false);
    }
  };

  const startRename = () => {
    if (!detail) return;
    setTitleInput(detail.meeting.title);
    setEditingTitle(true);
    setError(null);
  };

  const saveTitle = async () => {
    const title = titleInput.trim();
    if (!title) {
      setError("Meeting title cannot be empty");
      return;
    }

    setSavingTitle(true);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/title`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      const data = (await res.json().catch(() => ({}))) as { error?: string; meeting?: MeetingListItem };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (!data.meeting) throw new Error("Meeting title update returned no meeting");
      setDetail((prev) => (prev ? { ...prev, meeting: { ...prev.meeting, title: data.meeting!.title } } : prev));
      setEditingTitle(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to rename meeting");
    } finally {
      setSavingTitle(false);
    }
  };

  const startEditStartTime = () => {
    if (!detail) return;
    setStartTimeInput(toDatetimeLocalValue(detail.meeting.startTime));
    setEditingStartTime(true);
    setError(null);
  };

  const saveStartTime = async () => {
    const ms = new Date(startTimeInput).getTime();
    if (!startTimeInput || Number.isNaN(ms)) {
      setError("Enter a valid meeting date and time");
      return;
    }

    setSavingStartTime(true);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/start-time`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ startTime: ms })
      });
      const data = (await res.json().catch(() => ({}))) as { error?: string; meeting?: MeetingListItem };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (!data.meeting) throw new Error("Meeting time update returned no meeting");
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              meeting: {
                ...prev.meeting,
                startTime: data.meeting!.startTime,
                endTime: data.meeting!.endTime
              }
            }
          : prev
      );
      setEditingStartTime(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update meeting time");
    } finally {
      setSavingStartTime(false);
    }
  };

  const handleAutoTitle = async () => {
    if (autoTitling || savingTitle || editingTitle) return;
    setAutoTitling(true);
    setError(null);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/auto-title`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
      });
      const data = (await res.json().catch(() => ({}))) as { error?: string; meeting?: MeetingListItem };
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      if (!data.meeting) throw new Error("Auto-title returned no meeting");
      setDetail((prev) => (prev ? { ...prev, meeting: { ...prev.meeting, title: data.meeting!.title } } : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to auto-generate title");
    } finally {
      setAutoTitling(false);
    }
  };

  const saveTags = async (nextTags: string[]) => {
    setSavingTags(true);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/tags`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tags: nextTags })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as { meeting: MeetingListItem };
      setDetail((prev) => (prev ? { ...prev, meeting: { ...prev.meeting, tags: data.meeting.tags } } : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update tags");
    } finally {
      setSavingTags(false);
    }
  };

  const handleAddTag = () => {
    const tag = newTagInput.trim();
    if (!tag) return;
    const current = detail?.meeting.tags ?? [];
    if (current.includes(tag)) {
      setNewTagInput("");
      return;
    }
    setNewTagInput("");
    void saveTags([...current, tag]);
  };

  const handleRemoveTag = (tag: string) => {
    const current = detail?.meeting.tags ?? [];
    void saveTags(current.filter((t) => t !== tag));
  };

  const fetchDetail = async () => {
    if (!detail) {
      setLoading(true);
    }
    try {
      const [res, cfgRes] = await Promise.all([
        fetch(`/api/meetings/${meetingId}`),
        fetch("/api/config")
      ]);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as MeetingDetailResponse;
      setDetail(data);
      if (data.summaryGeneration?.stage === "running") {
        setSummaryGeneration(data.summaryGeneration);
      }
      if (data.speakerReidentification) {
        const stage = data.speakerReidentification.stage;
        if (stage === "queued" || stage === "running") {
          setReidentifyStatus(data.speakerReidentification);
          setReidentifyingSpeakers(true);
        } else {
          setReidentifyStatus(data.speakerReidentification);
          setReidentifyingSpeakers(false);
        }
      }
      if (data.transcriptionProgress) {
        const stage = data.transcriptionProgress.stage;
        if (stage === "done" || stage === "error" || stage === "cancelled") {
          // Terminal state (e.g. just-cancelled): not active work. Clearing
          // here keeps a cancel from flipping back to in-progress when the
          // detail refresh lands after the optimistic cancelled UI.
          setTranscriptionProgress(null);
          setTranscribing(false);
        } else {
          setTranscriptionProgress(data.transcriptionProgress);
          setTranscribing(true);
        }
      } else if (data.meeting?.status === "processing") {
        setTranscribing(true);
      } else if (data.meeting?.status === "ready") {
        setTranscribing(false);
      }
      if (data.summaries && data.summaries.length > 0) {
        const primary = data.summaries.find((s) => s.isPrimary);
        setSelectedSummaryId((prev) => prev || primary?.id || data.summaries[0].id);
      }
      if (cfgRes.ok) {
        const cfg = await cfgRes.json();
        if (cfg.localClusteringThreshold !== undefined) setCustomClusteringThreshold(cfg.localClusteringThreshold);
        if (cfg.localSimilarityThreshold !== undefined) setCustomSimilarityThreshold(cfg.localSimilarityThreshold);
        if (cfg.transcriptionProvider === "speechmatics" || cfg.transcriptionProvider === "assemblyai" || cfg.transcriptionProvider === "muse") {
          setSelectedEngine(cfg.transcriptionProvider);
        } else {
          setSelectedEngine("local:cohere");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load meeting");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDetail();

    let eventSource: EventSource | null = null;
    let isClosed = false;

    const connectProgress = () => {
      try {
        eventSource = new EventSource(`/api/meetings/${meetingId}/transcribe/progress?stream=true`);

        eventSource.addEventListener("progress", (e) => {
          if (isClosed) return;
          try {
            const data = JSON.parse(e.data);
            if (data && (data.stage === "done" || data.stage === "error" || data.stage === "cancelled")) {
              // Mislabeled terminal state (e.g. a snapshot taken right after
              // cancel): never treat it as active work.
              return;
            }
            setTranscriptionProgress(data);
            setTranscribing(true);
          } catch {}
        });

        eventSource.addEventListener("result", (e) => {
          if (isClosed) return;
          try {
            const data = JSON.parse(e.data);
            setTranscriptionProgress(data);
          } catch {}
          setTranscribing(false);
          void fetchDetail();
          setTimeout(() => {
            if (!isClosed) setTranscriptionProgress(null);
          }, 3500);
        });

        eventSource.addEventListener("cancelled", () => {
          if (isClosed) return;
          setTranscriptionProgress({
            stage: "done",
            percent: 0,
            message: "Transcription was cancelled."
          });
          setTranscribing(false);
          void fetchDetail();
          setTimeout(() => {
            if (!isClosed) setTranscriptionProgress(null);
          }, 2500);
        });

        eventSource.addEventListener("error", (e) => {
          if (isClosed) return;
          if (e instanceof MessageEvent && e.data) {
            try {
              const data = JSON.parse(e.data);
              setError(data.error || data.message || "Transcription failed");
            } catch {}
            setTranscribing(false);
            setTimeout(() => {
              if (!isClosed) setTranscriptionProgress(null);
            }, 3500);
          }
        });
      } catch (err) {
        console.warn("Could not connect to SSE progress stream:", err);
      }
    };

    connectProgress();

    return () => {
      isClosed = true;
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [meetingId]);

  const summaryGenerating = summaryGeneration?.stage === "running";

  useEffect(() => {
    if (!summaryGenerating) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`/api/meetings/${meetingId}/summarize/status`);
        if (!res.ok) return;
        const data = (await res.json()) as { generating: boolean; status: SummaryGenerationStatus | null };
        if (cancelled) return;

        if (!data.status) {
          // Lost track of the job (e.g. server restart) — stop waiting
          setSummaryGeneration(null);
          return;
        }
        if (data.status.stage === "running") return;

        if (data.status.stage === "done") {
          setSummaryGeneration(null);
          if (data.status.artifactId) {
            setSelectedSummaryId(data.status.artifactId);
          }
          await fetchDetail();
        } else {
          setSummaryGeneration(data.status);
          setTimeout(() => {
            if (!cancelled) setSummaryGeneration(null);
          }, 8000);
        }
      } catch {
        // transient network error — keep polling
      }
    };

    const interval = setInterval(() => void poll(), 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [summaryGenerating, meetingId]);

  const handleCancelTranscription = async () => {
    setCancellingTranscription(true);
    if (transcriptionAbortRef.current) {
      try {
        transcriptionAbortRef.current.abort();
      } catch {}
      transcriptionAbortRef.current = null;
    }

    setTranscriptionProgress({
      stage: "done",
      percent: 0,
      message: "Cancelling transcription..."
    });

    try {
      await fetch(`/api/meetings/${meetingId}/transcribe/cancel`, { method: "POST" });
    } catch (err) {
      console.warn("Failed to call cancel API:", err);
    } finally {
      setTranscribing(false);
      setCancellingTranscription(false);
      await fetchDetail();
      setTimeout(() => setTranscriptionProgress(null), 2500);
    }
  };

  const handleTriggerTranscription = async () => {
    if (hasNoAudio) {
      setError(
        audioStatus === "retrieving"
          ? "Audio is still being fetched from Plaud Cloud — transcription unlocks once it lands."
          : "No audio file is attached to this meeting — transcription needs audio first."
      );
      return;
    }
    setTranscribing(true);
    setCancellingTranscription(false);
    const abortController = new AbortController();
    transcriptionAbortRef.current = abortController;

    setTranscriptionProgress({
      stage: "decoding",
      percent: 3,
      message: "Starting transcription pipeline..."
    });

    const isLocal = selectedEngine.startsWith("local");
    const modelId = "onnx-community/cohere-transcribe-03-2026-ONNX";

    try {
      const res = await fetch(`/api/meetings/${meetingId}/transcribe`, {
        method: "POST",
        signal: abortController.signal,
        headers: {
          "content-type": "application/json"
        },
        body: JSON.stringify({
          background: true,
          provider: isLocal ? "local" : selectedEngine,
          modelId: isLocal ? modelId : undefined,
          poll: true,
          force: true,
          clusteringThreshold: isLocal ? customClusteringThreshold : undefined,
          similarityThreshold: isLocal ? customSimilarityThreshold : undefined
        })
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
    } catch (err: any) {
      if (abortController.signal.aborted || err?.name === "AbortError") {
        return;
      }
      setTranscribing(false);
      setTranscriptionProgress(null);
      notify("error", `Transcription error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      transcriptionAbortRef.current = null;
    }
  };

  const handleSetPrimarySummary = async (artifactId: string) => {
    try {
      const res = await fetch(`/api/meetings/${meetingId}/summaries/${artifactId}/primary`, {
        method: "POST"
      });
      if (!res.ok) throw new Error("Failed to set primary summary");
      await fetchDetail();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  const handleReidentifySpeakers = async () => {
    if (!detail || reidentifyingSpeakers) return;
    setReidentifyingSpeakers(true);
    setError(null);
    try {
      const res = await fetch(`/api/meetings/${meetingId}/speakers/reidentify`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      // Poll the tracked job status (queued -> running -> done/error) just
      // like transcription progress and summary generation — no updatedAt
      // guessing.
      const deadline = Date.now() + 5 * 60 * 1000;
      for (;;) {
        await new Promise((r) => setTimeout(r, 2000));
        try {
          const poll = await fetch(`/api/meetings/${meetingId}/speakers/reidentify/status`);
          if (poll.ok) {
            const body = (await poll.json()) as { status: SpeakerReidentificationStatus | null };
            if (body.status) setReidentifyStatus(body.status);
            const stage = body.status?.stage;
            if (stage === "done" || stage === "error" || stage === "cancelled") break;
          }
        } catch {}
        if (Date.now() > deadline) break;
      }
      await fetchDetail();
      onSpeakerUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to re-identify speakers");
    } finally {
      setReidentifyingSpeakers(false);
    }
  };

  const handleDeleteMeeting = async () => {
    if (!detail) return;
    if (
      !(await confirmAction({ title: "Delete meeting", message: `Are you sure you want to delete meeting "${detail.meeting.title}"? This will permanently remove its recordings, transcripts, and summaries.`, confirmLabel: "Delete", danger: true }))
    ) {
      return;
    }
    try {
      const res = await fetch(`/api/meetings/${meetingId}`, { method: "DELETE" });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error || `Failed to delete meeting (HTTP ${res.status})`);
      }
      onDeleted();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  const handleDeleteSummary = async (artifactId: string) => {
    if (!(await confirmAction({ title: "Delete summary", message: "Are you sure you want to delete this summary?", confirmLabel: "Delete", danger: true }))) {
      return;
    }
    try {
      const res = await fetch(`/api/meetings/${meetingId}/summaries/${artifactId}`, {
        method: "DELETE"
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        throw new Error(data.error || `Failed to delete summary (HTTP ${res.status})`);
      }
      setSelectedSummaryId(null);
      await fetchDetail();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  const handleCopySummary = (content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleConfirmSegment = async (segmentIndex: number) => {
    try {
      const res = await fetch(`/api/meetings/${meetingId}/speakers/confirm-segment`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ segmentIndex })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      const data = (await res.json().catch(() => ({}))) as {
        voiceprintStatus?: string;
        statusReason?: string;
        speaker?: { name: string };
      };

      if (data.voiceprintStatus === "enrolled") {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}" & enrolled voiceprint sample`);
      } else if (data.voiceprintStatus === "enrolled_evicted") {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}" (pool full: rotated oldest confirmed clip)`);
      } else if (data.voiceprintStatus === "redundant_skipped") {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}" (profile already optimal — redundant audio skipped)`);
      } else if (data.voiceprintStatus === "outlier_rejected") {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}" in transcript (voiceprint skipped: acoustic mismatch/noise)`);
      } else if (data.voiceprintStatus === "duration_skipped") {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}" in transcript (voiceprint skipped: clip under 3s — too short to enroll)`);
      } else {
        setConfirmationNotice(`✓ Confirmed "${data.speaker?.name || "speaker"}"`);
      }
      setTimeout(() => setConfirmationNotice(null), 4000);

      await fetchDetail();
      onSpeakerUpdated();
    } catch (err) {
      notify("error", `Failed to confirm speaker: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleUnassignSegment = async (segmentIndex: number) => {
    try {
      const res = await fetch(`/api/meetings/${meetingId}/speakers/unassign-segment`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ segmentIndex })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      await fetchDetail();
      onSpeakerUpdated();
    } catch (err) {
      notify("error", `Failed to unassign speaker: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handleMergeWithNext = async (segmentIndex: number) => {
    try {
      const res = await fetch(`/api/meetings/${meetingId}/speakers/merge-segments`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ segmentIndex })
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      await fetchDetail();
      onSpeakerUpdated();
    } catch (err) {
      notify("error", `Failed to merge segments: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const handlePlaySegment = (seg: { startMs: number; endMs: number }, idx: number) => {
    if (!audioRef.current) return;

    if (playingSegmentIndex === idx && !audioRef.current.paused) {
      audioRef.current.pause();
      setPlayingSegmentIndex(null);
      playbackStopMsRef.current = null;
      return;
    }

    audioRef.current.currentTime = seg.startMs / 1000;
    playbackStopMsRef.current = seg.endMs;
    setPlayingSegmentIndex(idx);
    void audioRef.current.play();
  };

  let parsedTranscriptSegments: Array<{
    speaker: string;
    text: string;
    startMs: number;
    endMs: number;
    verified?: boolean;
    segmentIndex: number;
  }> = [];
  if (detail?.transcriptContent) {
    try {
      const parsed = JSON.parse(detail.transcriptContent);
      if (Array.isArray(parsed.segments)) {
        parsedTranscriptSegments = parsed.segments.map((s: any, idx: number) => ({
          speaker: s.speaker || "Speaker",
          text: s.text || s.content || "",
          startMs: s.startMs ?? s.start_time ?? 0,
          endMs: s.endMs ?? s.end_time ?? 0,
          verified: Boolean(s.verified),
          segmentIndex: idx
        }));
      } else if (Array.isArray(parsed)) {
        parsedTranscriptSegments = parsed
          .filter((item: any) => (item.text && item.text.trim()) || (item.content && item.content.trim()))
          .map((item: any, idx: number) => ({
            speaker: item.speaker || "Speaker",
            text: item.text || item.content || "",
            startMs: item.startMs ?? item.start_time ?? 0,
            endMs: item.endMs ?? item.end_time ?? 0,
            verified: Boolean(item.verified),
            segmentIndex: idx
          }));
      }
    } catch {
      // plain text fallback
    }
  }

  const summaries = detail?.summaries || [];
  const activeSummary =
    summaries.find((s) => s.id === selectedSummaryId) ||
    summaries.find((s) => s.isPrimary) ||
    summaries[0] ||
    null;

  // Phone/tablet-portrait only: pick the active segment (defaults to Summary when one exists, else Transcript).
  const activeSection = mobileSection ?? (summaries.length > 0 ? "summary" : "transcript");

  return (
    <div className="space-y-6">
      <button
        type="button"
        onClick={onBack}
        className="flex min-h-[44px] items-center gap-2 text-sm text-base-content/70 hover:text-primary transition"
      >
        ← Back to Meetings
      </button>

      {loading && <p className="text-base-content/70">Loading meeting details…</p>}
      {error && <div className="rounded-xl border border-error bg-error/10 p-4 text-error">{error}</div>}

      {detail && (
        <div className="space-y-8">
          <header className="space-y-4 border-b border-base-300 pb-6">
            <div className="w-full">
              {editingTitle ? (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    void saveTitle();
                  }}
                  className="w-full"
                >
                  <input
                    type="text"
                    value={titleInput}
                    onChange={(event) => setTitleInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") setEditingTitle(false);
                    }}
                    aria-label="Meeting title"
                    autoFocus
                    disabled={savingTitle}
                    className="input w-full text-2xl font-bold tracking-tight sm:text-3xl"
                  />
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <button
                      type="submit"
                      disabled={savingTitle}
                      className="btn btn-primary min-h-11"
                    >
                      {savingTitle ? "Saving…" : "Save"}
                    </button>
                    <button
                      type="button"
                      onClick={() => setEditingTitle(false)}
                      disabled={savingTitle}
                      className="btn min-h-11"
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              ) : (
                <>
                  <h1 className="text-2xl sm:text-3xl font-bold tracking-tight break-words">{detail.meeting.title}</h1>
                  <div className="mt-2 flex flex-wrap items-center gap-1">
                    <button
                      type="button"
                      onClick={startRename}
                      className="btn btn-ghost px-1.5 min-h-11"
                      aria-label="Rename meeting"
                      title="Rename meeting"
                    >
                      <Pencil className="h-4 w-4 shrink-0" aria-hidden="true" />
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleAutoTitle()}
                      disabled={autoTitling || savingTitle}
                      className="btn btn-ghost btn-sm inline-flex items-center gap-1.5 font-medium min-h-11"
                      aria-label="Auto-generate meeting title with default LLM"
                      title="Auto-generate title with default LLM"
                    >
                      <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
                      {autoTitling ? "Titling…" : "Auto-title"}
                    </button>
                  </div>
                </>
              )}
            </div>
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs sm:text-sm text-base-content/70">
                  {editingStartTime ? (
                    <form
                      onSubmit={(event) => {
                        event.preventDefault();
                        void saveStartTime();
                      }}
                      className="flex flex-wrap items-center gap-2"
                    >
                      <input
                        type="datetime-local"
                        value={startTimeInput}
                        onChange={(event) => setStartTimeInput(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") setEditingStartTime(false);
                        }}
                        aria-label="Meeting start time"
                        autoFocus
                        disabled={savingStartTime}
                        className="input input-sm text-xs"
                      />
                      <button
                        type="submit"
                        disabled={savingStartTime}
                        className="btn btn-primary btn-sm min-h-11"
                      >
                        {savingStartTime ? "Saving…" : "Save"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditingStartTime(false)}
                        disabled={savingStartTime}
                        className="rounded-md border border-base-300 px-2 py-1 text-xs text-base-content transition hover:border-base-300 hover:text-base-content disabled:opacity-50"
                      >
                        Cancel
                      </button>
                    </form>
                  ) : (
                    <>
                      <span>{formatDate(detail.meeting.startTime)}</span>
                      <button
                        type="button"
                        onClick={startEditStartTime}
                        className="btn btn-ghost px-1 py-0.5 min-h-11"
                        aria-label="Change meeting time"
                        title="Change meeting time"
                      >
                        <Pencil className="h-4 w-4 shrink-0" aria-hidden="true" />
                      </button>
                    </>
                  )}
                  <span>•</span>
                  <span>{formatDuration(detail.meeting.endTime - detail.meeting.startTime)}</span>
                  <span>•</span>
                  <span className="capitalize">{detail.meeting.source}</span>
                </div>
                <MeetingUsageLine meetingId={meetingId} />
                <div className="mt-3 flex flex-wrap items-center gap-1.5">
                  {detail.meeting.tags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1.5 rounded-full border border-base-300 bg-base-300/80 pl-2.5 pr-1.5 py-1 text-xs text-base-content"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => handleRemoveTag(tag)}
                        disabled={savingTags}
                        className="text-base-content/50 hover:text-error transition disabled:opacity-50"
                        aria-label={`Remove tag ${tag}`}
                      >
                        <X className="h-4 w-4 shrink-0" aria-hidden="true" />
                      </button>
                    </span>
                  ))}
                  <input
                    type="text"
                    value={newTagInput}
                    onChange={(e) => setNewTagInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        handleAddTag();
                      }
                    }}
                    placeholder="+ Add tag"
                    disabled={savingTags}
                    className="input input-md w-full w-24 rounded-full border-dashed bg-transparent px-2.5 text-xs text-base-content placeholder:text-base-content/50 focus:w-32 transition-all disabled:opacity-50"
                  />
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
              <button
                type="button"
                onClick={() => onChatOpenChange(true)}
                className="inline-flex min-h-[44px] basis-full items-center justify-center gap-2 rounded-lg border border-accent/40 bg-accent/10 px-3.5 py-2 text-sm font-semibold text-accent transition hover:border-accent/60 hover:bg-accent/20 shadow-md shrink-0 flex-1 sm:flex-none sm:basis-auto"
              >
                <span><Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
                <span>Chat with Meeting</span>
              </button>
              <div className="w-full sm:w-72">
                <Combobox
                  value={selectedEngine}
                  onChange={setSelectedEngine}
                  placeholder="Select engine…"
                  ariaLabel="Transcription engine"
                  searchPlaceholder="Search engines…"
                  options={ENGINE_OPTIONS}
                />
              </div>

              {selectedEngine.startsWith("local") && (
                <button
                  type="button"
                  onClick={() => setShowDiarizationSettings(!showDiarizationSettings)}
                  className={`min-h-[44px] rounded-lg border px-2.5 py-2 text-xs font-mono transition shrink-0 ${
                    showDiarizationSettings
                      ? "border-primary bg-primary/20 text-primary"
                      : "border-base-300 bg-base-200 text-base-content/70 hover:text-base-content"
                  }`}
                  title="Configure local diarization thresholds"
                >
                  <Settings className="h-4 w-4 shrink-0" aria-hidden="true" /> Thresholds
                </button>
              )}

              <button
                type="button"
                disabled={transcribing || hasNoAudio}
                onClick={() => void handleTriggerTranscription()}
                title={hasNoAudio ? "Transcription unlocks once the meeting audio is attached" : undefined}
                className="btn btn-primary inline-flex min-h-[44px] items-center gap-2 disabled:opacity-75 disabled:cursor-not-allowed shrink-0"
              >
                {transcribing && (
                  <svg className="h-4 w-4 animate-spin text-primary-content" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                )}
                <span>
                  {transcribing
                    ? transcriptionProgress
                      ? transcriptionProgress.stage === "diarizing"
                        ? `Diarizing (${Math.round(transcriptionProgress.percent || 15)}%)…`
                        : transcriptionProgress.stage === "uploading"
                        ? `Uploading (${Math.round(transcriptionProgress.percent || 15)}%)…`
                        : transcriptionProgress.stage === "decoding"
                        ? "Enhancing Audio…"
                        : `Transcribing (${Math.round(transcriptionProgress.percent || 25)}%)…`
                      : "Transcribing…"
                    : "Transcribe"}
                </span>
              </button>

              {transcribing && (
                <button
                  type="button"
                  disabled={cancellingTranscription}
                  onClick={() => void handleCancelTranscription()}
                  className="btn btn-error btn-soft inline-flex min-h-[44px] items-center gap-1.5 border hover:border-error/60 shadow-md shrink-0"
                  title="Cancel inflight transcription"
                >
                  <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                  </svg>
                  <span>{cancellingTranscription ? "Cancelling…" : "Cancel"}</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => void handleDeleteMeeting()}
                className="btn btn-error btn-soft inline-flex min-h-[44px] items-center gap-1.5 border hover:border-error/60 shadow-md shrink-0"
                title="Delete this meeting and all its artifacts"
              >
                <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"></polyline>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                </svg>
                <span>Delete</span>
              </button>
            </div>
            </div>
          </header>

          {/* Audio not yet attached — retrieving or missing banner */}
          {hasNoAudio && (
            <div
              className={`rounded-xl border p-4 shadow-xl space-y-3  ${
                audioStatus === "retrieving"
                  ? "border-info/30 bg-info/30"
                  : "border-error/30 bg-error/10"
              }`}
            >
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  {audioStatus === "retrieving" ? (
                    <svg className="h-4 w-4 animate-spin text-info shrink-0" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                  ) : (
                    <span className="text-error shrink-0">⚠</span>
                  )}
                  <p className={`text-sm ${audioStatus === "retrieving" ? "text-info" : "text-error"}`}>
                    {audioStatus === "retrieving" ? (
                      <><span className="font-semibold">Retrieving audio…</span> This Plaud recording is still downloading from Plaud Cloud. Transcription unlocks automatically once it lands.</>
                    ) : detail?.meeting.source === "plaud" ? (
                      <><span className="font-semibold">Audio file missing.</span> The recording row exists but no audio is on disk. Re-queue the download from Plaud Cloud.</>
                    ) : (
                      <><span className="font-semibold">Audio file missing.</span> No playable audio is attached to this meeting, so transcription is unavailable.</>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  {detail?.meeting.canRetryAudio && (
                    <button
                      type="button"
                      disabled={retryingAudio}
                      onClick={() => void handleRetryAudio()}
                      className="inline-flex min-h-[40px] items-center gap-1.5 rounded-lg border border-info/40 bg-info/10 px-3 py-1.5 text-xs font-semibold text-info transition hover:bg-info/20 disabled:opacity-50"
                    >
                      <span><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
                      <span>{retryingAudio ? "Queueing…" : "Retry download"}</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => void fetchDetail()}
                    className="btn btn-sm inline-flex min-h-[40px] items-center"
                  >
                    Check again
                  </button>
                </div>
              </div>
              {audioRetryNotice && (
                <p className="text-xs text-base-content">{audioRetryNotice}</p>
              )}
            </div>
          )}

          {/* Live Transcription Progress Banner */}
          {transcriptionProgress && (
            <div className="rounded-xl border border-primary/30 bg-base-200/95 p-4 shadow-xl space-y-3 ">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-primary"></span>
                  </span>
                  <span className="font-semibold text-base-content uppercase tracking-wide">
                    {selectedEngine.startsWith("local")
                      ? "Local Pipeline (Cohere Transcribe + Diarization)"
                      : selectedEngine === "assemblyai" ? "AssemblyAI Cloud" : selectedEngine === "muse" ? "Muse Voice Cloud" : "Speechmatics Cloud"}
                  </span>
                  <span className="text-base-content/50">•</span>
                  <span className="text-primary font-medium capitalize">
                    {transcriptionProgress.stage}
                  </span>
                </div>

                <div className="flex items-center gap-3 font-mono text-xs">
                  {transcriptionProgress.current && transcriptionProgress.total && (
                    <span className="text-base-content">
                      Segment <strong className="text-primary">{transcriptionProgress.current}</strong> / {transcriptionProgress.total}
                    </span>
                  )}
                  {transcriptionProgress.currentMs !== undefined && transcriptionProgress.totalMs && (
                    <span className="text-base-content/70">
                      {formatTimeOffset(transcriptionProgress.currentMs)} / {formatTimeOffset(transcriptionProgress.totalMs)}
                    </span>
                  )}
                  <span className="font-bold text-primary text-sm">
                    {Math.round(transcriptionProgress.percent)}%
                  </span>

                  {transcribing && (
                    <button
                      type="button"
                      disabled={cancellingTranscription}
                      onClick={() => void handleCancelTranscription()}
                      className="btn btn-error btn-soft btn-sm ml-1 inline-flex items-center gap-1 py-0.5 border min-h-11"
                    >
                      <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                      </svg>
                      <span>{cancellingTranscription ? "Cancelling…" : "Cancel"}</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Progress Bar */}
              <progress className="progress progress-primary w-full" value={Math.max(3, Math.min(100, transcriptionProgress.percent))} max={100} aria-label="Transcription progress" />

              {/* Status Message */}
              <div className="flex items-center justify-between text-xs text-base-content/70">
                <p className="truncate text-base-content">
                  {transcriptionProgress.message}
                </p>
                {transcriptionProgress.speaker && (
                  <span className="shrink-0 rounded bg-base-300 px-2 py-0.5 text-[11px] font-medium text-accent border border-base-300">
                    Speaker: {transcriptionProgress.speaker}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Expandable Diarization Threshold Tuning */}
          {showDiarizationSettings && selectedEngine.startsWith("local") && (
            <div className="rounded-xl border border-base-300 bg-base-200/90 p-4 space-y-3 backdrop-blur">
              <div className="flex items-center justify-between border-b border-base-300 pb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-primary">
                  <Mic className="h-4 w-4 shrink-0" aria-hidden="true" /> Diarization & Voiceprint Thresholds for this Transcription
                </span>
                <span className="text-[11px] text-base-content/50 font-mono">Overrides Global Settings</span>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 pt-1">
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="meeting-clustering-slider" className="text-base-content font-medium">
                      Within-Meeting Clustering Threshold:
                    </label>
                    <span className="font-mono font-bold text-primary">
                      {customClusteringThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    id="meeting-clustering-slider"
                    type="range"
                    min="0.10"
                    max="0.95"
                    step="0.01"
                    value={customClusteringThreshold}
                    onChange={(e) => setCustomClusteringThreshold(parseFloat(e.target.value))}
                    className="range range-sm w-full"
                  />
                  <p className="text-[10px] text-base-content/50">
                    Controls separation of distinct voices in this meeting. Recommended: 0.50.
                  </p>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="meeting-similarity-slider" className="text-base-content font-medium">
                      Enrolled Speaker Recognition Sensitivity:
                    </label>
                    <span className="font-mono font-bold text-accent">
                      {customSimilarityThreshold.toFixed(2)}
                    </span>
                  </div>
                  <input
                    id="meeting-similarity-slider"
                    type="range"
                    min="0.10"
                    max="0.95"
                    step="0.01"
                    value={customSimilarityThreshold}
                    onChange={(e) => setCustomSimilarityThreshold(parseFloat(e.target.value))}
                    className="range range-sm w-full"
                  />
                  <p className="text-[10px] text-base-content/50">
                    Cosine similarity threshold to recognize known people. Recommended: 0.45.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Speakers summary */}
          {(() => {
            const activeSpeakerNames = new Set(
              parsedTranscriptSegments.map((s) => s.speaker.trim().toLowerCase()).filter(Boolean)
            );
            const visibleSpeakers = (
              activeSpeakerNames.size > 0
                ? detail.speakers.filter((s) => activeSpeakerNames.has(s.name.trim().toLowerCase()))
                : detail.speakers
            ).filter((s) => !isGenericSpeakerLabel(s.name));

            return visibleSpeakers.length > 0 || parsedTranscriptSegments.length > 0 ? (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-base-content/70">Speakers:</span>
                {visibleSpeakers.map((s) => (
                  <span
                    key={s.id}
                    className="rounded-full border border-base-300 bg-base-300/80 px-3 py-1 text-xs font-medium text-primary"
                  >
                    {s.name}
                  </span>
                ))}
                <button
                  type="button"
                  onClick={() => void handleReidentifySpeakers()}
                  disabled={reidentifyingSpeakers}
                  className="btn btn-sm rounded-full hover:border-primary/40 hover:text-primary disabled:cursor-wait min-h-11"
                  title="Re-match every speaker label against your current enrolled voiceprints"
                >
                  {reidentifyingSpeakers
                    ? reidentifyStatus?.stage === "queued"
                      ? <><RotateCw className="h-4 w-4 shrink-0 animate-spin" aria-hidden="true" /> Queued…</>
                      : <><RotateCw className="h-4 w-4 shrink-0 animate-spin" aria-hidden="true" /> Re-identifying…</>
                    : reidentifyStatus?.stage === "error"
                      ? <><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /> Retry re-identify</>
                      : <><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /> Re-identify</>}
                </button>
              </div>
            ) : null;
          })()}

          {/* Audio Player if recording exists */}
          {detail.recordings.length > 0 && (
            <section className="rounded-xl border border-base-300 bg-base-200/90 p-3.5 sm:p-4 space-y-2.5 shadow-lg w-full max-w-full min-w-0 overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-base-content/70">
                <div className="flex flex-wrap items-center gap-2 sm:gap-3 min-w-0">
                  <span className="font-semibold uppercase tracking-wider text-base-content shrink-0">Recording Audio</span>
                  <span className="truncate">{detail.recordings[0].mime} • {(detail.recordings[0].sizeBytes / 1024 / 1024).toFixed(2)} MB</span>
                  {playingSegmentIndex !== null && (
                    <span className="inline-flex items-center gap-1.5 rounded bg-primary/20 px-2 py-0.5 text-[11px] font-semibold text-primary border border-primary/40 animate-pulse shrink-0">
                      <span className="inline-flex items-center gap-1.5"><Play className="h-4 w-4 shrink-0" aria-hidden="true" /> Playing turn #{playingSegmentIndex + 1}</span>
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 shrink-0">
                {/* Voice Clarity Enhancement Toggle */}
                <button
                  type="button"
                  onClick={() => {
                    const currentTime = audioRef.current?.currentTime || 0;
                    const wasPlaying = !audioRef.current?.paused;
                    setEnhancedAudio(!enhancedAudio);
                    setTimeout(() => {
                      if (audioRef.current) {
                        audioRef.current.currentTime = currentTime;
                        if (wasPlaying) void audioRef.current.play();
                      }
                    }, 50);
                  }}
                  className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-medium transition shrink-0 ${
                    enhancedAudio
                      ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/20"
                      : "border-base-300 bg-base-300 text-base-content/70 hover:text-base-content"
                  }`}
                  title="Toggle FFmpeg speech clarity equalizer & adaptive de-noiser"
                >
                  <span className="inline-flex items-center gap-1.5">{enhancedAudio ? <><Sparkles className="h-4 w-4 shrink-0" aria-hidden="true" /> Voice Enhanced (Active)</> : <><VolumeX className="h-4 w-4 shrink-0" aria-hidden="true" /> Original Raw Audio</>}</span>
                </button>
                {splitTotalMs !== null && splitTotalMs > 15000 && (
                  <button
                    type="button"
                    onClick={() => {
                      if (!showSplitCard && splitMs === null) {
                        const secs = audioRef.current?.currentTime;
                        if (secs && Number.isFinite(secs)) setSplitMs(clampSplitMs(Math.round(secs * 1000)));
                      }
                      setShowSplitCard(!showSplitCard);
                    }}
                    className={`btn btn-sm shrink-0 ${
                      showSplitCard
                        ? "btn-primary btn-soft"
                        : ""
                    }`}
                    title="Cut this meeting in two at a chosen point"
                  >
                    <span className="inline-flex items-center gap-1.5"><Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split Meeting</span>
                  </button>
                )}
                </div>
              </div>
              {splitResult && (
                <div className="flex flex-wrap items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 px-3 py-2 text-xs text-primary">
                  <span className="inline-flex items-center gap-1.5"><Check className="h-4 w-4 shrink-0" aria-hidden="true" /> Split complete — Part 2 created as “{splitResult.newTitle}”.</span>
                  <button
                    type="button"
                    onClick={() => window.location.assign(`/meetings/${splitResult.newMeetingId}`)}
                    className="btn btn-primary min-h-11"
                  >
                    Open Part 2 →
                  </button>
                  <button
                    type="button"
                    onClick={() => setSplitResult(null)}
                    className="text-primary/70 hover:text-primary"
                    aria-label="Dismiss split notice"
                  >
                    Dismiss
                  </button>
                </div>
              )}
              <div className="rounded-xl bg-base-300 p-1.5">
              <audio
                ref={audioRef}
                controls
                src={`/api/meetings/${meetingId}/audio?enhanced=${enhancedAudio}`}
                className="block w-full max-w-full h-10 sm:h-11 rounded-full [color-scheme:dark]"
                onLoadedMetadata={() => {
                  const secs = audioRef.current?.duration;
                  if (secs && Number.isFinite(secs)) setAudioDurationMs(Math.round(secs * 1000));
                }}
                onTimeUpdate={() => {
                  if (audioRef.current && playbackStopMsRef.current !== null) {
                    if (audioRef.current.currentTime >= playbackStopMsRef.current / 1000) {
                      audioRef.current.pause();
                      playbackStopMsRef.current = null;
                      setPlayingSegmentIndex(null);
                    }
                  }
                }}
                onPause={() => {
                  setPlayingSegmentIndex(null);
                  playbackStopMsRef.current = null;
                }}
                onEnded={() => {
                  setPlayingSegmentIndex(null);
                  playbackStopMsRef.current = null;
                }}
              />
              </div>
              {showSplitCard && splitTotalMs !== null && splitTotalMs > 15000 && (
                <div className="space-y-2 rounded-lg border border-base-300 bg-base-100/60 p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                    <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider text-base-content/70"><Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split meeting</span>
                    <button
                      type="button"
                      onClick={() => {
                        const secs = audioRef.current?.currentTime;
                        if (secs && Number.isFinite(secs)) setSplitMs(clampSplitMs(Math.round(secs * 1000)));
                      }}
                      className="text-base-content/70 hover:text-primary"
                      title="Set the split point to the current playback position"
                    >
                      Use current position
                    </button>
                  </div>
                  <input
                    type="range"
                    min={Math.floor(SPLIT_EDGE_MS / 1000)}
                    max={Math.floor((splitTotalMs - SPLIT_EDGE_MS) / 1000)}
                    step={1}
                    value={splitMs !== null ? Math.floor(clampSplitMs(splitMs) / 1000) : Math.floor(SPLIT_EDGE_MS / 1000)}
                    onChange={(e) => setSplitMs(clampSplitMs(Number(e.target.value) * 1000))}
                    aria-label="Split point in seconds"
                    className="w-full accent-primary"
                  />
                  <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-base-content/70">
                    <span>
                      {splitMs !== null ? (
                        <>Split at <span className="font-mono font-semibold text-base-content">{formatTimeOffset(splitMs)}</span> → Part 1: {formatTimeOffset(splitMs)} · Part 2: {formatTimeOffset(splitTotalMs - splitMs)}</>
                      ) : (
                        <>Drag to choose where Part 1 ends and Part 2 begins (total {formatTimeOffset(splitTotalMs)})</>
                      )}
                    </span>
                    <button
                      type="button"
                      disabled={!splitValid}
                      onClick={() => setShowSplitConfirm(true)}
                      className="btn btn-primary disabled:opacity-40 disabled:cursor-not-allowed min-h-11"
                      title={!splitValid ? "Drag the slider at least 5 seconds from either edge" : undefined}
                    >
                      Split meeting…
                    </button>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Phone/tablet-portrait section switcher (hidden at lg+, where all sections stack) */}
          <div className="lg:hidden sticky top-14 sm:top-16 z-20 bg-base-100 py-2">
            <div role="tablist" aria-label="Meeting sections" className="tabs tabs-box w-full">
              {(["summary", "transcript"] as const).map((key) => (
                <button
                  key={key}
                  type="button"
                  role="tab"
                  aria-selected={activeSection === key}
                  onClick={() => setMobileSection(key)}
                  className={`tab min-h-[44px] flex-1 ${activeSection === key ? "tab-active" : ""}`}
                >
                  {key === "summary" ? "Summary" : "Transcript"}
                </button>
              ))}
            </div>
          </div>

          {/* AI Summaries Section */}
          <section className={`space-y-4 ${activeSection !== "summary" ? "hidden lg:block" : ""}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-semibold tracking-tight">AI Summaries</h2>
                {summaries.length > 0 && (
                  <span className="rounded-full bg-base-300 px-2.5 py-0.5 text-xs font-medium text-base-content">
                    {summaries.length} version{summaries.length > 1 ? "s" : ""}
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => setShowSummaryModal(true)}
                disabled={summaryGenerating || hasNoAudio}
                className="btn btn-primary btn-sm inline-flex items-center gap-2 px-3.5 disabled:cursor-not-allowed min-h-11"
                title={hasNoAudio ? "Summaries need meeting audio first" : summaryGenerating ? "Summary generation already in progress" : undefined}
              >
                <span>+</span>
                <span>Generate Summary</span>
              </button>
            </div>

            {/* In-flight summary generation placeholder / error */}
            {summaryGeneration?.stage === "error" ? (
              <div className="rounded-xl border border-error/30 bg-error/10 p-4 text-sm text-error ">
                <span className="font-semibold">Summary generation failed:</span> {summaryGeneration.error}
              </div>
            ) : summaryGenerating ? (
              <div className="rounded-xl border border-primary/30 bg-base-200/70 shadow-lg overflow-hidden ">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-base-300 bg-base-100/60 p-2">
                  <div className="flex items-center gap-2 text-xs">
                    <svg className="h-3.5 w-3.5 animate-spin text-primary" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                    </svg>
                    <span className="font-semibold text-base-content">
                      Generating summary
                      {summaryGeneration.provider && summaryGeneration.model
                        ? ` with ${summaryGeneration.provider}:${summaryGeneration.model}`
                        : summaryGeneration.provider
                        ? ` with ${summaryGeneration.provider}`
                        : ""}
                      …
                    </span>
                  </div>
                  <span className="text-[11px] text-base-content/50 font-mono">This can take a minute</span>
                </div>
                <div className="p-6 space-y-3">
                  <div className="h-5 w-2/5 skeleton" />
                  <div className="h-3 w-full skeleton" />
                  <div className="h-3 w-11/12 skeleton" />
                  <div className="h-3 w-4/5 skeleton" />
                  <div className="pt-3 h-4 w-1/3 skeleton" />
                  <div className="h-3 w-full skeleton" />
                  <div className="h-3 w-5/6 skeleton" />
                  <div className="h-3 w-3/5 skeleton" />
                </div>
              </div>
            ) : null}

            {summaries.length > 0 ? (
              <div className="rounded-xl border border-base-300 bg-base-200/70 shadow-lg overflow-hidden">
                {/* Summary Tabs Header */}
                <div className="flex items-center gap-2 border-b border-base-300 bg-base-100/60 p-2 overflow-x-auto">
                  {summaries.map((s, idx) => {
                    const parts = s.provider.split(":");
                    const label = parts[2] || parts[1] || `Summary ${idx + 1}`;
                    const isSelected = (activeSummary && activeSummary.id === s.id) || (!activeSummary && idx === 0);
                    return (
                      <button
                        key={s.id}
                        type="button"
                        onClick={() => setSelectedSummaryId(s.id)}
                        className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition whitespace-nowrap ${
                          isSelected
                            ? "bg-base-300 text-primary shadow border border-base-300"
                            : "text-base-content/70 hover:text-base-content hover:bg-base-200"
                        }`}
                      >
                        <span>{label}</span>
                        {s.isPrimary && (
                          <span className="rounded bg-primary/20 px-1.5 py-0.2 text-[10px] text-primary font-bold">
                            <Star className="h-4 w-4 shrink-0" aria-hidden="true" /> Primary
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>

                {/* Active Summary Content */}
                {activeSummary && (
                  <div className="p-6 space-y-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-base-300/80 pb-3">
                      <div className="flex items-center gap-2 text-xs text-base-content/70">
                        <span className="font-mono text-base-content">{activeSummary.provider}</span>
                        <span>•</span>
                        <span>{formatDate(activeSummary.createdAt)}</span>
                        {activeSummary.isPrimary && (
                          <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary border border-primary/20">
                            Primary Note
                          </span>
                        )}
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        {/* Rendered vs Raw view toggle */}
                        <div className="tabs tabs-box tabs-xs" role="tablist" aria-label="Summary view">
                          <button
                            type="button"
                            role="tab"
                            aria-selected={summaryViewMode === "rendered"}
                            onClick={() => setSummaryViewMode("rendered")}
                            className={`tab ${summaryViewMode === "rendered" ? "tab-active" : ""}`}
                          >
                            <Eye className="h-4 w-4 shrink-0" aria-hidden="true" /> Preview
                          </button>
                          <button
                            type="button"
                            role="tab"
                            aria-selected={summaryViewMode === "raw"}
                            onClick={() => setSummaryViewMode("raw")}
                            className={`tab ${summaryViewMode === "raw" ? "tab-active" : ""}`}
                          >
                            <FileText className="h-4 w-4 shrink-0" aria-hidden="true" /> Raw
                          </button>
                        </div>

                        {!activeSummary.isPrimary && (
                          <button
                            type="button"
                            onClick={() => void handleSetPrimarySummary(activeSummary.id)}
                            className="btn btn-primary btn-soft btn-xs min-h-11"
                          >
                            <Star className="h-4 w-4 shrink-0" aria-hidden="true" /> Set as Primary
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => handleCopySummary(activeSummary.content)}
                          className="btn btn-xs min-h-11"
                        >
                          {copied ? <><Check className="h-4 w-4 shrink-0" aria-hidden="true" /> Copied!</> : <><ClipboardList className="h-4 w-4 shrink-0" aria-hidden="true" /> Copy Markdown</>}
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleDeleteSummary(activeSummary.id)}
                          className="btn btn-error btn-soft btn-xs min-h-11"
                        >
                          Delete
                        </button>
                      </div>
                    </div>

                    {summaryViewMode === "rendered" ? (
                      <MarkdownRenderer content={activeSummary.content} />
                    ) : (
                      <pre className="whitespace-pre-wrap font-mono text-xs text-base-content rounded-xl border border-base-300 bg-base-100/80 p-4 overflow-x-auto leading-relaxed">
                        {activeSummary.content}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            ) : !summaryGenerating ? (
              <div className="rounded-xl border border-dashed border-base-300 bg-base-200/40 p-8 text-center space-y-3">
                <p className="text-sm text-base-content/70">
                  No summary generated yet. Create AI meeting notes using a custom or built-in template.
                </p>
                <button
                  type="button"
                  onClick={() => setShowSummaryModal(true)}
                  className="btn btn-primary btn-sm inline-flex items-center gap-2 min-h-11"
                >
                  <span>+</span>
                  <span>Generate Summary</span>
                </button>
              </div>
            ) : null}
          </section>

          {/* Transcript Viewer */}
          <section className={`space-y-4 w-full max-w-full min-w-0 ${activeSection !== "transcript" ? "hidden lg:block" : ""}`}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-xl font-semibold tracking-tight">Transcript</h2>
              {parsedTranscriptSegments.length > 0 && (
                <button
                  type="button"
                  onClick={() => setShowRapidEnroll(true)}
                  className="btn btn-primary btn-soft inline-flex min-h-[44px] items-center gap-1.5 border"
                  title="Enroll voiceprints for unmatched or unverified speakers in this meeting"
                >
                  <span><Zap className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
                  <span>Rapid enroll</span>
                </button>
              )}
              {confirmationNotice && (
                <div className="rounded-lg border border-primary/40 bg-primary/10 px-3 py-1 text-xs font-medium text-primary ">
                  {confirmationNotice}
                </div>
              )}
            </div>

            {parsedTranscriptSegments.length > 0 ? (
              <div className="space-y-3 sm:space-y-4 rounded-xl border border-base-300 bg-base-200/60 p-3 sm:p-6 w-full max-w-full min-w-0">
                {parsedTranscriptSegments.map((seg, idx) => (
                  <div
                    key={idx}
                    className="group relative rounded-xl border border-base-300/70 bg-base-200/40 p-3 sm:p-4 transition hover:border-base-300 hover:bg-base-200/80 space-y-2 w-full max-w-full min-w-0"
                  >
                    {idx > 0 && splitAllowedAt(seg.startMs) && (
                      <div className="absolute -top-3 left-0 right-0 z-10 flex justify-center opacity-0 transition group-hover:opacity-100 hover:opacity-100 focus-within:opacity-100">
                        <button
                          type="button"
                          onClick={() => {
                            setSplitMs(seg.startMs);
                            setShowSplitConfirm(true);
                          }}
                          title={`Split meeting before this turn (${formatTimeOffset(seg.startMs)})`}
                          className="inline-flex items-center gap-1 rounded-full border border-base-300 bg-base-100 px-2.5 py-0.5 text-[11px] font-medium text-base-content/50 shadow-lg transition hover:border-primary/50 hover:text-primary"
                        >
                          <Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split meeting here · {formatTimeOffset(seg.startMs)}
                        </button>
                      </div>
                    )}
                    {/* Header: Play/Timestamp + Speaker Badge + Action Buttons */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-base-300/50 pb-2">
                      <div className="flex flex-wrap items-center gap-2 min-w-0">
                        {/* Play / Timestamp button */}
                        <button
                          type="button"
                          onClick={() => handlePlaySegment(seg, idx)}
                          title={playingSegmentIndex === idx ? "Pause playback" : "Play only this turn"}
                          className={`inline-flex min-h-[40px] items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-mono transition cursor-pointer shrink-0 ${
                            playingSegmentIndex === idx
                              ? "bg-primary/20 text-primary font-bold border border-primary/40 shadow-sm"
                              : "bg-base-300/90 text-base-content/70 hover:text-primary hover:bg-base-300 border border-base-300/70"
                          }`}
                        >
                          <span className="text-[10px] inline-flex">{playingSegmentIndex === idx ? <Pause className="h-3 w-3 shrink-0" aria-hidden="true" /> : <Play className="h-3 w-3 shrink-0" aria-hidden="true" />}</span>
                          <span>{formatTimeOffset(seg.startMs)}</span>
                        </button>

                        {/* Speaker Badge */}
                        {seg.verified ? (
                          <span
                            className="inline-flex items-center gap-1 rounded-md bg-primary/20 px-2 py-0.5 text-xs font-semibold text-primary border border-primary/40 truncate max-w-[180px] sm:max-w-none"
                            title="Speaker verified & voiceprint trained"
                          >
                            <span><Check className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
                            <span className="truncate">{seg.speaker}</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 rounded-md bg-base-300 px-2 py-0.5 text-xs font-semibold text-base-content border border-base-300 truncate max-w-[180px] sm:max-w-none">
                            <span className="truncate">{seg.speaker}</span>
                          </span>
                        )}
                      </div>

                      {/* Turn actions menu */}
                      <SegmentActionsMenu
                        showConfirm={!seg.verified && !isGenericSpeakerLabel(seg.speaker)}
                        showMerge={idx < parsedTranscriptSegments.length - 1}
                        showWrong={seg.speaker !== "Unknown"}
                        onConfirm={() => void handleConfirmSegment(seg.segmentIndex)}
                        onChange={() => setReassigningSpeaker({ speaker: seg.speaker, segmentIndex: seg.segmentIndex })}
                        onSplit={() => setSplittingSegment(seg)}
                        onMerge={() => void handleMergeWithNext(seg.segmentIndex)}
                        onWrong={() => void handleUnassignSegment(seg.segmentIndex)}
                      />
                    </div>

                    {/* Turn Content */}
                    <p className="text-sm leading-relaxed text-base-content pl-0.5 break-words">{seg.text}</p>
                  </div>
                ))}
              </div>
            ) : detail.transcriptContent ? (
              <pre className="whitespace-pre-wrap rounded-xl border border-base-300 bg-base-200/60 p-4 sm:p-6 text-sm font-sans text-base-content break-words overflow-x-auto">
                {detail.transcriptContent}
              </pre>
            ) : (
              <div className="rounded-xl border border-dashed border-base-300 bg-base-200/40 p-8 text-center text-base-content/70">
                No transcript generated yet. Click "Transcribe" above to generate a transcript.
              </div>
            )}
          </section>

          {/* Generate Summary Modal */}
          {showSummaryModal && (
            <GenerateSummaryModal
              meetingId={meetingId}
              onClose={() => setShowSummaryModal(false)}
              onStarted={(status) => {
                setShowSummaryModal(false);
                setSummaryGeneration(status);
              }}
            />
          )}

          {/* Chat renders in the App-level drawer. Report the title up for its header. */}
          <ChatTitleReporter title={detail.meeting.title} onChatTitle={onChatTitle} />

          {/* Split Segment Modal */}
          {splittingSegment && (
            <SplitSegmentModal
              meetingId={meetingId}
              segment={splittingSegment}
              knownSpeakers={knownSpeakers}
              onClose={() => setSplittingSegment(null)}
              onSuccess={async () => {
                setSplittingSegment(null);
                await fetchDetail();
                onSpeakerUpdated();
              }}
            />
          )}

          {/* Split Meeting Confirm Modal */}
          {showSplitConfirm && splitMs !== null && splitTotalMs !== null && (
            <Modal onClose={() => setShowSplitConfirm(false)} title="Split meeting in two" ariaLabel="Split meeting confirmation" maxWidth="max-w-md">
              <div className="space-y-4">
                <p className="text-sm text-base-content">
                  Split at <span className="font-mono font-semibold text-base-content">{formatTimeOffset(splitMs)}</span>? The
                  transcript is divided on the same boundary — no re-transcription needed.
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg border border-base-300 bg-base-200 p-3">
                    <p className="font-semibold text-base-content">Part 1 (this meeting)</p>
                    <p className="mt-1 font-mono text-base-content/70">00:00 → {formatTimeOffset(splitMs)}</p>
                  </div>
                  <div className="rounded-lg border border-base-300 bg-base-200 p-3">
                    <p className="font-semibold text-base-content">Part 2 (new meeting)</p>
                    <p className="mt-1 font-mono text-base-content/70">{formatTimeOffset(splitMs)} → {formatTimeOffset(splitTotalMs)}</p>
                  </div>
                </div>
                <p className="text-xs text-warning/90">
                  Summaries on both parts are deleted (they described the whole recording). Delete the unwanted half
                  afterwards if you only want one.
                </p>
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setShowSplitConfirm(false)}
                    disabled={splittingMeeting}
                    className="rounded-lg border border-base-300 px-3.5 py-2 text-sm text-base-content hover:border-base-300 disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => void confirmSplitMeeting()}
                    disabled={splittingMeeting || !splitValid}
                    className="btn btn-primary px-3.5 min-h-11"
                  >
                    {splittingMeeting ? "Splitting…" : "Split meeting"}
                  </button>
                </div>
              </div>
            </Modal>
          )}

          {/* Rapid Enroll Modal */}
          {showRapidEnroll && (
            <RapidEnrollModal
              meetingId={meetingId}
              knownSpeakers={knownSpeakers}
              playingSegmentIndex={playingSegmentIndex}
              onPlaySample={(sample) => handlePlaySegment(sample, sample.segmentIndex)}
              onClose={() => setShowRapidEnroll(false)}
              onAssigned={() => {
                void fetchDetail();
                onSpeakerUpdated();
              }}
            />
          )}

          {/* Reassign Speaker Modal */}
          {reassigningSpeaker && (
            <ReassignSpeakerModal
              meetingId={meetingId}
              fromLabel={reassigningSpeaker.speaker}
              segmentIndex={reassigningSpeaker.segmentIndex}
              knownSpeakers={knownSpeakers}
              onClose={() => setReassigningSpeaker(null)}
              onSuccess={(optimistic) => {
                const prevLabel = reassigningSpeaker.speaker;
                setReassigningSpeaker(null);
                if (optimistic && detail?.transcriptContent) {
                  try {
                    const parsed = JSON.parse(detail.transcriptContent);
                    const targetArray = Array.isArray(parsed.segments)
                      ? parsed.segments
                      : Array.isArray(parsed)
                      ? parsed
                      : null;
                    if (targetArray) {
                      if (optimistic.scope === "single" && optimistic.segmentIndex !== undefined) {
                        if (targetArray[optimistic.segmentIndex]) {
                          targetArray[optimistic.segmentIndex].speaker = optimistic.targetName;
                          targetArray[optimistic.segmentIndex].verified = true;
                        }
                      } else {
                        for (const s of targetArray) {
                          if ((s.speaker || "").trim().toLowerCase() === prevLabel.trim().toLowerCase()) {
                            s.speaker = optimistic.targetName;
                            s.verified = true;
                          }
                        }
                      }
                      setDetail((prev) => (prev ? { ...prev, transcriptContent: JSON.stringify(parsed) } : prev));
                    }
                  } catch {}
                }
              }}
              onSettled={() => {
                void fetchDetail();
                onSpeakerUpdated();
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}

function GenerateSummaryModal({
  meetingId,
  onClose,
  onStarted
}: {
  meetingId: string;
  onClose: () => void;
  onStarted: (status: SummaryGenerationStatus) => void;
}): ReactElement {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [models, setModels] = useState<LlmModelCatalogItem[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>("");
  const [selectedProvider, setSelectedProvider] = useState<string>("google");
  const [selectedModel, setSelectedModel] = useState<string>("gemini-2.5-flash");
  const [thinkingLevel, setThinkingLevel] = useState<string>("low");
  const [setPrimary, setSetPrimary] = useState(true);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [tplRes, llmRes, configRes] = await Promise.all([
          fetch("/api/templates"),
          fetch("/api/llm/models"),
          fetch("/api/llm/config")
        ]);

        if (tplRes.ok) {
          const tData = await tplRes.json();
          const tList: Template[] = tData.templates || [];
          setTemplates(tList);
          const defaultTpl = tList.find((t) => t.isDefault) || tList[0];
          if (defaultTpl) {
            setSelectedTemplateId(defaultTpl.id);
          }
        }

        if (llmRes.ok) {
          const mData = await llmRes.json();
          setModels(mData.models || []);
        }

        if (configRes.ok) {
          const cData = await configRes.json();
          if (cData.defaultProvider) setSelectedProvider(cData.defaultProvider);
          if (cData.defaultModel) setSelectedModel(cData.defaultModel);
          if (cData.defaultThinkingLevel) setThinkingLevel(cData.defaultThinkingLevel);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    };
    void load();
  }, []);

  const handleProviderChange = (pId: string) => {
    setSelectedProvider(pId);
    const pModels = models.filter((m) => m.provider === pId);
    if (pModels.length > 0) {
      setSelectedModel(pModels[0].id);
      const supported = pModels[0].supportedThinkingLevels || ["low", "medium", "high"];
      setThinkingLevel(supported.includes("off") ? "off" : supported[0] || "low");
    }
  };

  const handleModelChange = (mId: string) => {
    setSelectedModel(mId);
    const m = models.find((mod) => mod.provider === selectedProvider && mod.id === mId);
    if (m?.supportedThinkingLevels) {
      if (!m.supportedThinkingLevels.includes(thinkingLevel as any)) {
        setThinkingLevel(m.supportedThinkingLevels[0] || "low");
      }
    }
  };

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setStarting(true);
    setError(null);

    try {
      const res = await fetch(`/api/meetings/${meetingId}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          templateId: selectedTemplateId || undefined,
          provider: selectedProvider,
          model: selectedModel,
          thinkingLevel: selectedModelObj?.reasoning ? thinkingLevel : undefined,
          setPrimary
        })
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      onStarted(data.status);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStarting(false);
    }
  };

  const providers = Array.from(new Set(models.map((m) => m.provider)));
  const providerModels = models.filter((m) => m.provider === selectedProvider);
  const selectedModelObj = models.find((m) => m.provider === selectedProvider && m.id === selectedModel);

  return (
    <Modal onClose={onClose} title={"Generate AI Summary"} ariaLabel="Generate AI summary" maxWidth="max-w-lg">
      {error && (
        <div className="alert alert-soft alert-error p-3 text-xs text-error">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-8 text-center text-sm text-base-content/70">Loading templates & models…</div>
      ) : (
        <form onSubmit={handleGenerate} className="space-y-4">
          <div className="space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
              Summary Template
            </label>
            <Combobox
              value={selectedTemplateId}
              onChange={setSelectedTemplateId}
              placeholder="Select template…"
              ariaLabel="Summary template"
              searchPlaceholder="Search templates…"
              emptyText="No templates match your search."
              options={templateOptions(templates)}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
                AI Provider
              </label>
              <Combobox
                value={selectedProvider}
                onChange={handleProviderChange}
                placeholder="Select provider…"
                ariaLabel="AI provider"
                searchPlaceholder="Search providers…"
                options={providers.map((p) => ({
                  value: p,
                  label: p.charAt(0).toUpperCase() + p.slice(1),
                }))}
              />
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
                Model
              </label>
              <Combobox
                value={selectedModel}
                onChange={handleModelChange}
                placeholder="Select model…"
                ariaLabel="Model"
                searchPlaceholder="Search models…"
                emptyText="No models match your search."
                options={modelOptions(providerModels)}
              />
            </div>
          </div>

          {selectedModelObj && selectedModelObj.reasoning && (
            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70 flex items-center gap-1.5">
                <ThinkingLabel />
              </label>
              <Combobox
                value={thinkingLevel}
                onChange={setThinkingLevel}
                placeholder="Select thinking level…"
                ariaLabel="Thinking reasoning level"
                options={thinkingLevelOptions(selectedModelObj.supportedThinkingLevels || ["low", "medium", "high"])}
              />
            </div>
          )}

          <div className="flex items-center gap-2 pt-1">
            <input
              type="checkbox"
              id="setPrimarySummary"
              checked={setPrimary}
              onChange={(e) => setSetPrimary(e.target.checked)}
              className="checkbox checkbox-primary"
            />
            <label htmlFor="setPrimarySummary" className="text-xs text-base-content cursor-pointer">
              Set as primary summary for this meeting
            </label>
          </div>

          <div className="flex justify-end gap-3 border-t border-base-300 pt-4">
            <button
              type="button"
              onClick={onClose}
              disabled={starting}
              className="btn btn-sm min-h-11"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={starting}
              className="btn btn-primary btn-sm inline-flex items-center gap-2 min-h-11"
            >
              {starting ? "Starting…" : "Generate Summary"}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}

interface RapidEnrollCompletion {
  label: string;
  targetName: string;
  updatedSegmentsCount: number;
  enrolledSamplesCount: number;
  sampleCount: number;
  notes: string[];
}

function RapidEnrollModal({
  meetingId,
  knownSpeakers,
  playingSegmentIndex,
  onPlaySample,
  onClose,
  onAssigned
}: {
  meetingId: string;
  knownSpeakers: SpeakerWithStats[];
  playingSegmentIndex: number | null;
  onPlaySample: (sample: { startMs: number; endMs: number; segmentIndex: number }) => void;
  onClose: () => void;
  onAssigned: () => void;
}): ReactElement {
  const [groups, setGroups] = useState<RapidEnrollCandidateGroup[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSamples, setSelectedSamples] = useState<Record<string, number[]>>({});
  const [targets, setTargets] = useState<Record<string, { speakerId: string; customName: string }>>({});
  // Labels with an in-flight assign request. Assigns run in the background
  // so the user can keep enrolling other voices while embeddings train.
  const [pendingLabels, setPendingLabels] = useState<Record<string, string>>({});
  const [failedLabels, setFailedLabels] = useState<Record<string, string>>({});
  const [completions, setCompletions] = useState<RapidEnrollCompletion[]>([]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch(`/api/meetings/${meetingId}/speakers/rapid-enroll-candidates?limit=5`);
        const data = (await res.json().catch(() => ({}))) as { groups?: RapidEnrollCandidateGroup[]; error?: string };
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        if (cancelled) return;
        const loaded = data.groups ?? [];
        setGroups(loaded);
        // Pre-check the top 3 enrollment-eligible samples per group.
        setSelectedSamples((prev) => {
          const next = { ...prev };
          for (const g of loaded) {
            if (!(g.label in next)) {
              const eligible = (g.samples ?? []).filter((s) => s.enrollEligible);
              const chosen = (eligible.length > 0 ? eligible : (g.samples ?? []).slice(0, 1)).slice(0, 3);
              next[g.label] = chosen.map((s) => s.segmentIndex);
            }
          }
          return next;
        });
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load enrollment candidates");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  const toggleSample = (label: string, segmentIndex: number) => {
    setSelectedSamples((prev) => {
      const current = prev[label] ?? [];
      return {
        ...prev,
        [label]: current.includes(segmentIndex)
          ? current.filter((i) => i !== segmentIndex)
          : [...current, segmentIndex]
      };
    });
  };

  const handleAssign = (group: RapidEnrollCandidateGroup) => {
    const target = targets[group.label] ?? { speakerId: "", customName: "" };
    let toSpeakerId: string | undefined;
    let toSpeakerName: string | undefined;
    if (target.speakerId && target.speakerId !== "new") {
      toSpeakerId = target.speakerId;
      toSpeakerName = knownSpeakers.find((s) => s.id === target.speakerId)?.name;
    } else {
      toSpeakerName = target.customName.trim();
    }
    if (!toSpeakerName && !toSpeakerId) {
      notify("warning", "Pick an existing person or enter a name first");
      return;
    }
    // Send samples in ranked order so the best clips enroll first.
    const segmentIndices = (group.samples ?? [])
      .map((s) => s.segmentIndex)
      .filter((idx) => (selectedSamples[group.label] ?? []).includes(idx));
    const displayName = toSpeakerName ?? knownSpeakers.find((s) => s.id === toSpeakerId)?.name ?? "speaker";

    // Optimistic update: remove the group immediately so the UI stays
    // interactive while enrollment trains in the background.
    setGroups((prev) => (prev ?? []).filter((g) => g.label !== group.label));
    setPendingLabels((prev) => ({ ...prev, [group.label]: displayName }));
    setFailedLabels((prev) => {
      if (!(group.label in prev)) return prev;
      const next = { ...prev };
      delete next[group.label];
      return next;
    });

    void fetch(`/api/meetings/${meetingId}/speakers/rapid-enroll`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ fromLabel: group.label, toSpeakerId, toSpeakerName, segmentIndices })
    })
      .then(async (res) => {
        const data = (await res.json().catch(() => ({}))) as RapidEnrollAssignResponse & { error?: string };
        if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
        setCompletions((prev) => [
          {
            label: group.label,
            targetName: data.speaker?.name ?? displayName,
            updatedSegmentsCount: data.updatedSegmentsCount ?? 0,
            enrolledSamplesCount: data.enrolledSamplesCount ?? 0,
            sampleCount: segmentIndices.length,
            notes: (data.sampleResults ?? [])
              .filter((r) => !r.voiceprintEnrolled && r.statusReason)
              .map((r) => r.statusReason as string)
          },
          ...prev
        ]);
        onAssigned();
      })
      .catch((err) => {
        // Restore the group so the user can retry.
        setGroups((prev) => (prev ? [...prev, group] : [group]));
        setFailedLabels((prev) => ({
          ...prev,
          [group.label]: err instanceof Error ? err.message : String(err)
        }));
      })
      .finally(() => {
        setPendingLabels((prev) => {
          if (!(group.label in prev)) return prev;
          const next = { ...prev };
          delete next[group.label];
          return next;
        });
      });
  };

  return (
    <Modal onClose={onClose} title={<><Zap className="h-4 w-4 shrink-0" aria-hidden="true" /> Rapid enroll</>} ariaLabel="Rapid enroll speakers" maxWidth="max-w-3xl">
      <p className="text-sm text-base-content/70">
        Speakers in this meeting that are unmatched or auto-matched but unverified, with their best voiceprint samples.
        Listen, check the good ones, then say who it is — every turn for that voice gets renamed and the checked samples
        train their voiceprint.
      </p>

      {error && <div className="rounded-xl border border-error bg-error/10 p-4 text-sm text-error">{error}</div>}
      {groups === null && !error && <p className="text-sm text-base-content/70">Finding candidate speakers…</p>}
      {groups !== null && groups.length === 0 && completions.length === 0 && Object.keys(pendingLabels).length === 0 && (
        <p className="rounded-xl border border-base-300 bg-base-100/50 p-4 text-sm text-base-content">
          🎉 Nothing to enroll — every speaker in this meeting is matched and verified.
        </p>
      )}

      {Object.entries(pendingLabels).map(([label, targetName]) => (
        <div key={`pending-${label}`} className="rounded-xl border border-warning/40 bg-warning/10 p-4 text-sm text-warning">
          <p className="font-semibold">
            ⏳ “{label}” is {targetName} — renaming now, enrolling voiceprint in the background…
          </p>
        </div>
      ))}

      {Object.entries(failedLabels).map(([label, message]) => (
        <div key={`failed-${label}`} className="rounded-xl border border-error bg-error/10 p-4 text-sm text-error">
          <p className="font-semibold">Failed to assign “{label}”: {message}</p>
          <p className="mt-1 text-xs">The voice has been restored to the list so you can retry.</p>
        </div>
      ))}

      {completions.map((c) => (
        <div key={`${c.label}-${c.targetName}`} className="rounded-xl border border-primary/40 bg-primary/10 p-4 text-sm text-primary">
          <p className="font-semibold">
            <Check className="h-4 w-4 shrink-0" aria-hidden="true" /> “{c.label}” is {c.targetName} — renamed {c.updatedSegmentsCount} turn{c.updatedSegmentsCount === 1 ? "" : "s"},
            enrolled {c.enrolledSamplesCount}/{c.sampleCount} sample{c.sampleCount === 1 ? "" : "s"}
          </p>
          {c.notes.map((note, i) => (
            <p key={i} className="mt-1 text-xs text-primary/80">{note}</p>
          ))}
        </div>
      ))}

      {(groups ?? []).map((group) => {
        const target = targets[group.label] ?? { speakerId: "", customName: "" };
        const targetName =
          target.speakerId && target.speakerId !== "new"
            ? knownSpeakers.find((s) => s.id === target.speakerId)?.name
            : target.customName.trim();
        return (
          <section key={group.label} className="space-y-3 rounded-xl border border-base-300 bg-base-100/50 p-4">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="text-base font-semibold text-base-content">“{group.label}”</h3>
              <span
                className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${
                  group.status === "unmatched"
                    ? "border-warning/40 bg-warning/10 text-warning"
                    : "border-accent/40 bg-accent/10 text-accent"
                }`}
              >
                {group.status === "unmatched" ? "Unmatched" : "Unverified match"}
              </span>
              <span className="text-xs text-base-content/50">
                {group.segmentCount} turn{group.segmentCount === 1 ? "" : "s"} • {formatDuration(group.totalDurationMs ?? 0)} of speech
              </span>
            </div>
            <p className="text-xs text-base-content/50">{group.reason}</p>

            <div className="space-y-2">
              {(group.samples ?? []).map((sample) => (
                <div key={sample.segmentIndex} className="flex items-start gap-2.5 rounded-lg border border-base-300 bg-base-200/70 p-2.5">
                  <input
                    type="checkbox"
                    checked={(selectedSamples[group.label] ?? []).includes(sample.segmentIndex)}
                    onChange={() => toggleSample(group.label, sample.segmentIndex)}
                    className="checkbox mt-1.5 shrink-0"
                    aria-label={`Enroll sample at ${formatTimeOffset(sample.startMs)}`}
                  />
                  <button
                    type="button"
                    onClick={() => onPlaySample(sample)}
                    title={playingSegmentIndex === sample.segmentIndex ? "Pause playback" : "Play this sample"}
                    className={`inline-flex min-h-[36px] shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-mono transition ${
                      playingSegmentIndex === sample.segmentIndex
                        ? "border border-primary/40 bg-primary/20 font-bold text-primary"
                        : "border border-base-300/70 bg-base-300/90 text-base-content/70 hover:text-primary"
                    }`}
                  >
                    <span className="text-[10px] inline-flex">{playingSegmentIndex === sample.segmentIndex ? <Pause className="h-3 w-3 shrink-0" aria-hidden="true" /> : <Play className="h-3 w-3 shrink-0" aria-hidden="true" />}</span>
                    <span>{formatTimeOffset(sample.startMs)}</span>
                  </button>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs leading-relaxed text-base-content line-clamp-2">{sample.text}</p>
                    <p className="mt-0.5 text-[10px] text-base-content/50">
                      {(sample.durationMs / 1000).toFixed(1)}s
                      {!sample.enrollEligible && <span className="text-warning"> • under 3s — too short to enroll (rename only)</span>}
                    </p>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <SpeakerCombobox
                value={target.speakerId}
                onChange={(next) =>
                  setTargets((prev) => ({ ...prev, [group.label]: { ...target, speakerId: next } }))
                }
                knownSpeakers={knownSpeakers}
                emptyLabel="-- Who is this? --"
                ariaLabel={`Who is ${group.label}?`}
              />
              {(target.speakerId === "new" || !target.speakerId) && (
                <input
                  type="text"
                  placeholder="e.g. Asad"
                  value={target.customName}
                  onChange={(e) =>
                    setTargets((prev) => ({ ...prev, [group.label]: { ...target, customName: e.target.value } }))
                  }
                  className="input input-md w-full"
                  aria-label="New person name"
                />
              )}
              <button
                type="button"
                onClick={() => handleAssign(group)}
                className="btn btn-primary inline-flex min-h-[44px] shrink-0 items-center justify-center disabled:cursor-not-allowed disabled:opacity-60"
              >
                {targetName ? `This is ${targetName}` : "Assign & enroll"}
              </button>
            </div>
          </section>
        );
      })}
    </Modal>
  );
}

function ReassignSpeakerModal({
  meetingId,
  fromLabel,
  segmentIndex,
  knownSpeakers,
  onClose,
  onSuccess,
  onSettled
}: {
  meetingId: string;
  fromLabel: string;
  segmentIndex?: number;
  knownSpeakers: SpeakerWithStats[];
  onClose: () => void;
  onSuccess: (optimistic?: { targetName: string; scope: "single" | "all"; segmentIndex?: number }) => void;
  onSettled?: () => void;
}): ReactElement {
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<string>("");
  const [customName, setCustomName] = useState<string>("");
  const [scope, setScope] = useState<"single" | "all">(segmentIndex !== undefined ? "single" : "all");
  const [adoptVoiceprint, setAdoptVoiceprint] = useState(true);
  const [saveToPeople, setSaveToPeople] = useState(true);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    let resolvedTargetName = "";
    let targetId: string | undefined = undefined;
    const localOnly = !saveToPeople;

    if (localOnly || selectedSpeakerId === "new" || !selectedSpeakerId) {
      if (!customName.trim()) {
        notify("warning", localOnly ? "Please enter a name to show in this transcript" : "Please enter a name for the new speaker");
        return;
      }
      resolvedTargetName = customName.trim();
    } else {
      targetId = selectedSpeakerId;
      const matched = knownSpeakers.find((s) => s.id === selectedSpeakerId);
      resolvedTargetName = matched ? matched.name : selectedSpeakerId;
    }

    const payload: any = {
      fromLabel,
      adoptVoiceprint: localOnly ? false : adoptVoiceprint,
      localOnly,
      segmentIndex,
      scope,
      toSpeakerName: resolvedTargetName,
      toSpeakerId: targetId
    };

    // 1. Immediately dismiss modal and optimistically update caller
    onSuccess({
      targetName: resolvedTargetName,
      scope,
      segmentIndex
    });

    // 2. Fire request asynchronously in background, then sync canonical state once it completes
    void fetch(`/api/meetings/${meetingId}/speakers/reassign`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(async (res) => {
        if (!res.ok) {
          const data = await res.json().catch(() => ({}));
          console.error("Async speaker reassignment error:", data.error || `HTTP ${res.status}`);
        }
      })
      .catch((err) => {
        console.error("Async speaker reassignment network error:", err);
      })
      .finally(() => {
        onSettled?.();
      });
  };

  return (
    <Modal onClose={onClose} title={<>Assign Speaker: "{fromLabel}"</>} ariaLabel="Assign speaker" maxWidth="max-w-md">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1.5">
            Name scope
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setSaveToPeople(true)}
              className={`rounded-lg border px-3 py-2 text-xs font-semibold transition text-left ${
                saveToPeople
                  ? "border-primary bg-primary/20 text-primary"
                  : "border-base-300 bg-base-100 text-base-content/70 hover:text-base-content"
              }`}
            >
              <div className="font-bold">Save to People</div>
              <div className="text-[10px] text-base-content/70 font-normal mt-0.5">Recognize in future meetings</div>
            </button>
            <button
              type="button"
              onClick={() => setSaveToPeople(false)}
              className={`rounded-lg border px-3 py-2 text-xs font-semibold transition text-left ${
                !saveToPeople
                  ? "border-primary bg-primary/20 text-primary"
                  : "border-base-300 bg-base-100 text-base-content/70 hover:text-base-content"
              }`}
            >
              <div className="font-bold">This transcript only</div>
              <div className="text-[10px] text-base-content/70 font-normal mt-0.5">One-off name, not saved</div>
            </button>
          </div>
        </div>

        {segmentIndex !== undefined && (
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1.5">
              Apply Scope
            </label>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setScope("single")}
                className={`rounded-lg border px-3 py-2 text-xs font-semibold transition text-left ${
                  scope === "single"
                    ? "border-primary bg-primary/20 text-primary"
                    : "border-base-300 bg-base-100 text-base-content/70 hover:text-base-content"
                }`}
              >
                <div className="font-bold">This turn only</div>
                <div className="text-[10px] text-base-content/70 font-normal mt-0.5">Correct single turn</div>
              </button>
              <button
                type="button"
                onClick={() => setScope("all")}
                className={`rounded-lg border px-3 py-2 text-xs font-semibold transition text-left ${
                  scope === "all"
                    ? "border-primary bg-primary/20 text-primary"
                    : "border-base-300 bg-base-100 text-base-content/70 hover:text-base-content"
                }`}
              >
                <div className="font-bold">All "{fromLabel}" turns</div>
                <div className="text-[10px] text-base-content/70 font-normal mt-0.5">Batch rename all</div>
              </button>
            </div>
          </div>
        )}

        {!saveToPeople ? (
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
              Display Name
            </label>
            <input
              type="text"
              placeholder="e.g. Fred"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="input input-md w-full"
            />
            <p className="mt-1.5 text-[11px] leading-snug text-base-content/50">
              Just relabels this transcript — nothing is added to People and no voiceprint is learned.
            </p>
          </div>
        ) : (
          <>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
                Select Person
              </label>
              <SpeakerCombobox
                value={selectedSpeakerId}
                onChange={setSelectedSpeakerId}
                knownSpeakers={knownSpeakers}
                emptyLabel="-- Create new or choose person --"
                createLabel="+ Create New Person"
                ariaLabel="Select person"
              />
            </div>

            {(selectedSpeakerId === "new" || !selectedSpeakerId) && (
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
                  Person Name
                </label>
                <input
                  type="text"
                  placeholder="e.g. Matt Cowger"
                  value={customName}
                  onChange={(e) => setCustomName(e.target.value)}
                  className="input input-md w-full"
                />
              </div>
            )}

            <div className="flex items-center gap-3 rounded-lg border border-base-300 bg-base-100/50 p-3">
              <input
                type="checkbox"
                id="voiceprint-toggle"
                checked={adoptVoiceprint}
                onChange={(e) => setAdoptVoiceprint(e.target.checked)}
                className="checkbox"
              />
              <label htmlFor="voiceprint-toggle" className="text-xs text-base-content leading-snug cursor-pointer">
                <span className="font-semibold text-base-content">Learn voiceprint from this audio</span>
                <br />
                Olive will automatically recognize and attribute this person in future meetings.
              </label>
            </div>
          </>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-base-content/70 hover:text-base-content"
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn btn-primary min-h-11"
          >
            Apply & Save
          </button>
        </div>
      </form>
    </Modal>
  );
}

function SplitSegmentModal({
  meetingId,
  segment,
  knownSpeakers,
  onClose,
  onSuccess
}: {
  meetingId: string;
  segment: {
    segmentIndex: number;
    speaker: string;
    text: string;
    startMs: number;
    endMs: number;
  };
  knownSpeakers: SpeakerWithStats[];
  onClose: () => void;
  onSuccess: () => void;
}): ReactElement {
  const words = useMemo(() => segment.text.split(/\s+/).filter(Boolean), [segment.text]);
  const [splitWordIndex, setSplitWordIndex] = useState<number>(Math.max(1, Math.floor(words.length / 2)));
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<string>("");
  const [customName, setCustomName] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);

  const durationMs = segment.endMs - segment.startMs;
  const ratio = splitWordIndex / Math.max(1, words.length);
  const splitTimestamp = Math.round(segment.startMs + ratio * durationMs);

  const part1Text = words.slice(0, splitWordIndex).join(" ");
  const part2Text = words.slice(splitWordIndex).join(" ");

  const part2SpeakerName =
    selectedSpeakerId === "new"
      ? customName || "New Speaker"
      : selectedSpeakerId
      ? knownSpeakers.find((s) => s.id === selectedSpeakerId)?.name || segment.speaker
      : segment.speaker;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      const payload: any = {
        segmentIndex: segment.segmentIndex,
        wordIndex: splitWordIndex
      };

      if (selectedSpeakerId === "new") {
        if (!customName.trim()) {
          notify("warning", "Please enter a name for the new speaker");
          setSubmitting(false);
          return;
        }
        payload.newSpeakerName = customName.trim();
      } else if (selectedSpeakerId) {
        payload.newSpeakerId = selectedSpeakerId;
        const matched = knownSpeakers.find((s) => s.id === selectedSpeakerId);
        if (matched) {
          payload.newSpeakerName = matched.name;
        }
      }

      const res = await fetch(`/api/meetings/${meetingId}/speakers/split-segment`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      onSuccess();
    } catch (err) {
      notify("error", `Failed to split segment: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal onClose={onClose} title={<><span className="inline-flex items-center gap-1.5"><Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split Turn:</span> <span className="text-primary">{segment.speaker}</span></>} ariaLabel="Split turn" maxWidth="max-w-2xl">
      <p className="text-xs text-base-content/70">Click any word to choose where to split this turn into two separate segments.</p>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Interactive Word Picker */}
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-2">
            Select Split Point (Click any word)
          </label>
          <div className="rounded-xl border border-base-300 bg-base-100 p-4 leading-loose flex flex-wrap gap-1.5 items-center max-h-56 overflow-y-auto">
            {words.map((word, idx) => {
              const isPart1 = idx < splitWordIndex;
              const isSplitTarget = idx === splitWordIndex;
              return (
                <Fragment key={idx}>
                  {isSplitTarget && (
                    <span className="badge badge-sm badge-warning gap-1 font-bold shadow-md animate-pulse">
                      <Scissors className="h-4 w-4 shrink-0" aria-hidden="true" /> Split Point
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setSplitWordIndex(idx === 0 ? 1 : idx)}
                    className={`rounded px-1.5 py-0.5 text-xs font-medium transition cursor-pointer ${
                      isPart1
                        ? "bg-primary/20 text-primary border border-primary/40 hover:bg-primary/30"
                        : "bg-accent/20 text-accent border border-accent/40 hover:bg-accent/30"
                    }`}
                    title={`Click to split before "${word}"`}
                  >
                    {word}
                  </button>
                </Fragment>
              );
            })}
          </div>
        </div>

        {/* Split Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Part 1 */}
          <div className="rounded-xl border border-primary/30 bg-primary/10 p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-primary uppercase tracking-wide">
                Part 1 ({segment.speaker})
              </span>
              <span className="font-mono text-base-content/70 text-[11px]">
                {formatTimeOffset(segment.startMs)} → {formatTimeOffset(splitTimestamp)}
              </span>
            </div>
            <p className="text-xs text-base-content line-clamp-4 italic">
              "{part1Text || "(empty)"}"
            </p>
          </div>

          {/* Part 2 */}
          <div className="rounded-xl border border-accent/30 bg-accent/10 p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-accent uppercase tracking-wide">
                Part 2 ({part2SpeakerName})
              </span>
              <span className="font-mono text-base-content/70 text-[11px]">
                {formatTimeOffset(splitTimestamp)} → {formatTimeOffset(segment.endMs)}
              </span>
            </div>
            <p className="text-xs text-base-content line-clamp-4 italic">
              "{part2Text || "(empty)"}"
            </p>
          </div>
        </div>

        {/* Reassign Part 2 Speaker */}
        <div className="space-y-3 pt-1 border-t border-base-300">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
              Speaker for Part 2
            </label>
            <SpeakerCombobox
              value={selectedSpeakerId}
              onChange={setSelectedSpeakerId}
              knownSpeakers={knownSpeakers}
              emptyLabel={`Keep as "${segment.speaker}"`}
              createLabel="+ Create New Person"
              ariaLabel="Speaker for Part 2"
            />
          </div>

          {selectedSpeakerId === "new" && (
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
                New Person Name
              </label>
              <input
                type="text"
                placeholder="e.g. Harrison"
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                className="input input-md w-full focus:border-accent"
              />
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-base-content/70 hover:text-base-content"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting || !part1Text || !part2Text}
            className="btn btn-warning min-h-11"
          >
            {submitting ? "Splitting…" : "Confirm & Split Turn"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function UploadAudioModal({
  onClose,
  onUploaded
}: {
  onClose: () => void;
  onUploaded: (meetingId: string) => void;
}): ReactElement {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [autoTranscribe, setAutoTranscribe] = useState(true);
  const [selectedEngine, setSelectedEngine] = useState<string>("local:cohere");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetch("/api/config")
      .then((res) => (res.ok ? res.json() : null))
      .then((config) => {
        if (config?.transcriptionProvider === "speechmatics" || config?.transcriptionProvider === "assemblyai" || config?.transcriptionProvider === "muse") {
          setSelectedEngine(config.transcriptionProvider);
        } else {
          setSelectedEngine("local:cohere");
        }
      })
      .catch(() => {});
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      if (!title.trim()) {
        setTitle(selected.name.replace(/\.[^/.]+$/, "").replace(/[_-]/g, " "));
      }
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      notify("warning", "Please select an audio file to upload");
      return;
    }

    setUploading(true);
    setError(null);

    const isLocal = selectedEngine.startsWith("local");
    const formData = new FormData();
    formData.append("file", file);
    if (title.trim()) {
      formData.append("title", title.trim());
    }
    formData.append("autoTranscribe", String(autoTranscribe));
    formData.append("provider", isLocal ? "local" : selectedEngine);
    if (isLocal) {
      formData.append("modelId", "onnx-community/cohere-transcribe-03-2026-ONNX");
    }

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const data = (await res.json()) as { meetingId: string; deduped: boolean };
      onUploaded(data.meetingId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <Modal onClose={onClose} title={"Upload Audio Recording"} ariaLabel="Upload audio recording" maxWidth="max-w-md">
      {error && <div className="rounded-xl border border-error bg-error/10 p-3 text-xs text-error">{error}</div>}

      <form onSubmit={handleUpload} className="space-y-4 text-sm">
        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
            Audio File (.m4a, .mp3, .wav, .aac, .ogg, .flac)
          </label>
          <input
            type="file"
            required
            accept="audio/*,.m4a,.mp3,.wav,.aac,.ogg,.flac,.webm"
            onChange={handleFileChange}
            className="file-input w-full p-2.5 text-base-content hover:file:bg-base-300"
          />
          {file && (
            <p className="mt-1 text-xs text-base-content/70">
              Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        <div>
          <label className="block text-xs font-semibold uppercase tracking-wider text-base-content/70 mb-1">
            Meeting / Recording Title
          </label>
          <input
            type="text"
            placeholder="e.g. Planning Session with Team"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="input input-md w-full"
          />
        </div>

        <div className="space-y-2 rounded-lg border border-base-300 bg-base-100/50 p-3">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="auto-transcribe"
              checked={autoTranscribe}
              onChange={(e) => setAutoTranscribe(e.target.checked)}
              className="checkbox"
            />
            <label htmlFor="auto-transcribe" className="text-xs font-semibold text-base-content cursor-pointer">
              Automatically transcribe after upload
            </label>
          </div>

          {autoTranscribe && (
            <div className="pt-2 pl-6">
              <label className="block text-xs text-base-content/70 mb-1">Transcription Engine:</label>
              <Combobox
                value={selectedEngine}
                onChange={setSelectedEngine}
                placeholder="Select engine…"
                ariaLabel="Transcription engine"
                searchPlaceholder="Search engines…"
                options={ENGINE_OPTIONS}
              />
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm text-base-content/70 hover:text-base-content"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={uploading || !file}
            className="btn btn-primary min-h-11"
          >
            {uploading ? "Ingesting…" : "Upload & Save"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function SpeakersListView({
  speakers,
  onRefresh,
  onSelectMeeting
}: {
  speakers: SpeakerWithStats[];
  onRefresh: () => void;
  onSelectMeeting: (meetingId: string) => void;
}): ReactElement {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingName, setEditingName] = useState("");
  const [mergingId, setMergingId] = useState<string | null>(null);
  const [targetMergeId, setTargetMergeId] = useState("");
  const [syncing, setSyncing] = useState(false);

  const handleBackfill = async () => {
    setSyncing(true);
    try {
      const res = await fetch("/api/speakers/backfill", { method: "POST" });
      if (res.ok) {
        onRefresh();
      }
    } catch {
      notify("error", "Failed to backfill voiceprints");
    } finally {
      setSyncing(false);
    }
  };

  const handleRename = async (id: string) => {
    if (!editingName.trim()) return;
    try {
      const res = await fetch(`/api/speakers/${id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: editingName.trim() })
      });
      if (res.ok) {
        setEditingId(null);
        onRefresh();
      }
    } catch {
      notify("error", "Failed to rename speaker");
    }
  };

  const handleMerge = async (sourceId: string) => {
    if (!targetMergeId) return;
    try {
      const res = await fetch("/api/speakers/merge", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ sourceSpeakerId: sourceId, targetSpeakerId: targetMergeId })
      });
      if (res.ok) {
        setMergingId(null);
        setTargetMergeId("");
        onRefresh();
      }
    } catch {
      notify("error", "Failed to merge speaker");
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!(await confirmAction({ title: "Delete speaker", message: `Are you sure you want to delete speaker "${name}"?`, confirmLabel: "Delete", danger: true }))) return;
    try {
      const res = await fetch(`/api/speakers/${id}`, { method: "DELETE" });
      if (res.ok) {
        onRefresh();
      }
    } catch {
      notify("error", "Failed to delete speaker");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">People & Voiceprints</h1>
          <p className="mt-1 text-sm text-base-content/70">
            Recognized speakers and their persistent voice identifiers learned from meetings.
          </p>
        </div>
        <div>
          <button
            type="button"
            disabled={syncing}
            onClick={() => void handleBackfill()}
            className="rounded-lg border border-base-300 bg-base-200 px-3.5 py-1.5 text-xs font-semibold text-base-content hover:border-primary hover:text-primary transition disabled:opacity-50"
          >
            {syncing ? "Syncing Voiceprints…" : "Sync & Cross-fill Voiceprints"}
          </button>
        </div>
      </div>

      {speakers.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-base-300 bg-base-200/40 py-16 text-center">
          <p className="text-base-content/70">
            No people enrolled yet. Open any meeting transcript and click a speaker label (e.g. S1) to assign a person and learn their voice.
          </p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {speakers.map((s) => {
            return (
              <div
                key={s.id}
                className="rounded-xl border border-base-300 bg-base-200/90 p-5 space-y-4 flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-start justify-between">
                    {editingId === s.id ? (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          className="input input-md w-full min-w-0 rounded"
                        />
                        <button
                          type="button"
                          onClick={() => void handleRename(s.id)}
                          className="btn btn-primary min-h-[44px] shrink-0 rounded sm:min-h-0 sm:px-2 sm:py-1 sm:text-xs"
                        >
                          Save
                        </button>
                      </div>
                    ) : (
                      <h2 className="text-lg font-bold text-base-content flex items-center gap-2">
                        {s.name}
                        <button
                          type="button"
                          onClick={() => {
                            setEditingId(s.id);
                            setEditingName(s.name);
                          }}
                          className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center text-xs text-base-content/50 hover:text-base-content sm:min-h-0 sm:min-w-0"
                        >
                          <Pencil className="h-4 w-4 shrink-0" aria-hidden="true" />
                        </button>
                      </h2>
                    )}
                    <div className="flex flex-col gap-1 items-end">
                      <SpeakerStatusBadges speaker={s} />
                    </div>
                  </div>

                  <div className="mt-3 space-y-1 text-xs text-base-content/70">
                    <p>Participated in: <span className="font-semibold text-base-content">{s.meetingCount} meeting(s)</span></p>
                    {s.enrollmentClipPaths?.length > 0 && (
                      <p>
                        Stored clips: <span className="font-semibold text-base-content">{s.enrollmentClipPaths.length}/8</span>
                        {s.enrollmentClipPaths.length >= 8 && (
                          <span className="text-base-content/50 ml-1">(Cap reached — auto-rolling)</span>
                        )}
                      </p>
                    )}
                    {s.enrolledAt && (
                      <p>Last learned: {formatDate(s.enrolledAt)}</p>
                    )}
                  </div>
                </div>

                {mergingId === s.id ? (
                  <div className="border-t border-base-300 pt-3 space-y-2">
                    <label className="text-xs text-base-content/70">Merge {s.name} into:</label>
                    <Combobox
                      value={targetMergeId}
                      onChange={setTargetMergeId}
                      placeholder="Select target…"
                      ariaLabel={`Merge ${s.name} into`}
                      searchPlaceholder="Search people…"
                      emptyText="No people match your search."
                      options={[
                        { value: "", label: "-- Choose target --" },
                        ...speakers
                          .filter((candidate) => candidate.id !== s.id)
                          .map((candidate) => ({
                            value: candidate.id,
                            label: candidate.name,
                          })),
                      ]}
                    />
                    <div className="flex gap-2 justify-end">
                      <button
                        type="button"
                        onClick={() => setMergingId(null)}
                        className="min-h-[44px] px-3 text-sm text-base-content/70 sm:min-h-0 sm:px-2 sm:py-1 sm:text-xs"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleMerge(s.id)}
                        className="btn btn-primary min-h-[44px] rounded sm:min-h-0 sm:px-2 sm:py-1 sm:text-xs"
                      >
                        Confirm Merge
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-stretch gap-2 border-t border-base-300/80 pt-3 text-xs text-base-content/70">
                    <button
                      type="button"
                      onClick={() => setMergingId(s.id)}
                      className="btn flex min-h-[44px] flex-1 items-center justify-center hover:border-primary hover:text-primary sm:min-h-0 sm:py-1.5 sm:text-xs"
                    >
                      Merge into…
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDelete(s.id, s.name)}
                      className="btn flex min-h-[44px] flex-1 items-center justify-center hover:border-error/50 hover:text-error sm:min-h-0 sm:py-1.5 sm:text-xs"
                    >
                      Delete
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TemplatesListView(): ReactElement {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [editingTemplate, setEditingTemplate] = useState<Template | null>(null);
  const [cloningTemplate, setCloningTemplate] = useState<Template | null>(null);
  const [isCreating, setIsCreating] = useState(false);

  const fetchTemplates = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/templates");
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: Failed to fetch templates`);
      }
      const data = (await res.json()) as { templates: Template[] };
      setTemplates(data.templates);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchTemplates();
  }, []);

  const handleSetDefault = async (id: string) => {
    try {
      const res = await fetch(`/api/templates/${id}/default`, { method: "POST" });
      if (!res.ok) {
        throw new Error("Failed to set default template");
      }
      await fetchTemplates();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!(await confirmAction({ title: "Delete template", message: `Are you sure you want to delete template "${name}"?`, confirmLabel: "Delete", danger: true }))) {
      return;
    }
    try {
      const res = await fetch(`/api/templates/${id}`, { method: "DELETE" });
      if (!res.ok) {
        throw new Error("Failed to delete template");
      }
      await fetchTemplates();
    } catch (err) {
      notify("error", err instanceof Error ? err.message : String(err));
    }
  };

  const filteredTemplates = templates.filter((t) => {
    if (!search.trim()) return true;
    const query = search.toLowerCase();
    return (
      t.name.toLowerCase().includes(query) ||
      (t.description && t.description.toLowerCase().includes(query)) ||
      t.userPrompt.toLowerCase().includes(query)
    );
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Summary Templates</h1>
          <p className="text-sm text-base-content/70">
            Customizable prompt templates for generating structured meeting summaries & action items
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setEditingTemplate(null);
            setCloningTemplate(null);
            setIsCreating(true);
          }}
          className="btn btn-primary inline-flex min-h-[44px] items-center justify-center gap-2 sm:min-h-0"
        >
          <span>+</span>
          <span>New Template</span>
        </button>
      </div>

      <div className="flex items-center gap-3">
        <input
          type="text"
          placeholder="Filter templates…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input input-md w-full max-w-sm bg-base-200/80 px-3.5"
        />
      </div>

      {error && (
        <div className="alert alert-soft alert-error p-4 text-sm text-error">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-12 text-center text-sm text-base-content/50">Loading templates…</div>
      ) : filteredTemplates.length === 0 ? (
        <div className="rounded-xl border border-base-300 bg-base-200/30 p-12 text-center text-sm text-base-content/50">
          No templates found.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {filteredTemplates.map((t) => (
            <div
              key={t.id}
              className={`flex flex-col justify-between rounded-xl border p-5 transition ${
                t.isDefault
                  ? "border-primary/40 bg-base-200/80 ring-1 ring-primary/20"
                  : "border-base-300 bg-base-200/50 hover:border-base-300"
              }`}
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-2">
                  <h3 className="font-semibold text-base-content">{t.name}</h3>
                  <div className="flex items-center gap-1.5 shrink-0">
                    {t.isDefault && (
                      <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[11px] font-medium text-primary border border-primary/20">
                        Default
                      </span>
                    )}
                    {t.isBuiltin ? (
                      <span className="rounded-full bg-base-300 px-2 py-0.5 text-[11px] font-medium text-base-content/70">
                        Built-in
                      </span>
                    ) : (
                      <span className="badge badge-sm badge-accent badge-soft font-medium">
                        Custom
                      </span>
                    )}
                  </div>
                </div>

                {t.description && (
                  <p className="text-xs text-base-content/70 leading-relaxed">{t.description}</p>
                )}

                {t.systemPrompt && (
                  <div className="space-y-1">
                    <span className="text-[10px] font-medium tracking-wider uppercase text-base-content/50">
                      System Instructions
                    </span>
                    <p className="text-xs text-base-content bg-base-100/60 rounded p-2 border border-base-300/80 line-clamp-2">
                      {t.systemPrompt}
                    </p>
                  </div>
                )}

                <div className="space-y-1">
                  <span className="text-[10px] font-medium tracking-wider uppercase text-base-content/50">
                    Prompt Template
                  </span>
                  <pre className="max-h-48 overflow-y-auto rounded-lg border border-base-300/80 bg-base-100/80 p-3 font-mono text-[11px] text-base-content whitespace-pre-wrap leading-relaxed">
                    {t.userPrompt}
                  </pre>
                </div>
              </div>

              <div className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t border-base-300/80 pt-3 text-xs">
                <div>
                  {!t.isDefault && (
                    <button
                      type="button"
                      onClick={() => void handleSetDefault(t.id)}
                      className="btn btn-primary btn-soft min-h-[44px] border border-primary/30 sm:min-h-[36px] sm:text-xs"
                    >
                      Make Default
                    </button>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {t.isBuiltin ? (
                    <button
                      type="button"
                      onClick={() => {
                        setCloningTemplate(t);
                        setEditingTemplate(null);
                        setIsCreating(true);
                      }}
                      className="btn min-h-[44px] sm:min-h-[36px] sm:text-xs"
                    >
                      Duplicate & Edit
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        onClick={() => {
                          setEditingTemplate(t);
                          setCloningTemplate(null);
                          setIsCreating(true);
                        }}
                        className="btn min-h-[44px] sm:min-h-[36px] sm:text-xs"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => void handleDelete(t.id, t.name)}
                        className="btn min-h-[44px] hover:border-error/50 hover:text-error sm:min-h-[36px] sm:text-xs"
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {isCreating && (
        <TemplateEditorModal
          template={editingTemplate}
          initialClone={cloningTemplate}
          onClose={() => {
            setIsCreating(false);
            setEditingTemplate(null);
            setCloningTemplate(null);
          }}
          onSaved={() => {
            setIsCreating(false);
            setEditingTemplate(null);
            setCloningTemplate(null);
            void fetchTemplates();
          }}
        />
      )}
    </div>
  );
}

function TemplateEditorModal({
  template,
  initialClone,
  onClose,
  onSaved
}: {
  template: Template | null;
  initialClone: Template | null;
  onClose: () => void;
  onSaved: () => void;
}): ReactElement {
  const isEditing = Boolean(template && !initialClone);
  const [name, setName] = useState(
    template ? template.name : initialClone ? `${initialClone.name} (Custom)` : ""
  );
  const [description, setDescription] = useState(
    template ? template.description || "" : initialClone ? initialClone.description || "" : ""
  );
  const [systemPrompt, setSystemPrompt] = useState(
    template ? template.systemPrompt : initialClone ? initialClone.systemPrompt : ""
  );
  const [userPrompt, setUserPrompt] = useState(
    template
      ? template.userPrompt
      : initialClone
      ? initialClone.userPrompt
      : `# Meeting Summary: {{title}}\n**Date:** {{date}}\n**Participants:** {{speakers}}\n\n## Summary\n\n## Action Items\n- [ ] \n\n---\n### Transcript\n{{transcript}}`
  );
  const [isDefault, setIsDefault] = useState(template ? template.isDefault : false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const promptTextareaRef = useRef<HTMLTextAreaElement>(null);

  const insertVariable = (tag: string) => {
    const textarea = promptTextareaRef.current;
    if (!textarea) {
      setUserPrompt((prev) => `${prev} ${tag}`);
      return;
    }
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const nextVal = userPrompt.substring(0, start) + tag + userPrompt.substring(end);
    setUserPrompt(nextVal);
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + tag.length, start + tag.length);
    }, 0);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Template name is required");
      return;
    }
    if (!userPrompt.trim()) {
      setError("User prompt template is required");
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const url = isEditing ? `/api/templates/${template!.id}` : "/api/templates";
      const method = isEditing ? "PATCH" : "POST";
      const payload = {
        name: name.trim(),
        description: description.trim() || null,
        systemPrompt: systemPrompt.trim(),
        userPrompt: userPrompt.trim(),
        isDefault
      };

      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const data = (await res.json()) as { error?: string };
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  };

  return (
    <Modal onClose={onClose} title={isEditing ? "Edit Template" : initialClone ? `Customize "${initialClone.name}"` : "New Summary Template"} ariaLabel="Summary template editor" maxWidth="max-w-2xl">
      {error && (
        <div className="alert alert-soft alert-error p-3 text-xs text-error">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1">
          <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
            Template Name *
          </label>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Sprint Retrospective"
            className="input input-md w-full"
          />
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
            Description
          </label>
          <input
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. Highlights wins, blockers, and next sprint commitments"
            className="input input-md w-full"
          />
        </div>

        <div className="space-y-1">
          <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
            System Instructions (Optional)
          </label>
          <textarea
            rows={2}
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            placeholder="e.g. You are an expert agile coach summarizing a team retrospective."
            className="textarea w-full textarea-md font-mono"
          />
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
              User Prompt Template *
            </label>
            <div className="flex items-center gap-1">
              <span className="text-[11px] text-base-content/50 mr-1">Insert tags:</span>
              {["{{title}}", "{{date}}", "{{speakers}}", "{{transcript}}"].map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => insertVariable(tag)}
                  className="rounded bg-base-300 px-1.5 py-0.5 text-[11px] font-mono text-primary hover:bg-base-300 transition"
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
          <textarea
            ref={promptTextareaRef}
            rows={9}
            required
            value={userPrompt}
            onChange={(e) => setUserPrompt(e.target.value)}
            className="textarea w-full textarea-md font-mono"
          />
        </div>

        <div className="flex items-center gap-2 pt-1">
          <input
            type="checkbox"
            id="isDefaultCheckbox"
            checked={isDefault}
            onChange={(e) => setIsDefault(e.target.checked)}
            className="checkbox checkbox-primary"
          />
          <label htmlFor="isDefaultCheckbox" className="text-xs text-base-content cursor-pointer">
            Set as default summary template
          </label>
        </div>

        <div className="flex justify-end gap-3 border-t border-base-300 pt-4">
          <button
            type="button"
            onClick={onClose}
            disabled={saving}
            className="btn btn-sm min-h-11"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={saving}
            className="btn btn-primary btn-sm inline-flex items-center gap-2 min-h-11"
          >
            {saving ? "Saving…" : isEditing ? "Save Changes" : "Create Template"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function LlmSettingsModal({
  onClose,
  onSaved
}: {
  onClose: () => void;
  onSaved: () => void;
}): ReactElement {
  const [providers, setProviders] = useState<LlmProviderSummary[]>([]);
  const [models, setModels] = useState<LlmModelCatalogItem[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("google");
  const [selectedModel, setSelectedModel] = useState<string>("gemini-2.5-flash");
  const [thinkingLevel, setThinkingLevel] = useState<string>("off");
  const [apiKey, setApiKey] = useState<string>("");
  const [baseUrl, setBaseUrl] = useState<string>("");
  const [savedProviders, setSavedProviders] = useState<Record<string, { hasApiKey?: boolean; baseUrl?: string }>>({});
  const [localClusteringThreshold, setLocalClusteringThreshold] = useState<number>(0.50);
  const [localSimilarityThreshold, setLocalSimilarityThreshold] = useState<number>(0.45);
  const [localAsrModel, setLocalAsrModel] = useState<string>("onnx-community/cohere-transcribe-03-2026-ONNX");
  const [transcriptionProvider, setTranscriptionProvider] = useState<"speechmatics" | "local" | "assemblyai" | "muse">("local");
  const [plaudRetranscribeMode, setPlaudRetranscribeMode] = useState<"off" | "unnamed" | "always">("off");
  const [hfToken, setHfToken] = useState<string>("");
  const [speechmaticsApiKey, setSpeechmaticsApiKey] = useState<string>("");
  const [assemblyaiApiKey, setAssemblyaiApiKey] = useState<string>("");
  const [museApiKey, setMuseApiKey] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [refreshingCatalog, setRefreshingCatalog] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; message?: string; durationMs?: number } | null>(null);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [configRes, provRes, modelsRes, appCfgRes] = await Promise.all([
        fetch("/api/llm/config"),
        fetch("/api/llm/providers"),
        fetch("/api/llm/models"),
        fetch("/api/config")
      ]);

      if (configRes.ok) {
        const config = await configRes.json();
        setSelectedProvider(config.defaultProvider || "google");
        setSelectedModel(config.defaultModel || "gemini-2.5-flash");
        if (config.defaultThinkingLevel) {
          setThinkingLevel(config.defaultThinkingLevel);
        }
        setSavedProviders(config.providers || {});
        const provConfig = config.providers?.[config.defaultProvider || "google"];
        setBaseUrl(provConfig?.baseUrl || "");
      }

      if (appCfgRes.ok) {
        const appCfg = await appCfgRes.json();
        if (appCfg.localClusteringThreshold !== undefined) {
          setLocalClusteringThreshold(appCfg.localClusteringThreshold);
        }
        if (appCfg.localSimilarityThreshold !== undefined) {
          setLocalSimilarityThreshold(appCfg.localSimilarityThreshold);
        }
        if (appCfg.localAsrModel !== undefined) {
          setLocalAsrModel(appCfg.localAsrModel);
        }
        if (appCfg.transcriptionProvider === "speechmatics" || appCfg.transcriptionProvider === "local" || appCfg.transcriptionProvider === "assemblyai" || appCfg.transcriptionProvider === "muse") {
          setTranscriptionProvider(appCfg.transcriptionProvider);
        }
        if (appCfg.alwaysRetranscribePlaud) {
          setPlaudRetranscribeMode("always");
        } else if (appCfg.retranscribePlaudWhenUnnamed) {
          setPlaudRetranscribeMode("unnamed");
        } else {
          setPlaudRetranscribeMode("off");
        }
        if (typeof appCfg.hfToken === "string") {
          setHfToken(appCfg.hfToken);
        }
        if (typeof appCfg.speechmaticsApiKey === "string") {
          setSpeechmaticsApiKey(appCfg.speechmaticsApiKey);
        }
        if (typeof appCfg.assemblyaiApiKey === "string") {
          setAssemblyaiApiKey(appCfg.assemblyaiApiKey);
        }
        if (typeof appCfg.museApiKey === "string") {
          setMuseApiKey(appCfg.museApiKey);
        }
      }

      if (provRes.ok) {
        const pData = await provRes.json();
        setProviders(pData.providers || []);
      }

      if (modelsRes.ok) {
        const mData = await modelsRes.json();
        setModels(mData.models || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchData();
  }, []);

  const handleProviderChange = (pId: string) => {
    setSelectedProvider(pId);
    setApiKey("");
    setBaseUrl(savedProviders[pId]?.baseUrl || "");
    setTestResult(null);
    const providerModels = models.filter((m) => m.provider === pId);
    if (providerModels.length > 0) {
      setSelectedModel(providerModels[0].id);
      const supported = providerModels[0].supportedThinkingLevels || ["low", "medium", "high"];
      setThinkingLevel(supported.includes("off") ? "off" : supported[0] || "low");
    }
  };

  const handleModelChange = (mId: string) => {
    setSelectedModel(mId);
    setTestResult(null);
    const m = models.find((mod) => mod.provider === selectedProvider && mod.id === mId);
    if (m?.supportedThinkingLevels) {
      if (!m.supportedThinkingLevels.includes(thinkingLevel as any)) {
        setThinkingLevel(m.supportedThinkingLevels[0] || "low");
      }
    }
  };

  const handleRefreshCatalog = async () => {
    setRefreshingCatalog(true);
    try {
      const res = await fetch("/api/llm/refresh-catalog", { method: "POST" });
      if (res.ok) {
        const mRes = await fetch("/api/llm/models");
        if (mRes.ok) {
          const mData = await mRes.json();
          setModels(mData.models || []);
        }
      }
    } catch (err) {
      notify("error", "Failed to refresh catalog: " + (err instanceof Error ? err.message : String(err)));
    } finally {
      setRefreshingCatalog(false);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/llm/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: selectedProvider,
          model: selectedModel,
          apiKey: apiKey.trim() || undefined,
          baseUrl: baseUrl.trim() || undefined,
          thinkingLevel: selectedModelObj?.reasoning ? thinkingLevel : undefined
        })
      });

      const data = await res.json();
      if (res.ok && data.ok) {
        setTestResult({
          ok: true,
          message: `Connected successfully! Response: "${data.response}"`,
          durationMs: data.durationMs
        });
      } else {
        setTestResult({
          ok: false,
          message: data.error || "Connection failed"
        });
      }
    } catch (err) {
      setTestResult({
        ok: false,
        message: err instanceof Error ? err.message : String(err)
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const payload: Record<string, any> = {
        defaultProvider: selectedProvider,
        defaultModel: selectedModel,
        defaultThinkingLevel: selectedModelObj?.reasoning ? thinkingLevel : "off",
        providers: {
          [selectedProvider]: {
            ...(apiKey.trim() ? { apiKey: apiKey.trim() } : {}),
            baseUrl: baseUrl.trim()
          }
        }
      };

      const [resLlm, resApp] = await Promise.all([
        fetch("/api/llm/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }),
        fetch("/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            localClusteringThreshold,
            localSimilarityThreshold,
            localAsrModel,
            transcriptionProvider,
            retranscribePlaudWhenUnnamed: plaudRetranscribeMode === "unnamed",
            alwaysRetranscribePlaud: plaudRetranscribeMode === "always",
            hfToken,
            speechmaticsApiKey,
            assemblyaiApiKey,
            museApiKey
          })
        })
      ]);

      if (!resLlm.ok) {
        const data = await resLlm.json();
        throw new Error(data.error || "Failed to save LLM configuration");
      }

      if (!resApp.ok) {
        const data = await resApp.json();
        throw new Error(data.error || "Failed to save audio settings");
      }

      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSaving(false);
    }
  };

  const providerModels = models.filter((m) => m.provider === selectedProvider);

  const selectedModelObj = models.find((m) => m.provider === selectedProvider && m.id === selectedModel);
  const currentProviderSummary = providers.find((p) => p.id === selectedProvider);

  return (
    <Modal onClose={onClose} title={"LLM Provider & Model Settings"} ariaLabel="LLM provider and model settings" maxWidth="max-w-2xl">
      <p className="text-xs text-base-content/70">Powered by pi-ai unified multi-provider model catalog</p>
      {error && (
        <div className="alert alert-soft alert-error p-3 text-xs text-error">
          {error}
        </div>
      )}

      {loading ? (
        <div className="p-8 text-center text-sm text-base-content/70">Loading catalog…</div>
      ) : (
        <form onSubmit={handleSave} className="space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
                AI Provider
              </label>
              <Combobox
                value={selectedProvider}
                onChange={handleProviderChange}
                placeholder="Select provider…"
                ariaLabel="AI provider"
                searchPlaceholder="Search providers…"
                options={llmProviderOptions(providers)}
              />
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-base-content/50">
                  Status: {currentProviderSummary?.isConfigured ? "🟢 Key Configured" : "🟡 Needs API Key"}
                </span>
                <button
                  type="button"
                  onClick={() => void handleRefreshCatalog()}
                  disabled={refreshingCatalog}
                  className="text-[11px] text-primary hover:underline disabled:opacity-50"
                >
                  {refreshingCatalog ? "Syncing…" : "↻ Sync Web Catalog"}
                </button>
              </div>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
                Model
              </label>
              <Combobox
                value={selectedModel}
                onChange={handleModelChange}
                placeholder="Select model…"
                ariaLabel="Model"
                searchPlaceholder="Search models…"
                emptyText="No models match your search."
                options={modelOptions(providerModels)}
              />
            </div>
          </div>

          {selectedModelObj && selectedModelObj.reasoning && (
            <div className="space-y-1">
              <label className="text-xs font-medium uppercase tracking-wider text-base-content/70 flex items-center gap-1.5">
                <ThinkingLabel />
              </label>
              <Combobox
                value={thinkingLevel}
                onChange={setThinkingLevel}
                placeholder="Select thinking level…"
                ariaLabel="Thinking reasoning level"
                options={thinkingLevelOptions(selectedModelObj.supportedThinkingLevels || ["low", "medium", "high"])}
              />
            </div>
          )}

          <div className="space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
              API Key Override (Optional)
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={currentProviderSummary?.isConfigured ? "Using configured environment key (leave blank to keep)" : "Enter provider API key…"}
              className="input input-md w-full placeholder-base-content/50 font-mono"
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-base-content/70">
              Base URL Override (Optional)
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="e.g. http://localhost:11434/v1 for Ollama, or custom LiteLLM proxy"
              className="input input-md w-full placeholder-base-content/50 font-mono"
            />
          </div>

          {/* Local Audio Diarization & Speaker Recognition Settings */}
          <div className="rounded-xl border border-base-300 bg-base-100/60 p-4 space-y-3.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-base-content">
                <Mic className="h-4 w-4 shrink-0" aria-hidden="true" /> Local ASR & Speaker Recognition
              </span>
              <span className="text-[11px] text-base-content/50 font-mono">Engine & Sensitivity Tuning</span>
            </div>

            {/* Hugging Face Token for Gated Diarization Models */}
            <div className="space-y-1.5">
              <label
                htmlFor="hf-token-input"
                className="text-xs text-base-content font-medium flex items-center justify-between"
              >
                <span>Hugging Face Access Token (pyannote community-1):</span>
                <span className="text-[10px] text-base-content/50 font-mono">HF_TOKEN env overrides</span>
              </label>
              <input
                id="hf-token-input"
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_..."
                className="input input-md w-full placeholder-base-content/50 font-mono"
              />
              <p className="text-[11px] text-base-content/50 leading-tight">
                Used to download gated pyannote diarization models. Accept terms on huggingface.co/pyannote/speaker-diarization-community-1.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 pt-1">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="clustering-threshold-slider" className="text-base-content font-medium">
                    Within-Meeting Clustering:
                  </label>
                  <span className="font-mono font-bold text-primary">
                    {localClusteringThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  id="clustering-threshold-slider"
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.01"
                  value={localClusteringThreshold}
                  onChange={(e) => setLocalClusteringThreshold(parseFloat(e.target.value))}
                  className="range range-sm w-full"
                />
                <p className="text-[11px] text-base-content/50 leading-tight">
                  Controls separation of distinct voices within a meeting (recommended: 0.50).
                </p>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="similarity-threshold-slider" className="text-base-content font-medium">
                    Enrolled Voiceprint Matching:
                  </label>
                  <span className="font-mono font-bold text-accent">
                    {localSimilarityThreshold.toFixed(2)}
                  </span>
                </div>
                <input
                  id="similarity-threshold-slider"
                  type="range"
                  min="0.10"
                  max="0.95"
                  step="0.01"
                  value={localSimilarityThreshold}
                  onChange={(e) => setLocalSimilarityThreshold(parseFloat(e.target.value))}
                  className="range range-sm w-full"
                />
                <p className="text-[11px] text-base-content/50 leading-tight">
                  Minimum cosine similarity required to match speech against enrolled WeSpeaker voiceprints (recommended: 0.45).
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-base-300 bg-base-100/60 p-4 space-y-2">
            <label className="block text-xs font-semibold uppercase tracking-wider text-base-content">
              Automatic Transcription Engine
            </label>
            <Combobox
              value={transcriptionProvider}
              onChange={(v) => setTranscriptionProvider(v as "speechmatics" | "local" | "assemblyai" | "muse")}
              placeholder="Select engine…"
              ariaLabel="Automatic transcription engine"
              searchPlaceholder="Search engines…"
              options={TRANSCRIBE_PROVIDER_OPTIONS}
            />
            <p className="text-[11px] text-base-content/50">
              Controls the default for new uploads and manual transcription. Keys entered below take effect immediately; environment variables (ASSEMBLY_API_KEY, SPEECHMATICS_API_KEY, MUSE_API_KEY) override them when set.
            </p>
            <div className="space-y-1.5">
              <label className="block text-xs font-semibold uppercase tracking-wider text-base-content">
                Plaud recordings
              </label>
              <Combobox
                value={plaudRetranscribeMode}
                onChange={(v) => setPlaudRetranscribeMode(v as "off" | "unnamed" | "always")}
                placeholder="Select Plaud transcript policy…"
                ariaLabel="Plaud transcript policy"
                searchPlaceholder="Search Plaud options…"
                options={PLAUD_RETRANSCRIBE_OPTIONS}
              />
              <p className="text-[11px] text-base-content/50">
                Plaud&apos;s transcript is always kept as a fallback. Re-transcribing uses the engine above as authoritative; changes take effect immediately without a restart.
              </p>
            </div>
            <div className="space-y-1.5 pt-1">
              <label htmlFor="speechmatics-key-input" className="text-xs text-base-content font-medium flex items-center justify-between">
                <span>Speechmatics API Key:</span>
                <span className="text-[10px] text-base-content/50 font-mono">SPEECHMATICS_API_KEY env overrides</span>
              </label>
              <input
                id="speechmatics-key-input"
                type="password"
                value={speechmaticsApiKey}
                onChange={(e) => setSpeechmaticsApiKey(e.target.value)}
                placeholder="Speechmatics key…"
                autoComplete="off"
                className="input input-md w-full placeholder-base-content/50 font-mono"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="assemblyai-key-input" className="text-xs text-base-content font-medium flex items-center justify-between">
                <span>AssemblyAI API Key:</span>
                <span className="text-[10px] text-base-content/50 font-mono">ASSEMBLY_API_KEY env overrides</span>
              </label>
              <input
                id="assemblyai-key-input"
                type="password"
                value={assemblyaiApiKey}
                onChange={(e) => setAssemblyaiApiKey(e.target.value)}
                placeholder="AssemblyAI key…"
                autoComplete="off"
                className="input input-md w-full placeholder-base-content/50 font-mono"
              />
            </div>
            <div className="space-y-1.5">
              <label htmlFor="muse-key-input" className="text-xs text-base-content font-medium flex items-center justify-between">
                <span>Muse Voice API Key:</span>
                <span className="text-[10px] text-base-content/50 font-mono">MUSE_API_KEY env overrides</span>
              </label>
              <input
                id="muse-key-input"
                type="password"
                value={museApiKey}
                onChange={(e) => setMuseApiKey(e.target.value)}
                placeholder="Muse Voice key…"
                autoComplete="off"
                className="input input-md w-full placeholder-base-content/50 font-mono"
              />
            </div>
          </div>

          {testResult && (
            <div
              className={`rounded-lg border p-3 text-xs ${
                testResult.ok
                  ? "border-primary/30 bg-primary/10 text-primary"
                  : "border-error/30 bg-error/10 text-error"
              }`}
            >
              <div className="flex items-center justify-between">
                <span>{testResult.message}</span>
                {testResult.durationMs && (
                  <span className="font-mono text-[10px] opacity-75">{testResult.durationMs}ms</span>
                )}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between border-t border-base-300 pt-4">
            <button
              type="button"
              onClick={() => void handleTestConnection()}
              disabled={testing}
              className="btn btn-sm bg-base-300 min-h-11"
            >
              {testing ? "Testing…" : <><Zap className="h-4 w-4 shrink-0" aria-hidden="true" /> Test Connection</>}
            </button>

            <div className="flex gap-3">
              <button
                type="button"
                onClick={onClose}
                disabled={saving}
                className="btn btn-sm min-h-11"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="btn btn-primary btn-sm inline-flex items-center gap-2 min-h-11"
              >
                {saving ? "Saving…" : "Save LLM Settings"}
              </button>
            </div>
          </div>
        </form>
      )}
    </Modal>
  );
}

function formatLogTimestamp(timestamp: number): string {
  const d = new Date(timestamp);
  const pad = (n: number, z = 2) => String(n).padStart(z, "0");
  const yr = d.getFullYear();
  const mo = pad(d.getMonth() + 1);
  const day = pad(d.getDate());
  const hr = pad(d.getHours());
  const min = pad(d.getMinutes());
  const sec = pad(d.getSeconds());
  const ms = pad(d.getMilliseconds(), 3);
  return `${yr}-${mo}-${day} ${hr}:${min}:${sec}.${ms}`;
}

function formatUsageUsd(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

function formatUsageTokens(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return `${value}`;
}

function formatUsageHours(audioSeconds: number | null | undefined): string {
  if (audioSeconds === null || audioSeconds === undefined) return "—";
  const hours = audioSeconds / 3600;
  if (hours >= 10) return `${hours.toFixed(1)}h`;
  if (hours >= 1) return `${hours.toFixed(2)}h`;
  return `${Math.round(audioSeconds / 60)}m`;
}

export function UsageView({ onSelectMeeting }: { onSelectMeeting: (id: string) => void }): ReactElement {
  const currentMonth = new Date().toISOString().slice(0, 7);
  const [month, setMonth] = useState(currentMonth);
  const [showAllTime, setShowAllTime] = useState(false);
  const [summary, setSummary] = useState<UsageSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const params = showAllTime ? "period=all" : `month=${encodeURIComponent(month)}`;
        const res = await fetch(`/api/usage/summary?${params}`);
        if (!res.ok) {
          const data = (await res.json().catch(() => ({}))) as { error?: string };
          throw new Error(data.error || `Request failed (${res.status})`);
        }
        const data = (await res.json()) as UsageSummary;
        if (!cancelled) setSummary(data);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof Error ? exc.message : "Failed to load usage");
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [month, showAllTime]);

  const rates = summary?.rates?.transcriptionUsdPerHour;
  const byProvider = summary?.byProvider ?? [];
  const byMeeting = summary?.byMeeting ?? [];
  const rateNote = rates
    ? Object.entries(rates)
        .filter(([, v]) => (v ?? 0) > 0)
        .map(([k, v]) => `${k} $${Number(v).toFixed(2)}/hr`)
        .join(" · ")
    : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-base-content">API Usage & Spend</h1>
          <p className="text-sm text-base-content/70">
            Estimated cloud spend: transcription billed per hour of audio, LLM billed per token.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="month"
            value={month}
            max={currentMonth}
            disabled={showAllTime}
            onChange={(e) => {
              if (e.target.value) {
                setMonth(e.target.value);
                setShowAllTime(false);
              }
            }}
            aria-label="Usage month"
            className="input input-md w-full py-1.5 text-base-content disabled:opacity-50"
          />
          <button
            type="button"
            onClick={() => setShowAllTime((v) => !v)}
            className={`min-h-[44px] rounded-lg border px-3 py-1.5 text-sm font-medium transition sm:min-h-0 ${
              showAllTime
                ? "border-primary/50 bg-primary/10 text-primary"
                : "border-base-300 text-base-content hover:border-base-300 hover:text-base-content"
            }`}
          >
            All time
          </button>
        </div>
      </div>

      {loading ? (
        <p className="text-sm text-base-content/70">Loading usage…</p>
      ) : error ? (
        <p className="text-sm text-error">{error}</p>
      ) : summary ? (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-base-300 bg-base-200/60 p-4">
              <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Total spend</div>
              <div className="mt-1 text-2xl font-bold text-base-content">{formatUsageUsd(summary.totalCostUsd)}</div>
              <div className="mt-1 text-xs text-base-content/50">{summary.eventCount} billed events</div>
            </div>
            <div className="rounded-xl border border-base-300 bg-base-200/60 p-4">
              <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">Transcription</div>
              <div className="mt-1 text-2xl font-bold text-base-content">{formatUsageUsd(summary.transcriptionCostUsd)}</div>
              <div className="mt-1 text-xs text-base-content/50">{formatUsageHours(summary.audioSeconds)} of audio</div>
            </div>
            <div className="rounded-xl border border-base-300 bg-base-200/60 p-4">
              <div className="text-xs font-medium uppercase tracking-wide text-base-content/50">LLM</div>
              <div className="mt-1 text-2xl font-bold text-base-content">{formatUsageUsd(summary.llmCostUsd)}</div>
              <div className="mt-1 text-xs text-base-content/50">
                {formatUsageTokens(summary.totalTokens)} tokens ({formatUsageTokens(summary.inputTokens)} in /{" "}
                {formatUsageTokens(summary.outputTokens)} out)
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-base-300 bg-base-200/60 p-4">
            <h2 className="text-sm font-semibold text-base-content">By provider</h2>
            {byProvider.length === 0 ? (
              <p className="mt-2 text-sm text-base-content/50">No cloud usage in this period. Local transcription is free.</p>
            ) : (
              <>
              <div className="mt-2 hidden overflow-x-auto sm:block">
                <table className="table table-sm w-full min-w-[560px] text-left">
                  <thead>
                    <tr className="text-xs uppercase tracking-wide text-base-content/50">
                      <th className="py-1.5 pr-3 font-medium">Provider</th>
                      <th className="py-1.5 pr-3 font-medium">Audio</th>
                      <th className="py-1.5 pr-3 font-medium">Tokens</th>
                      <th className="py-1.5 pr-3 text-right font-medium">Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {byProvider.map((p) => (
                      <tr key={p.provider} className="border-t border-base-300/80">
                        <td className="py-2 pr-3">
                          <span className="font-medium text-base-content">{p.provider}</span>
                          <span className="ml-2 text-xs text-base-content/50">{(p.kinds ?? []).join(", ")}</span>
                        </td>
                        <td className="py-2 pr-3 text-base-content">{formatUsageHours(p.audioSeconds)}</td>
                        <td className="py-2 pr-3 text-base-content">{formatUsageTokens(p.totalTokens)}</td>
                        <td className="py-2 text-right font-semibold text-base-content">{formatUsageUsd(p.costUsd)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ul className="mt-2 space-y-2 sm:hidden">
                {byProvider.map((p) => (
                  <li key={p.provider} className="rounded-lg border border-base-300 bg-base-100/60 p-3">
                    <div>
                      <span className="font-medium text-base-content">{p.provider}</span>
                      <span className="ml-2 text-xs text-base-content/50">{(p.kinds ?? []).join(", ")}</span>
                    </div>
                    <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
                      <div>
                        <dt className="text-base-content/50">Audio</dt>
                        <dd className="text-base-content">{formatUsageHours(p.audioSeconds)}</dd>
                      </div>
                      <div>
                        <dt className="text-base-content/50">Tokens</dt>
                        <dd className="text-base-content">{formatUsageTokens(p.totalTokens)}</dd>
                      </div>
                      <div className="text-right">
                        <dt className="text-base-content/50">Cost</dt>
                        <dd className="font-semibold text-base-content">{formatUsageUsd(p.costUsd)}</dd>
                      </div>
                    </dl>
                  </li>
                ))}
              </ul>
              </>
            )}
          </div>

          <div className="rounded-xl border border-base-300 bg-base-200/60 p-4">
            <h2 className="text-sm font-semibold text-base-content">Per meeting</h2>
            {byMeeting.length === 0 ? (
              <p className="mt-2 text-sm text-base-content/50">No per-meeting usage in this period.</p>
            ) : (
              <>
              <div className="mt-2 hidden overflow-x-auto sm:block">
                <table className="table table-sm w-full min-w-[640px] text-left">
                  <thead>
                    <tr className="text-xs uppercase tracking-wide text-base-content/50">
                      <th className="py-1.5 pr-3 font-medium">Meeting</th>
                      <th className="py-1.5 pr-3 font-medium">Audio</th>
                      <th className="py-1.5 pr-3 font-medium">Transcription</th>
                      <th className="py-1.5 pr-3 font-medium">LLM</th>
                      <th className="py-1.5 text-right font-medium">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {byMeeting.map((m) => (
                      <tr key={m.meetingId} className="border-t border-base-300/80">
                        <td className="py-2 pr-3">
                          <button
                            type="button"
                            onClick={() => onSelectMeeting(m.meetingId)}
                            className="font-medium text-primary hover:text-primary hover:underline text-left"
                          >
                            {m.meetingTitle || "Untitled meeting"}
                          </button>
                          {m.meetingStartTime ? (
                            <div className="text-xs text-base-content/50">{formatDate(m.meetingStartTime)}</div>
                          ) : null}
                        </td>
                        <td className="py-2 pr-3 text-base-content">{formatUsageHours(m.audioSeconds)}</td>
                        <td className="py-2 pr-3 text-base-content">{formatUsageUsd(m.transcriptionCostUsd)}</td>
                        <td className="py-2 pr-3 text-base-content">
                          {formatUsageUsd(m.llmCostUsd)}
                          <span className="ml-1 text-xs text-base-content/50">({formatUsageTokens(m.totalTokens)})</span>
                        </td>
                        <td className="py-2 text-right font-semibold text-base-content">
                          {formatUsageUsd(m.totalCostUsd)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <ul className="mt-2 space-y-2 sm:hidden">
                {byMeeting.map((m) => (
                  <li key={m.meetingId} className="rounded-lg border border-base-300 bg-base-100/60 p-3">
                    <button
                      type="button"
                      onClick={() => onSelectMeeting(m.meetingId)}
                      className="flex min-h-[44px] w-full flex-col justify-center text-left"
                    >
                      <span className="font-medium text-primary">{m.meetingTitle || "Untitled meeting"}</span>
                      {m.meetingStartTime ? (
                        <span className="text-xs text-base-content/50">{formatDate(m.meetingStartTime)}</span>
                      ) : null}
                    </button>
                    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                      <div>
                        <dt className="text-base-content/50">Audio</dt>
                        <dd className="text-base-content">{formatUsageHours(m.audioSeconds)}</dd>
                      </div>
                      <div>
                        <dt className="text-base-content/50">Transcription</dt>
                        <dd className="text-base-content">{formatUsageUsd(m.transcriptionCostUsd)}</dd>
                      </div>
                      <div>
                        <dt className="text-base-content/50">LLM</dt>
                        <dd className="text-base-content">
                          {formatUsageUsd(m.llmCostUsd)}
                          <span className="ml-1 text-base-content/50">({formatUsageTokens(m.totalTokens)})</span>
                        </dd>
                      </div>
                      <div>
                        <dt className="text-base-content/50">Total</dt>
                        <dd className="font-semibold text-base-content">{formatUsageUsd(m.totalCostUsd)}</dd>
                      </div>
                    </dl>
                  </li>
                ))}
              </ul>
              </>
            )}
          </div>

          <p className="text-xs text-base-content/50">
            Estimates, not invoices{rateNote ? `: transcription at ${rateNote}` : ""}; LLM per-token rates vary by
            model (see the AI model picker for per-model pricing). Summary LLM usage is tracked from this update
            onward — older summaries predate usage tracking, while transcription and chat history were backfilled.
          </p>
        </>
      ) : null}
    </div>
  );
}

function MeetingUsageLine({ meetingId }: { meetingId: string }): ReactElement | null {
  const [usage, setUsage] = useState<MeetingUsageResponse | null>(null);
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch(`/api/meetings/${meetingId}/usage`);
        if (res.ok) {
          const data = (await res.json()) as MeetingUsageResponse;
          if (!cancelled) setUsage(data);
        }
      } catch {
        // Usage is informational; never break the meeting view.
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [meetingId]);
  if (!usage || usage.eventCount === 0) return null;
  return (
    <div className="mt-2 text-xs sm:text-sm text-base-content/70" title="Estimated cloud spend for this meeting (transcription billed per audio hour, LLM per token).">
      Cost ≈ <span className="font-semibold text-base-content">{formatUsageUsd(usage.totalCostUsd)}</span>
      <span className="text-base-content/50">
        {" "}(transcription {formatUsageUsd(usage.transcriptionCostUsd)} · LLM {formatUsageUsd(usage.llmCostUsd)})
      </span>
    </div>
  );
}

export function LogsView({ onSelectMeeting }: { onSelectMeeting: (id: string) => void }): ReactElement {
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [selectedLevel, setSelectedLevel] = useState<LogLevel>("debug");
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(false);
  const [expandedLogIds, setExpandedLogIds] = useState<Set<string>>(new Set());

  const fetchLogs = async () => {
    try {
      const params = new URLSearchParams();
      if (selectedLevel) params.set("level", selectedLevel);
      if (selectedCategory) params.set("category", selectedCategory);
      if (searchQuery.trim()) params.set("search", searchQuery.trim());
      params.set("limit", "200");

      const res = await fetch(`/api/logs?${params.toString()}`);
      if (res.ok) {
        const data = (await res.json()) as LogListResponse;
        setLogs(data.logs);
        setTotalCount(data.pagination.total);
        if (data.categories && data.categories.length > 0) {
          setCategories(data.categories);
        }
      }
    } catch {
      // Ignore network errors on background poll
    }
  };

  useEffect(() => {
    setLoading(true);
    void fetchLogs().finally(() => setLoading(false));
  }, [selectedLevel, selectedCategory, searchQuery]);

  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      void fetchLogs();
    }, 3000);
    return () => clearInterval(interval);
  }, [autoRefresh, selectedLevel, selectedCategory, searchQuery]);

  const toggleExpand = (id: string) => {
    setExpandedLogIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleClearLogs = async () => {
    if (!(await confirmAction({ title: "Clear logs", message: "Are you sure you want to clear all stored system logs?", confirmLabel: "Clear", danger: true }))) {
      return;
    }
    try {
      const res = await fetch("/api/logs", { method: "DELETE" });
      if (res.ok) {
        await fetchLogs();
      }
    } catch {
      // ignore
    }
  };

  return (
    <div className="space-y-6">
      {/* Header & Controls */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-base-content">System & Job Logs</h1>
            {autoRefresh && (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 border border-primary/30 px-2.5 py-0.5 text-[10px] font-semibold text-primary">
                <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                Live (3s)
              </span>
            )}
          </div>
          <p className="text-sm text-base-content/70">
            Real-time execution trace for ingestion, Speechmatics/local transcription, diarization, speaker enrollment, and AI summarization.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 sm:gap-2.5">
          <button
            type="button"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`min-h-[44px] flex-1 rounded-lg border px-3 py-1.5 text-sm font-medium transition sm:min-h-0 sm:flex-none sm:text-xs ${
              autoRefresh
                ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/20"
                : "border-base-300 bg-base-300 text-base-content/70 hover:text-base-content"
            }`}
          >
            {autoRefresh ? <><Pause className="h-4 w-4 shrink-0" aria-hidden="true" /> Pause Stream</> : <><Play className="h-4 w-4 shrink-0" aria-hidden="true" /> Auto-Refresh</>}
          </button>
          <button
            type="button"
            onClick={() => void fetchLogs()}
            disabled={loading}
            className="btn min-h-[44px] flex-1 bg-base-300 sm:min-h-0 sm:flex-none sm:text-xs"
          >
            {loading ? "Refreshing…" : <><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /> Refresh</>}
          </button>
          <button
            type="button"
            onClick={() => void handleClearLogs()}
            className="btn btn-error btn-soft min-h-[44px] flex-1 border border-error/30 text-error hover:bg-error/20 sm:min-h-0 sm:flex-none sm:text-xs"
          >
            <Trash2 className="h-4 w-4 shrink-0" aria-hidden="true" /> Clear
          </button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="flex flex-col gap-3 rounded-xl border border-base-300 bg-base-200/60 p-3.5 sm:flex-row sm:flex-wrap sm:items-center">
        {/* Severity Level Filter */}
        <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-2">
          <span className="text-xs font-medium text-base-content/70">
            Min Level:
          </span>
          <div className="w-full sm:w-56">
            <Combobox
              value={selectedLevel}
              onChange={(v) => setSelectedLevel(v as LogLevel)}
              placeholder="Select level…"
              ariaLabel="Minimum log level"
              searchPlaceholder="Search levels…"
              options={LOG_LEVEL_OPTIONS}
            />
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:gap-2">
          <span className="text-xs font-medium text-base-content/70">
            Category:
          </span>
          <div className="w-full sm:w-56">
            <Combobox
              value={selectedCategory}
              onChange={setSelectedCategory}
              placeholder="Select category…"
              ariaLabel="Log category"
              searchPlaceholder="Search categories…"
              emptyText="No categories match your search."
              options={logCategoryOptions(categories)}
            />
          </div>
        </div>

        {/* Search Input */}
        <div className="flex flex-1 min-w-0 items-center gap-2 sm:min-w-[200px]">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search logs by message, meeting, or detail…"
            className="input input-md w-full py-1.5 sm:text-xs"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery("")}
              className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center text-base-content/50 hover:text-base-content text-xs sm:min-h-0 sm:min-w-0 sm:px-1"
            >
              <X className="h-4 w-4 shrink-0" aria-hidden="true" />
            </button>
          )}
        </div>

        {/* Total Badge */}
        <span className="text-[11px] font-mono text-base-content/50">
          Showing {logs.length} of {totalCount} logs
        </span>
      </div>

      {/* Logs Table / List */}
      <div className="rounded-xl border border-base-300 bg-base-200/60 backdrop-blur overflow-hidden">
        {logs.length === 0 ? (
          <div className="p-12 text-center text-base-content/50 text-sm">
            No log entries match the current filter criteria.
          </div>
        ) : (
          <div className="divide-y divide-base-300/60 font-mono text-xs">
            {logs.map((log) => {
              const isExpanded = expandedLogIds.has(log.id);
              const hasDetails = log.details && Object.keys(log.details).length > 0;

              return (
                <div
                  key={log.id}
                  className={`p-3.5 transition hover:bg-base-300/30 ${
                    log.level === "error"
                      ? "bg-error/5"
                      : log.level === "warn"
                      ? "bg-warning/5"
                      : ""
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2.5">
                      {/* Level Badge */}
                      <span
                        className={`rounded px-2 py-0.5 font-bold uppercase tracking-wider text-[10px] border ${
                          log.level === "error"
                            ? "border-error/40 bg-error/10 text-error"
                            : log.level === "warn"
                            ? "border-warning/40 bg-warning/10 text-warning"
                            : log.level === "info"
                            ? "border-info/40 bg-info/10 text-info"
                            : "border-base-300 bg-base-300/80 text-base-content/70"
                        }`}
                      >
                        {log.level}
                      </span>

                      {/* Category Badge */}
                      <span className="rounded bg-base-300 px-2 py-0.5 text-[10px] text-primary border border-base-300">
                        {log.category}
                      </span>

                      {/* Timestamp */}
                      <span className="text-base-content/50 text-[11px]">
                        {formatLogTimestamp(log.createdAt)}
                      </span>

                      {/* Linked Meeting */}
                      {log.meetingId && (
                        <button
                          type="button"
                          onClick={() => onSelectMeeting(log.meetingId!)}
                          className="rounded bg-base-300/60 px-2 py-0.5 text-[10px] text-accent hover:bg-accent/20 transition border border-accent/30 truncate max-w-[220px]"
                          title={`Open meeting ${log.meetingTitle || log.meetingId}`}
                        >
                          📌 {log.meetingTitle || log.meetingId.slice(0, 8)}
                        </button>
                      )}
                    </div>

                    {/* Details toggle */}
                    {hasDetails && (
                      <button
                        type="button"
                        onClick={() => toggleExpand(log.id)}
                        className="rounded border border-base-300 bg-base-300/60 px-2 py-0.5 text-[10px] text-base-content hover:bg-base-300 transition"
                      >
                        {isExpanded ? <><ChevronUp className="h-3 w-3 shrink-0" aria-hidden="true" /> Hide Details</> : <><ChevronDown className="h-3 w-3 shrink-0" aria-hidden="true" /> Details ({Object.keys(log.details!).length})</>}
                      </button>
                    )}
                  </div>

                  {/* Message */}
                  <div className="mt-2 text-base-content font-sans text-xs sm:text-sm leading-relaxed">
                    {log.message}
                  </div>

                  {/* Expanded JSON details */}
                  {isExpanded && log.details && (
                    <div className="mt-3 rounded-lg border border-base-300 bg-base-100 p-3 text-xs text-base-content overflow-x-auto">
                      <pre className="font-mono text-[11px] leading-snug">
                        {JSON.stringify(log.details, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

export function BackupRestoreView({ onRestoreComplete }: { onRestoreComplete: () => void }): ReactElement {
  const [backups, setBackups] = useState<BackupInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [restoring, setRestoring] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [statusMessage, setStatusMessage] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);
  const [confirmRestoreTarget, setConfirmRestoreTarget] = useState<{
    type: "file" | "stored";
    filename?: string;
    stats?: { meetings: number; audioFiles: number; sizeBytes?: number };
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadBackups = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/backup/list");
      if (res.ok) {
        const data = await res.json();
        setBackups(data.backups || []);
      }
    } catch {
      setStatusMessage({ type: "error", text: "Failed to fetch backup list from server." });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadBackups();
  }, []);

  const handleCreateServerBackup = async () => {
    setCreating(true);
    setStatusMessage(null);
    try {
      const res = await fetch("/api/backup", { method: "POST" });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to create backup");
      }
      setStatusMessage({
        type: "success",
        text: `Backup created successfully (${formatBytes(data.backup.sizeBytes)}). Included ${data.backup.manifest.stats.meetingCount} meetings & ${data.backup.manifest.stats.audioFilesCount} audio files.`
      });
      await loadBackups();
    } catch (err) {
      setStatusMessage({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteBackup = async (filename: string) => {
    if (!(await confirmAction({ title: "Delete backup", message: `Are you sure you want to delete backup archive "${filename}" from server?`, confirmLabel: "Delete", danger: true }))) {
      return;
    }
    try {
      const res = await fetch(`/api/backup/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || "Failed to delete backup");
      }
      setStatusMessage({ type: "success", text: `Deleted backup ${filename}` });
      await loadBackups();
    } catch (err) {
      setStatusMessage({ type: "error", text: err instanceof Error ? err.message : String(err) });
    }
  };

  const handleExecuteRestore = async () => {
    if (!confirmRestoreTarget) return;
    setRestoring(true);
    setStatusMessage(null);

    try {
      let res: Response;
      if (confirmRestoreTarget.type === "file" && selectedFile) {
        const formData = new FormData();
        formData.append("file", selectedFile);
        res = await fetch("/api/backup/restore", {
          method: "POST",
          body: formData
        });
      } else if (confirmRestoreTarget.type === "stored" && confirmRestoreTarget.filename) {
        res = await fetch("/api/backup/restore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ filename: confirmRestoreTarget.filename })
        });
      } else {
        throw new Error("No restore target selected");
      }

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Restore operation failed");
      }

      setStatusMessage({
        type: "success",
        text: `Restore complete! Successfully restored ${data.result.stats.meetings} meetings, ${data.result.stats.audioFiles} audio files, ${data.result.stats.summaries} summaries, and ${data.result.stats.speakers} speakers.`
      });

      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setConfirmRestoreTarget(null);
      onRestoreComplete();
      await loadBackups();
    } catch (err) {
      setStatusMessage({ type: "error", text: err instanceof Error ? err.message : String(err) });
    } finally {
      setRestoring(false);
      setConfirmRestoreTarget(null);
    }
  };

  const backupActions = (b: BackupInfo) => (
    <>
      <a
        href={`/api/backup/download/${encodeURIComponent(b.filename)}`}
        download
        className="inline-flex min-h-[44px] flex-1 items-center justify-center whitespace-nowrap rounded border border-base-300 bg-base-300 px-2.5 py-1 text-xs text-base-content hover:bg-base-300 hover:text-white transition sm:min-h-0 sm:flex-none"
      >
        <Download className="h-4 w-4 shrink-0" aria-hidden="true" /> Download
      </a>
      <button
        type="button"
        onClick={() => {
          setConfirmRestoreTarget({
            type: "stored",
            filename: b.filename,
            stats: {
              meetings: b.manifest?.stats.meetingCount ?? 0,
              audioFiles: b.manifest?.stats.audioFilesCount ?? 0,
              sizeBytes: b.sizeBytes
            }
          });
        }}
        className="inline-flex min-h-[44px] flex-1 items-center justify-center whitespace-nowrap rounded border border-warning/40 bg-warning/10 px-2.5 py-1 text-xs text-warning hover:bg-warning/20 transition sm:min-h-0 sm:flex-none"
      >
        <RefreshCw className="h-4 w-4 shrink-0" aria-hidden="true" /> Restore
      </button>
      <button
        type="button"
        onClick={() => void handleDeleteBackup(b.filename)}
        className="btn btn-error btn-soft btn-sm inline-flex min-h-[44px] flex-1 items-center justify-center whitespace-nowrap border border-error/30 text-error hover:bg-error/20 sm:min-h-0 sm:flex-none"
      >
        <Trash2 className="h-4 w-4 shrink-0" aria-hidden="true" /> Delete
      </button>
    </>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-base-content">Backup & Restore</h1>
          <p className="text-sm text-base-content/70">
            Create complete portable archives of your database, settings, templates, transcripts, notes, and all audio files.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <a
            href="/api/backup/export"
            download
            className="btn btn-primary btn-sm min-h-11"
          >
            <span><Download className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
            <span>Download Full Backup (.tar.gz)</span>
          </a>
          <button
            type="button"
            onClick={() => void handleCreateServerBackup()}
            disabled={creating}
            className="btn btn-sm inline-flex items-center gap-2 bg-base-200 font-semibold min-h-11"
          >
            <span><Save className="h-4 w-4 shrink-0" aria-hidden="true" /></span>
            <span>{creating ? "Creating Snapshot…" : "Save Server Snapshot"}</span>
          </button>
        </div>
      </div>

      {/* Status Alert */}
      {statusMessage && (
        <div
          className={`rounded-xl border p-4 text-sm flex items-start justify-between ${
            statusMessage.type === "success"
              ? "border-primary/40 bg-primary/10 text-primary"
              : statusMessage.type === "error"
              ? "border-error/40 bg-error/10 text-error"
              : "border-accent/40 bg-accent/10 text-accent"
          }`}
        >
          <div className="flex items-center gap-3">
            <span>{statusMessage.type === "success" ? <CheckCircle2 className="h-4 w-4 shrink-0" aria-hidden="true" /> : statusMessage.type === "error" ? <XCircle className="h-4 w-4 shrink-0" aria-hidden="true" /> : <Info className="h-4 w-4 shrink-0" aria-hidden="true" />}</span>
            <span>{statusMessage.text}</span>
          </div>
          <button
            type="button"
            onClick={() => setStatusMessage(null)}
            className="text-base-content/70 hover:text-base-content text-xs px-2"
          >
            <X className="h-4 w-4 shrink-0" aria-hidden="true" />
          </button>
        </div>
      )}

      {/* Restore from Upload Box */}
      <div className="rounded-xl border border-base-300 bg-base-200/60 p-5 backdrop-blur">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-base-content mb-2">
          Restore from Backup Archive
        </h2>
        <p className="text-xs text-base-content/70 mb-4">
          Select an Olive <code className="text-base-content">.tar.gz</code> archive from your computer. All meetings, audio recordings, transcripts, summaries, and configuration will be restored.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-4">
          <input
            ref={fileInputRef}
            type="file"
            accept=".tar.gz,.tgz,application/gzip,application/x-gzip"
            onChange={(e) => {
              const file = e.target.files?.[0] || null;
              setSelectedFile(file);
            }}
            className="block min-h-[44px] w-full text-base text-base-content/70 file:mr-4 file:min-h-[44px] file:rounded-lg file:border-0 file:bg-base-300 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-primary hover:file:bg-base-300 cursor-pointer sm:min-h-0 sm:text-xs sm:file:min-h-0 sm:file:text-xs"
          />
          {selectedFile && (
            <button
              type="button"
              onClick={() => {
                setConfirmRestoreTarget({
                  type: "file",
                  filename: selectedFile.name,
                  stats: { meetings: 0, audioFiles: 0, sizeBytes: selectedFile.size }
                });
              }}
              className="btn btn-warning min-h-[44px] w-full whitespace-nowrap shadow sm:w-auto sm:min-h-0 sm:text-xs"
            >
              Restore Uploaded File
            </button>
          )}
        </div>
      </div>

      {/* Saved Backups Table */}
      <div className="rounded-xl border border-base-300 bg-base-200/60 backdrop-blur overflow-hidden">
        <div className="flex items-center justify-between border-b border-base-300 px-5 py-4">
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-base-content">
              Server Backup Snapshots
            </h2>
            <p className="text-xs text-base-content/50">Stored on server persistent storage</p>
          </div>
          <button
            type="button"
            onClick={() => void loadBackups()}
            disabled={loading}
            className="btn btn-sm bg-base-100/60 min-h-11"
          >
            {loading ? "Refreshing…" : <><RotateCw className="h-4 w-4 shrink-0" aria-hidden="true" /> Refresh List</>}
          </button>
        </div>

        {backups.length === 0 ? (
          <div className="p-8 text-center text-base-content/50 text-sm">
            No backup snapshots currently saved on the server. Click &ldquo;Save Server Snapshot&rdquo; or &ldquo;Download Full Backup&rdquo; above.
          </div>
        ) : (
          <>
          <div className="hidden overflow-x-auto sm:block">
            <table className="table table-xs w-full text-left text-base-content/80">
              <thead className="border-b border-base-300 bg-base-100/40 text-base-content/70 font-medium">
                <tr>
                  <th className="px-5 py-3">Archive File</th>
                  <th className="px-5 py-3">Created</th>
                  <th className="px-5 py-3">Size</th>
                  <th className="px-5 py-3">Meetings</th>
                  <th className="px-5 py-3">Audio Recordings</th>
                  <th className="px-5 py-3">Summaries</th>
                  <th className="px-5 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-base-300/60">
                {backups.map((b) => (
                  <tr key={b.filename} className="hover:bg-base-300/30 transition">
                    <td className="px-5 py-3.5 font-mono text-primary font-medium">
                      {b.filename}
                    </td>
                    <td className="px-5 py-3.5 text-base-content/70">
                      {formatDate(new Date(b.createdAt).getTime())}
                    </td>
                    <td className="px-5 py-3.5 font-mono text-base-content">
                      {formatBytes(b.sizeBytes)}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="rounded bg-base-300 px-2 py-0.5 font-mono text-base-content">
                        {b.manifest?.stats.meetingCount ?? "—"}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-1.5">
                        <span className="rounded bg-base-300 px-2 py-0.5 font-mono text-base-content">
                          {b.manifest?.stats.audioFilesCount ?? "—"} files
                        </span>
                        {b.manifest?.stats.totalAudioSizeBytes ? (
                          <span className="text-[10px] text-base-content/50 font-mono">
                            ({formatBytes(b.manifest.stats.totalAudioSizeBytes)})
                          </span>
                        ) : null}
                      </div>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="rounded bg-base-300 px-2 py-0.5 font-mono text-base-content">
                        {b.manifest?.stats.summaryCount ?? "—"}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-right">
                      <div className="flex items-center justify-end gap-2">{backupActions(b)}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="divide-y divide-base-300/60 sm:hidden">
            {backups.map((b) => (
              <li key={b.filename} className="space-y-3 p-4">
                <div>
                  <div className="break-all font-mono text-sm font-medium text-primary">{b.filename}</div>
                  <div className="mt-1 text-xs text-base-content/70">
                    {formatDate(new Date(b.createdAt).getTime())} · <span className="font-mono text-base-content">{formatBytes(b.sizeBytes)}</span>
                  </div>
                </div>
                <dl className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <dt className="text-base-content/50">Meetings</dt>
                    <dd className="font-mono text-base-content">{b.manifest?.stats.meetingCount ?? "—"}</dd>
                  </div>
                  <div>
                    <dt className="text-base-content/50">Audio</dt>
                    <dd className="font-mono text-base-content">{b.manifest?.stats.audioFilesCount ?? "—"} files</dd>
                  </div>
                </dl>
                <div className="flex flex-wrap gap-2">{backupActions(b)}</div>
              </li>
            ))}
          </ul>
          </>
        )}
      </div>

      {/* Confirmation Modal for Restore */}
      {confirmRestoreTarget && (
        <div className="fixed inset-0 z-50 flex max-h-[100dvh] items-center justify-center overflow-y-auto bg-base-100/80 p-4 backdrop-blur-sm" role="dialog" aria-modal="true" aria-label="Confirm system restore">
          <div className="my-auto w-full max-w-lg rounded-2xl border border-warning/40 bg-base-200 p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3 text-warning">
              <span className="text-2xl">⚠️</span>
              <h3 className="text-lg font-bold text-base-content">Confirm System Restore</h3>
            </div>

            <p className="text-sm text-base-content leading-relaxed">
              Restoring from <strong className="text-warning font-mono">{confirmRestoreTarget.filename}</strong> will overwrite current meetings, audio recordings, transcripts, summaries, and configuration with the contents of this archive.
            </p>

            <div className="rounded-xl border border-base-300 bg-base-100 p-4 space-y-1.5 text-xs text-base-content/70">
              <div className="flex justify-between">
                <span>Source Archive:</span>
                <span className="font-mono text-base-content">{confirmRestoreTarget.filename}</span>
              </div>
              {confirmRestoreTarget.stats?.sizeBytes && (
                <div className="flex justify-between">
                  <span>Archive Size:</span>
                  <span className="font-mono text-base-content">{formatBytes(confirmRestoreTarget.stats.sizeBytes)}</span>
                </div>
              )}
              {confirmRestoreTarget.stats?.meetings ? (
                <div className="flex justify-between">
                  <span>Meetings included:</span>
                  <span className="font-mono text-base-content">{confirmRestoreTarget.stats.meetings}</span>
                </div>
              ) : null}
              {confirmRestoreTarget.stats?.audioFiles ? (
                <div className="flex justify-between">
                  <span>Audio recordings included:</span>
                  <span className="font-mono text-base-content">{confirmRestoreTarget.stats.audioFiles}</span>
                </div>
              ) : null}
            </div>

            <p className="text-xs text-base-content/50">
              Note: An automatic rollback snapshot will be created in temporary storage before file replacement in case of unexpected errors.
            </p>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setConfirmRestoreTarget(null)}
                disabled={restoring}
                className="btn btn-sm min-h-11"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void handleExecuteRestore()}
                disabled={restoring}
                className="btn btn-warning btn-sm inline-flex items-center gap-2 font-bold shadow min-h-11"
              >
                {restoring ? "Restoring System…" : "Yes, Restore System"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

interface DebugQueueJob {
  workflowId: string;
  name: string;
  rawName: string;
  status: string;
  queueName: string | null;
  parentWorkflowId: string | null;
  meetingId: string | null;
  meetingTitle: string | null;
  createdAt: number | null;
  dequeuedAt: number | null;
  completedAt: number | null;
  elapsedMs: number | null;
  waitingMs: number | null;
  queuePosition: number | null;
  deduplicationId: string | null;
  priority: number | null;
  recoveryAttempts: number | null;
  error: string | null;
}

interface DebugStageQueue {
  name: string;
  workerConcurrency: number;
  activeCount: number;
  waitingCount: number;
  activeJobs: DebugQueueJob[];
  waitingJobs: DebugQueueJob[];
  recentJobs: DebugQueueJob[];
}

interface DebugQueuesSnapshot {
  timestamp: number;
  dbosDatabaseUrl: string;
  summary: {
    totalQueues: number;
    totalActiveJobs: number;
    totalWaitingJobs: number;
    activeParentWorkflows: number;
  };
  queues: DebugStageQueue[];
  activeWorkflows: DebugQueueJob[];
  activeTranscriptions: Record<string, TranscriptionProgressUpdate>;
  runningStageRuns: Array<{
    id: string;
    meetingId: string;
    meetingTitle: string | null;
    stage: string;
    status: string;
    jobId: string | null;
    attempts: number;
    startedAt: number | null;
    elapsedMs: number | null;
  }>;
}

function formatDurationCompact(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms}ms`;
  const sec = ms / 1000;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const mins = Math.floor(sec / 60);
  const remSec = Math.round(sec % 60);
  return `${mins}m ${remSec}s`;
}

function DebugQueuesPage() {
  const [snapshot, setSnapshot] = useState<DebugQueuesSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [paused, setPaused] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchSnapshot = useCallback(async () => {
    try {
      const res = await fetch("/api/debug/queues");
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as DebugQueuesSnapshot;
      setSnapshot(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchSnapshot();
    if (paused) return;
    const timer = setInterval(() => {
      void fetchSnapshot();
    }, 1500);
    return () => clearInterval(timer);
  }, [fetchSnapshot, paused]);

  const allTranscriptionEntries = Object.entries(snapshot?.activeTranscriptions || {});
  const activeTranscriptionEntries = allTranscriptionEntries.filter(
    ([, prog]) => prog.stage !== "queued"
  );
  const queuedTranscriptionCount = allTranscriptionEntries.length - activeTranscriptionEntries.length;

  return (
    <div className="min-h-screen bg-base-100 text-base-content p-6 font-sans">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-base-300 pb-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold tracking-tight font-mono">
                /debug · DBOS Stage Queues
              </h1>
              <span
                className={`badge badge-sm gap-1.5 font-medium ${paused ? "badge-neutral" : "badge-success badge-soft"}}`}
              >
                <span
                  className={`status status-xs ${paused ? "status-neutral" : "status-success animate-pulse"}}`}
                />
                {paused ? "Paused" : "Live (1.5s)"}
              </span>
            </div>
            {snapshot && (
              <p className="text-xs text-base-content/50 font-mono mt-1">
                DB: {snapshot.dbosDatabaseUrl} · Updated{" "}
                {new Date(snapshot.timestamp).toLocaleTimeString()}
              </p>
            )}
          </div>

          <div className="flex items-center gap-2">
            <a
              href="/"
              className="btn btn-sm min-h-11"
            >
              ← Back to Olive
            </a>
            <button
              type="button"
              onClick={() => setPaused((p) => !p)}
              className="btn btn-sm min-h-11"
            >
              {paused ? "Resume Polling" : "Pause Polling"}
            </button>
            <button
              type="button"
              onClick={() => void fetchSnapshot()}
              className="btn btn-warning btn-sm min-h-11"
            >
              Refresh Now
            </button>
          </div>
        </div>

        {error && (
          <div className="alert alert-soft alert-error p-4 text-xs font-mono">
            Error loading queue state: {error}
          </div>
        )}

        {loading && !snapshot ? (
          <div className="text-sm text-base-content/50 font-mono">Loading DBOS queue state…</div>
        ) : null}

        {snapshot && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="stat rounded-box border border-base-300 bg-base-200 p-4">
                <div className="stat-title">Stage Queues</div>
                <div className="stat-value font-mono mt-1">
                  {snapshot.summary.totalQueues}
                </div>
              </div>
              <div className="stat rounded-box border border-base-300 bg-base-200 p-4">
                <div className="stat-title">Active Stage Jobs</div>
                <div
                  className={`stat-value font-mono mt-1 ${snapshot.summary.totalActiveJobs > 0 ? "text-warning" : ""}`}
                >
                  {snapshot.summary.totalActiveJobs}
                </div>
              </div>
              <div className="stat rounded-box border border-base-300 bg-base-200 p-4">
                <div className="stat-title">Waiting in Queues</div>
                <div
                  className={`stat-value font-mono mt-1 ${snapshot.summary.totalWaitingJobs > 0 ? "text-info" : ""}`}
                >
                  {snapshot.summary.totalWaitingJobs}
                </div>
              </div>
              <div className="stat rounded-box border border-base-300 bg-base-200 p-4">
                <div className="stat-title">Active Parent Workflows</div>
                <div className="stat-value font-mono mt-1">
                  {snapshot.summary.activeParentWorkflows}
                </div>
              </div>
            </div>

            {(activeTranscriptionEntries.length > 0 ||
              queuedTranscriptionCount > 0 ||
              snapshot.runningStageRuns.length > 0) && (
              <div className="rounded-box border border-warning/30 bg-warning/5 p-4 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-warning">
                    Live Pipeline Activity
                  </h2>
                  {queuedTranscriptionCount > 0 && (
                    <span className="badge badge-sm badge-info badge-soft font-mono">
                      +{queuedTranscriptionCount} queued in pipeline
                    </span>
                  )}
                </div>
                {activeTranscriptionEntries.map(([mid, prog]) => (
                  <div
                    key={mid}
                    className="rounded-lg border border-base-300 bg-base-200/90 p-3 flex flex-wrap items-center justify-between gap-3 text-xs"
                  >
                    <div className="space-y-0.5">
                      <div className="font-mono text-base-content">
                        Meeting <span className="text-warning">{mid}</span> · Stage{" "}
                        <span className="font-semibold text-base-content">{prog.stage}</span>
                      </div>
                      <div className="text-base-content/70">{prog.message}</div>
                    </div>
                    <div className="font-mono text-sm font-bold text-warning">
                      {Math.round(prog.percent ?? 0)}%
                    </div>
                  </div>
                ))}
                {snapshot.runningStageRuns.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {snapshot.runningStageRuns.map((sr) => (
                      <div
                        key={sr.id}
                        className="rounded-lg border border-base-300 bg-base-200/70 px-3 py-2 flex items-center justify-between text-xs font-mono"
                      >
                        <div className="truncate">
                          <span className="text-warning">{sr.stage}</span> ·{" "}
                          <span className="text-base-content">
                            {sr.meetingTitle || sr.meetingId}
                          </span>{" "}
                          <span className="text-base-content/50">({sr.jobId || sr.id.slice(0, 8)})</span>
                        </div>
                        <span className="text-base-content/70 ml-2 shrink-0">
                          {formatDurationCompact(sr.elapsedMs)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {snapshot.queues.map((q) => {
                const isBusy = q.activeCount > 0;
                const hasWaiting = q.waitingCount > 0;
                return (
                  <div
                    key={q.name}
                    className={`card card-border bg-base-200 p-4 space-y-3 ${
                      isBusy
                        ? "border-warning/50"
                        : hasWaiting
                        ? "border-info/40"
                        : "border-base-300"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <span className="font-mono text-sm font-bold text-base-content">
                          {q.name}
                        </span>
                        <span
                          className={`badge badge-sm font-mono font-semibold ${
                            isBusy
                              ? "badge-warning"
                              : hasWaiting
                              ? "badge-info badge-soft"
                              : "badge-neutral"
                          }`}
                        >
                          {isBusy ? "ACTIVE" : hasWaiting ? "QUEUED" : "IDLE"}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs font-mono text-base-content/70">
                        <span>
                          Workers: <strong className="text-base-content">{q.activeCount}</strong>/
                          {q.workerConcurrency}
                        </span>
                        <span>
                          Depth: <strong className="text-base-content">{q.waitingCount}</strong>
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-base-content/70">
                        Currently Active ({q.activeJobs.length})
                      </div>
                      {q.activeJobs.length === 0 ? (
                        <div className="rounded-lg border border-base-300/70 bg-base-100/50 px-3 py-2 text-xs text-base-content/50 font-mono">
                          No active job
                        </div>
                      ) : (
                        q.activeJobs.map((job) => (
                          <div
                            key={job.workflowId}
                            className="rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-xs font-mono space-y-1"
                          >
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-semibold text-warning">{job.name}</span>
                              <span className="text-warning">
                                {formatDurationCompact(job.elapsedMs)}
                              </span>
                            </div>
                            <div className="text-base-content truncate">
                              ID: <span className="text-base-content/70">{job.workflowId}</span>
                            </div>
                            {(job.meetingTitle || job.meetingId) && (
                              <div className="text-base-content truncate">
                                Meeting:{" "}
                                <span className="text-base-content">
                                  {job.meetingTitle || job.meetingId}
                                </span>
                                {job.meetingTitle && job.meetingId ? (
                                  <span className="text-base-content/50"> ({job.meetingId})</span>
                                ) : null}
                              </div>
                            )}
                            {(() => {
                              const prog = job.meetingId
                                ? snapshot.activeTranscriptions[job.meetingId]
                                : undefined;
                              if (!prog) return null;
                              return (
                                <div className="space-y-0.5 pt-0.5">
                                  <div className="text-warning">
                                    Stage: <span className="text-base-content">{prog.stage}</span> · {Math.round(prog.percent ?? 0)}%
                                    {prog.current != null && prog.total != null
                                      ? ` · batch ${prog.current}/${prog.total}`
                                      : ""}
                                  </div>
                                  {prog.message && (
                                    <div className="text-base-content/70">{prog.message}</div>
                                  )}
                                </div>
                              );
                            })()}
                          </div>
                        ))
                      )}
                    </div>

                    <div className="space-y-1.5">
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-base-content/70">
                        Waiting Queue ({q.waitingJobs.length})
                      </div>
                      {q.waitingJobs.length === 0 ? (
                        <div className="rounded-lg border border-base-300/70 bg-base-100/50 px-3 py-2 text-xs text-base-content/50 font-mono">
                          Queue empty
                        </div>
                      ) : (
                        <div className="space-y-1.5 max-h-48 overflow-y-auto">
                          {q.waitingJobs.map((job) => (
                            <div
                              key={job.workflowId}
                              className="rounded-lg border border-info/30 bg-info/20 px-3 py-2 text-xs font-mono space-y-1"
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span>
                                  <span className="text-info font-bold">
                                    #{job.queuePosition}
                                  </span>{" "}
                                  <span className="text-base-content font-semibold">{job.name}</span>
                                </span>
                                <span className="text-base-content/70">
                                  wait {formatDurationCompact(job.waitingMs)}
                                </span>
                              </div>
                              <div className="text-base-content/70 truncate">ID: {job.workflowId}</div>
                              {(job.meetingTitle || job.meetingId) && (
                                <div className="text-base-content truncate">
                                  Meeting: {job.meetingTitle || job.meetingId}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {q.recentJobs.length > 0 && (
                      <details className="group pt-1">
                        <summary className="cursor-pointer text-[11px] font-mono text-base-content/70 hover:text-base-content select-none">
                          Recent jobs ({q.recentJobs.length})
                        </summary>
                        <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
                          {q.recentJobs.map((job) => (
                            <div
                              key={job.workflowId}
                              className="rounded border border-base-300 bg-base-100/80 px-2.5 py-1.5 text-[11px] font-mono flex items-center justify-between gap-2"
                            >
                              <div className="truncate">
                                <span
                                  className={`mr-2 ${
                                    job.status === "SUCCESS"
                                      ? "text-success"
                                      : job.status === "CANCELLED"
                                      ? "text-base-content/70"
                                      : "text-error"
                                  }`}
                                >
                                  {job.status}
                                </span>
                                <span className="text-base-content">{job.name}</span>
                                {job.meetingTitle || job.meetingId ? (
                                  <span className="text-base-content/70">
                                    {" "}
                                    · {job.meetingTitle || job.meetingId}
                                  </span>
                                ) : null}
                                <span className="text-base-content/50"> ({job.workflowId})</span>
                              </div>
                              <span className="text-base-content/70 shrink-0">
                                {formatDurationCompact(job.elapsedMs)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

const isDebugRoute =
  typeof window !== "undefined" &&
  (window.location.pathname === "/debug" || window.location.pathname.startsWith("/debug/"));

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ToastProvider>
      <ConfirmHostRoot />
      {isDebugRoute ? <DebugQueuesPage /> : <App />}
    </ToastProvider>
  </StrictMode>
);
