# Voiceprint Matching: What It Is and How to Tune It

Olive learns to recognize people's voices so transcripts say **"Matt"**
instead of **"Speaker 2"**. This guide explains how that works, what the two
sliders do, and how to fix common mistakes — no math degree required.

## How it works (the 30-second version)

1. **You teach Olive a voice.** Rename a speaker, confirm a speaker, or enroll
   someone from the People list. Olive saves a small digital summary of how
   that person sounds — their **voiceprint**. The more clean samples Olive has
   (up to 8 per person), the better it recognizes them.
2. **Each new meeting gets sorted into voices.** Olive first figures out "how
   many distinct voices are in this recording, and who spoke when" — before it
   even checks names.
3. **Each voice is compared to your saved voiceprints.** If a voice in the
   meeting sounds enough like someone you've taught Olive, it gets that
   person's name. Otherwise it stays labeled "Speaker 1", "Speaker 2", etc.,
   so you can teach it later.

Two sliders control steps 2 and 3. They're the only knobs you need.

## Where to find the sliders

There are two copies of the same two sliders:

- **Global defaults (Settings → Local ASR & Speaker Recognition).** These
  apply to every future transcription. Set them once and forget them.
- **Per-meeting overrides (meeting → Transcribe → Thresholds).** These apply
  to one transcription only and override the global settings. Use these when
  you're retrying a single tricky meeting without changing everything else.

Both places show the same recommended starting points:

| Slider | Default | Range |
|---|---|---|
| Within-Meeting Clustering | **0.50** | 0.10 – 0.95 |
| Enrolled Voiceprint Matching | **0.45** | 0.10 – 0.95 |

After changing a slider, **re-run transcription** (or press **Re-identify** on
an already-transcribed meeting) for it to take effect. Changing a slider never
rewrites old transcripts by itself.

## Slider 1: Within-Meeting Clustering (default 0.50)

**What it does:** Decides how picky Olive is about telling voices apart
*within a single recording* — before names enter the picture.

- Think of it as: "how different do two voices have to sound before I call
  them two different people?"
- It does **not** compare against your saved people. It only splits up the
  voices in this one meeting.

**When to adjust it:**

| What you see | What's happening | What to do |
|---|---|---|
| Two real people get merged into one "Speaker 1" | Olive is lumping voices together too aggressively | **Raise** the slider a little (try +0.05, e.g. 0.50 → 0.55), then re-transcribe |
| One person is split into "Speaker 1" *and* "Speaker 3" | Olive is splitting too aggressively — normal voice variation (a cold, a headset, excitement) looks like a new person | **Lower** the slider a little (try −0.05, e.g. 0.50 → 0.45), then re-transcribe |

**Tips:**

- Move in small steps (0.05 at a time). Big jumps usually overshoot.
- Meetings with 4+ similar-sounding voices on one conference-room mic usually
  need a slightly **higher** value. A single person recorded across a noisy
  room usually needs a slightly **lower** one.
- If a meeting has the right *number* of speakers but the wrong *names*, this
  slider is fine — look at Slider 2 instead.

## Slider 2: Enrolled Voiceprint Matching (default 0.45)

**What it does:** Decides how confident Olive must be before putting a known
person's name on a voice.

- Think of it as: "how close a match is good enough to use the name?"
- **Higher = stricter.** Fewer wrong names, but more voices left as
  "Speaker 1" for you to assign by hand.
- **Lower = more generous.** More names filled in automatically, but a higher
  chance of the wrong name.

**When to adjust it:**

| What you see | What's happening | What to do |
|---|---|---|
| People you enrolled show up as "Speaker 1 / Speaker 2" (Olive doesn't recognize them) | Olive is being too strict | **Lower** the slider a little (try 0.45 → 0.40), then press **Re-identify** |
| The *wrong* person's name appears on someone's words | Olive is being too generous | **Raise** the slider a little (try 0.45 → 0.50), then press **Re-identify** |

**Tips:**

- This slider only matters for people with an enrolled voiceprint. If nobody
  is enrolled yet, every voice stays "Speaker N" no matter where the slider
  sits — enroll people first.
- If two enrolled people sound alike (family members, similar pitch), Olive
  deliberately leaves close calls unnamed rather than guessing. That's
  intentional: a missing name is easier to fix than a wrong one. Confirm or
  correct the speaker by hand in that case.
- Phone calls, speakerphone audio, heavy accents on background noise, and very
  short clips ("yeah", "mm-hm") all match less reliably. One or two more clean
  enrollments of that person usually beats moving the slider.

## Why does Olive seem to have a "favorite" speaker?

A common report: the same person's name shows up on other people's words,
meeting after meeting — as if Olive prefers them. Nothing is rigged, but
several behaviors combine to produce exactly that effect. Understanding them
makes the fix obvious.

**The usual setup behind it:** one person (often you, the host — you're in
every recording) was enrolled first and has the most voice samples, while
other regulars have one sample or none. Here's what happens then:

1. **More samples = more chances to win.** Olive compares a voice against
every saved sample of every person and keeps each person's *best* score.
Someone with 8 varied samples gets 8 chances at a high score; someone with
1 sample gets one chance. On clear audio the right person still wins, but on
mushy audio (speakerphone, echo, short "yeah / mm-hm" turns) everything
scores in the same middle band — and the person with the most tickets wins
the lottery.

2. **When only one person is enrolled, there's no runner-up check.**
Normally Olive refuses to name someone when two enrolled people score
nearly the same (a close call stays unnamed rather than guessed). But with
only one enrolled person there is nobody to be "close" to, so that safety
check is skipped: anyone scoring above the matching slider gets that one
name. In the early days, when only one voiceprint exists, almost every
stranger looks like that person.

3. **Your corrections can widen the favorite's net.** When you confirm a
speaker by hand, Olive always learns from that clip — even if the audio was
noisy or borderline. That's usually good (your word beats the algorithm),
but if the favorite's name was confirmed on someone else's words, that
wrong voice becomes part of the favorite's profile, and they'll match even
more broadly next time. One mistaken confirmation can echo for weeks.

**How to fix it:**

- **Enroll the people being swallowed, not just the favorite.** The cure for
  "everyone looks like Matt" is giving Priya and Sam clean voice samples
  of their own — a clear 8–12 seconds of each person talking alone, no
  crosstalk. Balanced profiles beat slider-tweaking every time.
- **Raise the matching slider a notch** (try 0.45 → 0.50) and press
  **Re-identify**. A stricter bar hurts the broad, generic matches first —
  which is exactly what favorite-speaker errors are. Resist the urge to
  *lower* the slider because one person isn't recognized; that makes the
  favorite problem worse.
- **Check the People list for imbalance.** If the favorite has 7–8 samples
  and everyone else has 0–1, you've found it. Enroll the others before
  touching anything else.
- **If the favorite's profile is polluted** (it was confirmed on wrong
  turns in the past), the cleanest fix is to delete that speaker and
  re-enroll them from one or two unmistakably-correct clips, then enroll the
  others too — then Re-identify the affected meetings.
- **Use Re-identify, not full re-transcription.** Re-identify re-runs just
  the naming step against your current, now-balanced voiceprints.

As a rule of thumb: a favorite-speaker problem is almost never a clustering
(Slider 1) problem — the voices are being split up fine, they're just being
named wrong. Leave Slider 1 alone and work on Slider 2 plus enrollments.

## The workflow that works best

1. **Start with the defaults** (0.50 / 0.45). They suit most meetings.
2. **Enroll your regulars.** Confirm speakers in a transcript or enroll from
   the People list using a clear stretch of that person talking alone — 10+
   seconds of clean speech beats 3 seconds of crosstalk.
3. **Fix names by hand when Olive is unsure.** Renaming "Speaker 2" → "Priya"
   and confirming it teaches Olive Priya's voice for next time.
4. **Only then tune.** If recognition is still off across many meetings,
   adjust one slider by 0.05 and re-test. If only one meeting is bad, use the
   per-meeting override instead of changing your global defaults.
5. **Use Re-identify after enrolling.** Just taught Olive someone new? Press
   **Re-identify** on older meetings to re-match their speaker labels against
   your now-bigger voiceprint library — no full re-transcription needed.

## Quick troubleshooting

- **"Everyone is Unknown / Speaker N."** Nobody's enrolled yet, or the
  matching slider is too strict. Enroll your people first; if they're enrolled
  and still unmatched, lower Slider 2 slightly and Re-identify.
- **"Right number of speakers, wrong names."** Leave Slider 1 alone; raise
  Slider 2 slightly and Re-identify. If two specific people keep getting
  confused, enroll one more clean sample of each.
- **"One person appears as two speakers."** Lower Slider 1 slightly and
  re-transcribe.
- **"Two people merged into one speaker."** Raise Slider 1 slightly and
  re-transcribe.
- **"Slider changes did nothing."** You need to re-transcribe (Slider 1) or
  Re-identify (Slider 2) afterward — the change applies to the next run, not
  retroactively.
- **"Same person everywhere — Olive has a favorite."** See the section above: enroll the swallowed people with clean samples, raise Slider 2 slightly, and Re-identify. Don't lower the slider to fix it.
- **"It worked on good audio but not on this meeting."** Bad audio (echoey
  room, speakerphone, crosstalk, very short turns) is the #1 cause. No slider
  position rescues a 2-second noisy clip — confirm that speaker by hand so the
  voiceprint learns from your correction.

## Settings cheat sheet

- **Within-Meeting Clustering (0.50):** fixes *splitting* problems — too few
  vs. too many "Speaker N" labels in one meeting. Higher = split more apart.
- **Enrolled Voiceprint Matching (0.45):** fixes *naming* problems — unknown
  people vs. wrong names. Higher = stricter naming, fewer mistakes but more
  manual labeling.
- **Small moves, one at a time.** Change one slider by 0.05, re-run, and
  compare. If you're ever lost, put both back to 0.50 / 0.45 and re-enroll one
  clean sample of the person Olive keeps missing.
