/**
 * StreamingScheduler - throttled UI commit scheduler for streaming generation.
 *
 * Batches incoming chunks in a cheap non-reactive staging buffer and flushes
 * to the reactive store about every 400ms (~2.5Hz steady state). First
 * fragment flushes immediately for TTFT (unless hidden), final chunk and
 * semantic boundaries flush immediately, hidden tabs defer commits until
 * visible.
 */

export interface StreamingSchedulerCallbacks {
	onContent?: (text: string) => void;
	onReasoning?: (text: string) => void;
	onToolJson?: (json: string) => void;
}

export interface StreamingSchedulerOptions {
	flushIntervalMs?: number; // default 400
}

type PendingEvent = { type: 'content' | 'reasoning' | 'tool'; data: string };

export class StreamingScheduler {
	private pendingEvents: PendingEvent[] = [];
	private flushTimer: ReturnType<typeof setTimeout> | null = null;
	private firstFlushDone = false;
	private chunks = 0;
	private commits = 0;
	private destroyed = false;

	constructor(
		private readonly cbs: StreamingSchedulerCallbacks,
		private readonly opts: StreamingSchedulerOptions = {}
	) {}

	private get interval(): number {
		return this.opts.flushIntervalMs ?? 400;
	}

	private isHidden(): boolean {
		return typeof document !== 'undefined' && document.visibilityState !== 'visible';
	}

	private flush(): void {
		if (this.flushTimer !== null) {
			clearTimeout(this.flushTimer);
			this.flushTimer = null;
		}
		if (this.isHidden()) return;
		this.flushNow();
	}

	// Deliver staged events regardless of visibility. Used for abort/stop paths
	// where losing the last coalesced chunk is worse than a hidden-tab commit.
	forceFlush(): void {
		if (this.flushTimer !== null) {
			clearTimeout(this.flushTimer);
			this.flushTimer = null;
		}
		this.flushNow();
	}

	private flushNow(): void {
		if (this.pendingEvents.length === 0) return;
		const events = this.pendingEvents;
		this.pendingEvents = [];
		for (const ev of events) {
			if (ev.type === 'content') this.cbs.onContent?.(ev.data);
			else if (ev.type === 'reasoning') this.cbs.onReasoning?.(ev.data);
			else if (ev.type === 'tool') this.cbs.onToolJson?.(ev.data);
			this.commits++;
		}
		if (typeof window !== 'undefined') {
			try {
				const s = (window as unknown as { __LLAMA_STREAM_STATS?: { commits: number } }).__LLAMA_STREAM_STATS;
				if (s) s.commits = this.commits;
			} catch {}
		}
		// If new events arrived while flushing, schedule next.
		if (this.pendingEvents.length > 0) this.schedule();
	}

	private schedule(): void {
		if (this.flushTimer !== null) return;
		if (this.isHidden()) return;
		this.flushTimer = setTimeout(() => this.flush(), this.interval);
	}

	private pushEvent(type: PendingEvent['type'], data: string): void {
		if (this.destroyed || !data) return;
		this.chunks++;
		if (typeof window !== 'undefined') {
			try {
				const s = (window as unknown as { __LLAMA_STREAM_STATS?: { chunks: number } }).__LLAMA_STREAM_STATS;
				if (s) s.chunks = this.chunks;
			} catch {}
		}
		// Coalesce adjacent same-type events
		const last = this.pendingEvents[this.pendingEvents.length - 1];
		if (last && last.type === type) {
			if (type === 'tool') last.data = data; // tool: replace with latest aggregated JSON
			else last.data += data;
		} else {
			// Different type arrived - flush previous type immediately to preserve order,
			// unless it's the very first flush and we're hidden (defer).
			if (this.pendingEvents.length > 0) {
				// Flush previous pending events now to keep chronological order
				// (e.g. reasoning -> content transition should not reorder)
				this.flush();
			}
			this.pendingEvents.push({ type, data });
		}

		if (!this.firstFlushDone) {
			if (this.isHidden()) return; // defer first fragment when hidden
			this.firstFlushDone = true;
			this.flush();
		} else {
			this.schedule();
		}
	}

	pushContent(text: string): void {
		this.pushEvent('content', text);
	}

	pushReasoning(text: string): void {
		this.pushEvent('reasoning', text);
	}

	pushToolJson(json: string): void {
		this.pushEvent('tool', json);
	}

	flushAll(): void {
		this.flush();
	}

	onVisible(): void {
		if (this.pendingEvents.length > 0) this.flush();
	}

	onHidden(): void {
		if (this.flushTimer !== null) {
			clearTimeout(this.flushTimer);
			this.flushTimer = null;
		}
	}

	abortFlush(): void {
		this.forceFlush();
		this.destroyed = true;
		if (this.flushTimer !== null) {
			clearTimeout(this.flushTimer);
			this.flushTimer = null;
		}
	}

	destroy(): void {
		this.destroyed = true;
		if (this.flushTimer !== null) {
			clearTimeout(this.flushTimer);
			this.flushTimer = null;
		}
		this.pendingEvents = [];
	}

	getStats(): { chunks: number; commits: number; pendingContent: string; pendingReasoning: string } {
		let pendingContent = '';
		let pendingReasoning = '';
		for (const ev of this.pendingEvents) {
			if (ev.type === 'content') pendingContent += ev.data;
			else if (ev.type === 'reasoning') pendingReasoning += ev.data;
		}
		return { chunks: this.chunks, commits: this.commits, pendingContent, pendingReasoning };
	}

	hasPending(): boolean {
		return this.pendingEvents.length > 0;
	}
}
