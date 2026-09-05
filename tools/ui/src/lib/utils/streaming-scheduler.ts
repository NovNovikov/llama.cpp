/**
 * StreamingScheduler - throttled UI commit scheduler for streaming generation.
 *
 * Batches incoming chunks in a cheap non-reactive staging buffer and flushes
 * to the reactive store at most 3 times per second (400ms). First fragment
 * flushes immediately for TTFT, final chunk and semantic boundaries flush
 * immediately, hidden tabs defer commits until visible.
 *
 * Used by ChatService.handleStreamResponse to reduce per-token work:
 * - string rewrites, store updates, component re-renders, markdown parses,
 *   syntax highlighting, autoscroll, token stats, localStorage all become
 *   ~3Hz instead of per-token (~12Hz at 12 t/s).
 */

export interface StreamingSchedulerCallbacks {
	onContent?: (text: string) => void;
	onReasoning?: (text: string) => void;
	onToolJson?: (json: string) => void;
}

export interface StreamingSchedulerOptions {
	flushIntervalMs?: number; // default 400
}

export class StreamingScheduler {
	private pendingContent = '';
	private pendingReasoning = '';
	private pendingToolJson: string | null = null;
	private flushTimer: ReturnType<typeof setTimeout> | null = null;
	private firstContentDone = false;
	private firstReasoningDone = false;
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
		let didFlush = false;
		if (this.pendingContent) {
			const c = this.pendingContent;
			this.pendingContent = '';
			this.cbs.onContent?.(c);
			didFlush = true;
			this.commits++;
		}
		if (this.pendingReasoning) {
			const r = this.pendingReasoning;
			this.pendingReasoning = '';
			this.cbs.onReasoning?.(r);
			didFlush = true;
			this.commits++;
		}
		if (this.pendingToolJson) {
			const t = this.pendingToolJson;
			this.pendingToolJson = null;
			this.cbs.onToolJson?.(t);
			didFlush = true;
			this.commits++;
		}
		if (didFlush && typeof window !== 'undefined') {
			try {
				const s = (window as unknown as { __LLAMA_STREAM_STATS?: { commits: number } }).__LLAMA_STREAM_STATS;
				if (s) s.commits = this.commits;
			} catch {}
		}
	}

	private schedule(): void {
		if (this.flushTimer !== null) return;
		if (this.isHidden()) return;
		this.flushTimer = setTimeout(() => this.flush(), this.interval);
	}

	pushContent(text: string): void {
		if (this.destroyed || !text) return;
		this.chunks++;
		if (typeof window !== 'undefined') {
			try {
				const s = (window as unknown as { __LLAMA_STREAM_STATS?: { chunks: number } }).__LLAMA_STREAM_STATS;
				if (s) s.chunks = this.chunks;
			} catch {}
		}
		this.pendingContent += text;
		if (!this.firstContentDone) {
			this.firstContentDone = true;
			this.flush();
		} else {
			this.schedule();
		}
	}

	pushReasoning(text: string): void {
		if (this.destroyed || !text) return;
		this.chunks++;
		this.pendingReasoning += text;
		if (!this.firstReasoningDone) {
			this.firstReasoningDone = true;
			this.flush();
		} else {
			this.schedule();
		}
	}

	pushToolJson(json: string): void {
		if (this.destroyed || !json) return;
		this.chunks++;
		this.pendingToolJson = json;
		if (!this.firstContentDone && !this.firstReasoningDone) {
			this.firstContentDone = true;
			this.flush();
		} else {
			this.schedule();
		}
	}

	// Flush immediately - call on stream finish, semantic boundary, or visibility change.
	flushAll(): void {
		this.flush();
	}

	// Called when tab becomes visible - flush staged content.
	onVisible(): void {
		if (this.pendingContent || this.pendingReasoning || this.pendingToolJson) {
			this.flush();
		}
	}

	// Called on abort/error - flush staged to not lose content, then prevent further.
	abortFlush(): void {
		this.flush();
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
		this.pendingContent = '';
		this.pendingReasoning = '';
		this.pendingToolJson = null;
	}

	getStats(): { chunks: number; commits: number; pendingContent: string; pendingReasoning: string } {
		return {
			chunks: this.chunks,
			commits: this.commits,
			pendingContent: this.pendingContent,
			pendingReasoning: this.pendingReasoning
		};
	}

	hasPending(): boolean {
		return !!this.pendingContent || !!this.pendingReasoning || !!this.pendingToolJson;
	}
}
