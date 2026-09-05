/**
 * chatProcessingStore - Per-conversation processing state
 *
 * Owns the live processing snapshot shown while a conversation streams:
 * token counts, tokens/sec, prompt progress. Updated from stream timings,
 * restored from persisted message timings when a conversation loads.
 *
 * Composed under chatStore.processing; not exported from the stores barrel.
 */

import { MessageRole } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { modelsStore } from '$lib/stores/models/index.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type {
	ApiProcessingState,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	DatabaseMessage
} from '$lib/types';
import { SvelteMap } from 'svelte/reactivity';

interface ProcessingTimingData {
	cache_n: number;
	predicted_n: number;
	predicted_per_second: number;
	prompt_ms?: number;
	prompt_n: number;
	prompt_progress?: ChatMessagePromptProgress;
}

export class ChatProcessingStore {
	private _activeConversationId = $state<string | null>(null);
	private states = new SvelteMap<string, ApiProcessingState>();

	/** Processing state of the conversation currently shown in the UI. */
	activeState = $derived(
		this._activeConversationId ? (this.states.get(this._activeConversationId) ?? null) : null
	);

	get activeConversationId(): string | null {
		return this._activeConversationId;
	}

	// Latest-wins throttle for the live timings display. With
	// timings_per_token the server emits per chunk; writing a fresh
	// ApiProcessingState into the reactive SvelteMap on every token would
	// keep a per-token reactive path alive after the content batching.
	// Visual updates go out about every 300ms (~3.3Hz), the newest event always wins, and
	// flushStreamTimings() publishes the latest event on completion so the
	// final numbers are exact. (Final persisted accuracy lives in
	// message.timings, written by the completion handlers.)
	private static readonly TIMINGS_UI_FLUSH_MS = 300;
	private timingsPending = new Map<
		string,
		{ timings?: ChatMessageTimings; promptProgress?: ChatMessagePromptProgress }
	>();
	private timingsTimers = new Map<string, ReturnType<typeof setTimeout>>();

	/**
	 * Applies a stream timings event (tokens/sec + token counts) to the given
	 * conversation's processing state. Shared by the chat and continue flows.
	 * Throttled latest-wins: per-token events coalesce into ~3.3Hz visual updates.
	 */
	applyStreamTimings(
		timings?: ChatMessageTimings,
		promptProgress?: ChatMessagePromptProgress,
		conversationId?: string
	): void {
		const targetId = conversationId || this._activeConversationId;

		if (!targetId) return;

		this.timingsPending.set(targetId, { promptProgress, timings });

		if (!this.timingsTimers.has(targetId)) {
			this.timingsTimers.set(
				targetId,
				setTimeout(() => {
					this.timingsTimers.delete(targetId);
					this.flushStreamTimings(targetId);
				}, ChatProcessingStore.TIMINGS_UI_FLUSH_MS)
			);
		}
	}

	/** Publish the latest pending timings event now (completion path). */
	flushStreamTimings(conversationId?: string): void {
		const targetId = conversationId || this._activeConversationId;

		if (!targetId) return;

		const timer = this.timingsTimers.get(targetId);

		if (timer !== undefined) {
			clearTimeout(timer);
			this.timingsTimers.delete(targetId);
		}

		const pending = this.timingsPending.get(targetId);

		if (!pending) return;

		this.timingsPending.delete(targetId);

		const { promptProgress, timings } = pending;
		const tokensPerSecond =
			timings?.predicted_ms && timings?.predicted_n
				? (timings.predicted_n / timings.predicted_ms) * 1000
				: 0;

		this.updateFromTimings(
			{
				cache_n: timings?.cache_n || 0,
				predicted_n: timings?.predicted_n || 0,
				predicted_per_second: tokensPerSecond,
				prompt_ms: timings?.prompt_ms,
				prompt_n: timings?.prompt_n || 0,
				prompt_progress: promptProgress
			},
			targetId
		);
	}

	/** Drop pending timings without publishing (teardown/stop path). */
	dropStreamTimings(conversationId?: string): void {
		const targetId = conversationId || this._activeConversationId;

		if (!targetId) return;

		const timer = this.timingsTimers.get(targetId);

		if (timer !== undefined) {
			clearTimeout(timer);
			this.timingsTimers.delete(targetId);
		}

		this.timingsPending.delete(targetId);
	}

	getConversationIds(): string[] {
		return Array.from(this.states.keys());
	}

	getState(conversationId: string): ApiProcessingState | null {
		return this.states.get(conversationId) ?? null;
	}

	restoreFromMessages(messages: DatabaseMessage[], conversationId: string): void {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];

			if (message.role === MessageRole.ASSISTANT && message.timings) {
				this.setState(
					conversationId,
					this.parseTimingData({
						cache_n: message.timings.cache_n || 0,
						predicted_n: message.timings.predicted_n || 0,
						predicted_per_second:
							message.timings.predicted_n && message.timings.predicted_ms
								? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
								: 0,
						prompt_ms: message.timings.prompt_ms,
						prompt_n: message.timings.prompt_n || 0
					})
				);

				return;
			}
		}
	}

	setActiveConversation(conversationId: string | null): void {
		this._activeConversationId = conversationId;
	}

	/** Passing null clears the state for the conversation. */
	setState(conversationId: string, state: ApiProcessingState | null): void {
		if (state === null) this.states.delete(conversationId);
		else this.states.set(conversationId, state);
	}

	updateFromTimings(timingData: ProcessingTimingData, conversationId?: string): void {
		const targetId = conversationId || this._activeConversationId;

		if (targetId) {
			this.setState(targetId, this.parseTimingData(timingData));
		}
	}

	private getContextTotal(): number | null {
		const activeConvId = this._activeConversationId;
		const activeState = activeConvId ? this.getState(activeConvId) : null;

		if (activeState && typeof activeState.contextTotal === 'number' && activeState.contextTotal > 0)
			return activeState.contextTotal;

		if (serverStore.isRouterMode) {
			const modelContextSize = modelsStore.selectedModelContextSize;

			if (typeof modelContextSize === 'number' && modelContextSize > 0) {
				return modelContextSize;
			}
		} else {
			const propsContextSize = serverStore.contextSize;

			if (typeof propsContextSize === 'number' && propsContextSize > 0) {
				return propsContextSize;
			}
		}

		return null;
	}

	private parseTimingData(timingData: ProcessingTimingData): ApiProcessingState {
		const cacheTokens = timingData.cache_n || 0,
			predictedTokens = timingData.predicted_n || 0,
			promptMs = timingData.prompt_ms || undefined,
			promptTokens = timingData.prompt_n || 0,
			tokensPerSecond = timingData.predicted_per_second || 0;
		const promptProgress = timingData.prompt_progress;
		const contextTotal = this.getContextTotal();
		const currentConfig = settingsStore.config;
		const outputTokensMax = currentConfig.max_tokens || -1;
		const contextUsed = promptTokens + cacheTokens + predictedTokens,
			outputTokensUsed = predictedTokens;
		const progressCache = promptProgress?.cache || 0,
			progressActualDone = (promptProgress?.processed ?? 0) - progressCache,
			progressActualTotal = (promptProgress?.total ?? 0) - progressCache;
		const progressPercent = promptProgress
			? Math.round((progressActualDone / progressActualTotal) * 100)
			: undefined;

		return {
			cacheTokens,
			contextTotal,
			contextUsed,
			hasNextToken: predictedTokens > 0,
			outputTokensMax,
			outputTokensUsed,
			progressPercent,
			promptMs,
			promptProgress,
			promptTokens,
			speculative: false,
			status: predictedTokens > 0 ? 'generating' : promptProgress ? 'preparing' : 'idle',
			temperature: currentConfig.temperature ?? 0.8,
			tokensDecoded: predictedTokens,
			tokensPerSecond,
			tokensRemaining: outputTokensMax - predictedTokens,
			topP: currentConfig.top_p ?? 0.95
		};
	}
}

export const chatProcessingStore = new ChatProcessingStore();
