import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { StreamingScheduler } from './streaming-scheduler';

describe('StreamingScheduler', () => {
	beforeEach(() => {
		vi.useFakeTimers();
		// @ts-ignore
		global.document = { visibilityState: 'visible' } as unknown as Document;
		// @ts-ignore
		global.window = { __LLAMA_STREAM_STATS: { chunks: 0, commits: 0, renders: 0, scrolls: 0 } } as unknown as Window & typeof globalThis;
	});
	afterEach(() => {
		vi.useRealTimers();
		vi.restoreAllMocks();
	});

	it('first chunk publishes immediately (TTFT)', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent });
		s.pushContent('hello');
		expect(onContent).toHaveBeenCalledWith('hello');
		expect(onContent).toHaveBeenCalledTimes(1);
		s.destroy();
	});

	it('coalesces fast chunks into fewer commits (3Hz)', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a'); // immediate
		expect(onContent).toHaveBeenCalledTimes(1);
		// next 199 fast chunks should coalesce
		for (let i = 0; i < 199; i++) s.pushContent('x');
		expect(onContent).toHaveBeenCalledTimes(1); // not yet flushed
		vi.advanceTimersByTime(400);
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0].length).toBe(199);
		// total commits 2, not 200
		expect(s.getStats().commits).toBe(2);
		s.destroy();
	});

	it('preserves order and byte-for-byte final text', () => {
		const out: string[] = [];
		const s = new StreamingScheduler({ onContent: (c) => out.push(c) }, { flushIntervalMs: 400 });
		const parts = ['a', 'b', 'c', 'd', 'e'];
		parts.forEach((p) => s.pushContent(p));
		// first is immediate, rest pending
		vi.advanceTimersByTime(400);
		s.flushAll();
		expect(out.join('')).toBe('abcde');
		s.destroy();
	});

	it('final chunk always flushes', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a');
		s.pushContent('b');
		s.flushAll();
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0]).toBe('b');
		s.destroy();
	});

	it('abort/error flushes pending content', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a'); // immediate
		s.pushContent('b');
		s.abortFlush();
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0]).toBe('b');
		// after abort, further pushes are ignored
		s.pushContent('c');
		expect(onContent).toHaveBeenCalledTimes(2);
		s.destroy();
	});

	it('reasoning/content transitions are correct', () => {
		const onContent = vi.fn();
		const onReasoning = vi.fn();
		const s = new StreamingScheduler({ onContent, onReasoning }, { flushIntervalMs: 400 });
		s.pushContent('c1');
		s.pushReasoning('r1');
		expect(onContent).toHaveBeenCalledWith('c1');
		expect(onReasoning).toHaveBeenCalledWith('r1');
		s.pushContent('c2');
		s.pushReasoning('r2');
		expect(onContent).toHaveBeenCalledTimes(1);
		expect(onReasoning).toHaveBeenCalledTimes(1);
		vi.advanceTimersByTime(400);
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onReasoning).toHaveBeenCalledTimes(2);
		s.destroy();
	});

	it('tool-call chunks are not corrupted and coalesce', () => {
		const onTool = vi.fn();
		const s = new StreamingScheduler({ onToolJson: onTool }, { flushIntervalMs: 400 });
		s.pushToolJson('{"a":1}');
		expect(onTool).toHaveBeenCalledWith('{"a":1}');
		s.pushToolJson('{"a":2}');
		s.pushToolJson('{"a":3}');
		// only latest pending should be kept (replaces)
		vi.advanceTimersByTime(400);
		expect(onTool).toHaveBeenCalledTimes(2);
		expect(onTool.mock.calls[1][0]).toBe('{"a":3}');
		s.destroy();
	});

	it('hidden tab defers commits until visible', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a'); // immediate
		expect(onContent).toHaveBeenCalledTimes(1);
		// @ts-ignore
		global.document.visibilityState = 'hidden';
		s.pushContent('b');
		s.pushContent('c');
		vi.advanceTimersByTime(400);
		expect(onContent).toHaveBeenCalledTimes(1); // still deferred
		// @ts-ignore
		global.document.visibilityState = 'visible';
		s.onVisible();
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0]).toBe('bc');
		s.destroy();
	});

	it('timers are cleared on destroy/new generation', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a');
		s.pushContent('b');
		expect(onContent).toHaveBeenCalledTimes(1);
		s.destroy();
		vi.advanceTimersByTime(400);
		expect(onContent).toHaveBeenCalledTimes(1); // no flush after destroy
		// new scheduler should be clean
		const s2 = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s2.pushContent('x');
		expect(onContent).toHaveBeenCalledTimes(2);
		s2.destroy();
	});

	it('no update after destroy', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent });
		s.destroy();
		s.pushContent('a');
		expect(onContent).not.toHaveBeenCalled();
	});

	it('200 fast chunks => tens of commits, not 200', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		for (let i = 0; i < 200; i++) s.pushContent('x');
		vi.advanceTimersByTime(400 * 10);
		// first immediate + up to 1 per 400ms, but all 199 after first coalesced into 1 if within same interval
		// With our implementation, all 199 after first are in one pending and flush once, so total 2
		expect(onContent.mock.calls.length).toBeLessThan(10);
		expect(onContent.mock.calls.length).toBeGreaterThan(0);
		s.destroy();
	});
});
