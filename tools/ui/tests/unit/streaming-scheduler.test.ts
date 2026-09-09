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

	it('preserves chronological order across content/reasoning/tool transitions', () => {
		const log: Array<[string, string]> = [];
		const s = new StreamingScheduler(
			{
				onContent: (c) => log.push(['content', c]),
				onReasoning: (r) => log.push(['reasoning', r]),
				onToolJson: (t) => log.push(['tool', t])
			},
			{ flushIntervalMs: 400 }
		);
		// thinking -> answer transition: pending reasoning must publish before staged content
		s.pushReasoning('r1'); // immediate (first)
		s.pushReasoning('r2'); // pending
		s.pushContent('c1'); // type change -> flush r2 now, stage c1
		expect(log).toEqual([
			['reasoning', 'r1'],
			['reasoning', 'r2']
		]);
		// tool after content: staged content flushes first
		s.pushToolJson('{"a":1}');
		expect(log).toEqual([
			['reasoning', 'r1'],
			['reasoning', 'r2'],
			['content', 'c1']
		]);
		vi.advanceTimersByTime(400);
		expect(log).toEqual([
			['reasoning', 'r1'],
			['reasoning', 'r2'],
			['content', 'c1'],
			['tool', '{"a":1}']
		]);
		s.destroy();
	});

	it('same-type neighbours coalesce, different types do not reorder', () => {
		const log: Array<[string, string]> = [];
		const s = new StreamingScheduler(
			{
				onContent: (c) => log.push(['content', c]),
				onReasoning: (r) => log.push(['reasoning', r])
			},
			{ flushIntervalMs: 400 }
		);
		s.pushContent('a');
		s.pushContent('b');
		s.pushContent('c');
		vi.advanceTimersByTime(400);
		// 'a' immediate, 'bc' coalesced
		expect(log).toEqual([
			['content', 'a'],
			['content', 'bc']
		]);
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

	it('hidden cancels armed timer: no commit fires while hidden', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a'); // immediate
		s.pushContent('b'); // stages, arms timer
		// tab goes hidden before the timer fires
		// @ts-ignore
		global.document.visibilityState = 'hidden';
		s.onHidden();
		vi.advanceTimersByTime(4000);
		expect(onContent).toHaveBeenCalledTimes(1); // timer was cancelled
		// catch-up on return
		// @ts-ignore
		global.document.visibilityState = 'visible';
		s.onVisible();
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0]).toBe('b');
		s.destroy();
	});

	it('first fragment while hidden is deferred, not committed', () => {
		// @ts-ignore
		global.document.visibilityState = 'hidden';
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a');
		vi.advanceTimersByTime(4000);
		expect(onContent).not.toHaveBeenCalled();
		// @ts-ignore
		global.document.visibilityState = 'visible';
		s.onVisible();
		expect(onContent).toHaveBeenCalledWith('a');
		s.destroy();
	});

	it('abort delivers staged chunks even when hidden', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		s.pushContent('a'); // immediate
		// @ts-ignore
		global.document.visibilityState = 'hidden';
		s.pushContent('b');
		s.abortFlush(); // force-flush bypasses hidden deferral
		expect(onContent).toHaveBeenCalledTimes(2);
		expect(onContent.mock.calls[1][0]).toBe('b');
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

	it('200 synchronous chunks coalesce (burst case)', () => {
		const onContent = vi.fn();
		const s = new StreamingScheduler({ onContent }, { flushIntervalMs: 400 });
		for (let i = 0; i < 200; i++) s.pushContent('x');
		vi.advanceTimersByTime(400 * 10);
		// burst case: everything arrives inside one interval -> 1 immediate + 1 timed
		expect(onContent.mock.calls.length).toBeLessThan(10);
		expect(onContent.mock.calls.length).toBeGreaterThan(0);
		s.destroy();
	});

	it('paced 12 chunks/s: ~40 commits for 200 chunks, not 200', () => {
		const out: string[] = [];
		const s = new StreamingScheduler({ onContent: (c) => out.push(c) }, { flushIntervalMs: 400 });
		// ~83ms between chunks ~= 12 t/s over ~16.6s of stream time
		for (let i = 0; i < 200; i++) {
			s.pushContent('x');
			vi.advanceTimersByTime(83);
		}
		vi.advanceTimersByTime(400);
		const commits = out.length;
		// steady state ~= 1 commit per 400ms over 16.6s -> ~40-45, far below 200
		expect(commits).toBeGreaterThan(5);
		expect(commits).toBeLessThan(100);
		expect(out.join('').length).toBe(200);
		s.destroy();
	});
});
