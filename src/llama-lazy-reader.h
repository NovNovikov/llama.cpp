#pragma once

// Serves rows of a lazy tensor with explicit reads instead of demand paging.
// This works because the row indices of a whole ubatch are known host-side
// before the graph runs. Hands out F32 rows, like ggml_get_rows does.
//
// POSIX uses pread()s; Windows uses ReadFile with explicit offsets
// (OVERLAPPED) on a second handle opened for random access.

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef _WIN32
// one event per read chunk: concurrent overlapped reads on one handle
// need a private event each, a shared null event misattributes completions
struct llama_lazy_reader_event_guard {
    HANDLE h;
    llama_lazy_reader_event_guard() : h(CreateEventW(nullptr, FALSE, FALSE, nullptr)) {
        GGML_ASSERT(h != nullptr);
    }
    ~llama_lazy_reader_event_guard() {
        CloseHandle(h);
    }
    llama_lazy_reader_event_guard(const llama_lazy_reader_event_guard &) = delete;
    llama_lazy_reader_event_guard & operator=(const llama_lazy_reader_event_guard &) = delete;
};
#endif

struct llama_lazy_reader {
#ifdef _WIN32
    llama_lazy_reader(void * file, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : file(file), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        // F32 rows have no dequantizer; they are staged as-is, like ggml_get_rows
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
        if (file != nullptr) {
            CloseHandle((HANDLE) file);
        }
    }

    void * file; // owned HANDLE from CreateFileW
#else
    llama_lazy_reader(int fd, size_t base, size_t row_size, int64_t n_rows, int n_threads,
                      enum ggml_type type, int64_t head_dim)
        : fd(fd), base(base), row_size(row_size), n_rows(n_rows), n_threads(n_threads),
          head_dim(head_dim), to_float(type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(type)->to_float) {
        // F32 rows have no dequantizer; they are staged as-is, like ggml_get_rows
        GGML_ASSERT((type == GGML_TYPE_F32 || to_float != nullptr) && head_dim > 0);
    }

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    ~llama_lazy_reader() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    const int fd;
#endif

    const size_t   base;       // file offset of row 0
    const size_t   row_size;   // bytes per quantized row
    const int64_t  n_rows;
    const int      n_threads;  // in-flight read workers
    const int64_t  head_dim;   // F32 elements per staged row
    ggml_to_float_t to_float;  // same dequantizer the ggml_get_rows CPU kernel uses

    // fill dst with the n gathered rows, dequantized to F32:
    // dst[slot * head_dim, ...) = to_float(table[rows[slot]])
    // thread-safe; never lets an exception escape a worker thread
    void gather(const int32_t * rows, int64_t n, float * dst) const {
        std::vector<std::pair<int32_t, int32_t>> pairs; // (row, dst slot)
        pairs.reserve((size_t) n);
        for (int64_t i = 0; i < n; ++i) {
            GGML_ASSERT(rows[i] >= 0 && (int64_t) rows[i] < n_rows);
            pairs.emplace_back(rows[i], (int32_t) i);
        }

        std::sort(pairs.begin(), pairs.end()); // equal rows adjacent, file order

        // small gathers are not worth a thread per row
        const int n_workers = (int) std::min<int64_t>(n_threads, std::max<int64_t>(1, n / 32));

        // worker w reads rows pairs[n*w/n_workers, n*(w+1)/n_workers)
        auto run_chunk = [&](int w, std::exception_ptr & err) {
            try {
                run_range(pairs, n * w / n_workers, n * (w + 1) / n_workers, dst);
            } catch (...) {
                err = std::current_exception();
            }
        };

        // an exception leaving a joinable std::thread, or destroying one,
        // terminates the process; keep worker creation failure-safe
        std::vector<std::exception_ptr> errs((size_t) n_workers);
        std::vector<std::thread> workers;
        try {
            for (int w = 1; w < n_workers; ++w) {
                workers.emplace_back([&run_chunk, &errs, w]() {
                    run_chunk(w, errs[(size_t) w]);
                });
            }
        } catch (...) {
            for (auto & t : workers) {
                t.join();
            }
            throw;
        }

        run_chunk(0, errs[0]); // this thread takes the first chunk
        for (auto & t : workers) {
            t.join();
        }

        for (const auto & err : errs) {
            if (err) {
                std::rethrow_exception(err);
            }
        }
    }

private:
#ifdef _WIN32
    // one positional read; ReadFile takes a DWORD length, so chunk big rows
    void read_exact(size_t off, uint8_t * dst, size_t n, HANDLE ev) const {
        while (n > 0) {
            const DWORD want = n > 0x7FFFFFFFu ? 0x7FFFFFFFu : (DWORD) n;
            OVERLAPPED ov = {};
            ov.Offset     = (DWORD) (off & 0xFFFFFFFFu);
            ov.OffsetHigh = (DWORD) (off >> 32);
            ov.hEvent     = ev;
            DWORD done = 0;
            const BOOL ok = ReadFile((HANDLE) file, dst, want, &done, &ov);
            const DWORD err = GetLastError();
            if (!ok) {
                if (err != ERROR_IO_PENDING ||
                        !GetOverlappedResult((HANDLE) file, &ov, &done, TRUE)) {
                    throw std::runtime_error(format("lazy direct read sync=%d err=%lu done=%lu want=%lu off=%zu",
                            (int) ok, (unsigned long) err, (unsigned long) done, (unsigned long) want, off));
                }
            }
            if (done == 0) {
                throw std::runtime_error(format("lazy direct read sync=%d err=%lu done=0 want=%lu off=%zu (unexpected EOF)",
                        (int) ok, (unsigned long) err, (unsigned long) want, off));
            }
            dst += (size_t) done;
            off += (size_t) done;
            n   -= (size_t) done;
        }
    }
#else
    // one positional read; retries short reads and EINTR
    void read_exact(size_t off, uint8_t * dst, size_t n) const {
        for (size_t done = 0; done < n; ) {
            const ssize_t n_read = ::pread(fd, dst + done, n - done, (off_t) (off + done));
            if (n_read < 0 && errno == EINTR) {
                continue; // interrupted by a signal without SA_RESTART
            }
            if (n_read <= 0) {
                throw std::runtime_error(format("lazy direct read of %zu bytes at file offset %zu failed: %s",
                        n, off, n_read == 0 ? "unexpected EOF" : strerror(errno)));
            }
            done += (size_t) n_read;
        }
    }
#endif

    void run_range(const std::vector<std::pair<int32_t, int32_t>> & pairs,
                   int64_t begin, int64_t end, float * dst) const {
#ifdef _WIN32
        llama_lazy_reader_event_guard ev;
#endif
        std::vector<uint8_t> bounce(row_size);
        for (int64_t i = begin; i < end; ) {
            int64_t j = i;
            while (j + 1 < end && pairs[(size_t) j + 1].first == pairs[(size_t) i].first) {
                ++j; // dedup: one read serves the whole run
            }
            const size_t off = base + (size_t) pairs[(size_t) i].first * row_size;
            read_exact(off, bounce.data(), row_size
#ifdef _WIN32
                , ev.h
#endif
                );
            float * first = dst + (size_t) pairs[(size_t) i].second * (size_t) head_dim;
            if (to_float) {
                to_float(bounce.data(), first, head_dim);
            } else {
                memcpy(first, bounce.data(), (size_t) head_dim * sizeof(float));
            }
            for (int64_t k = i + 1; k <= j; ++k) {
                memcpy(dst + (size_t) pairs[(size_t) k].second * (size_t) head_dim, first, (size_t) head_dim * sizeof(float));
            }
            i = j + 1;
        }
    }
};
