# Coroutines in C Without the Assembly Headache

I wanted coroutines in C. Not a library with thousands of lines of platform-specific assembly, not a dependency on Boost or libuv - just a simple, portable implementation I could understand and modify.

The usual approach to coroutines involves writing a context-switching routine in assembly for every target architecture. Save all the registers, swap the stack pointer, restore the other context's registers. Do this for x86_64, ARM64, x86, and suddenly you're maintaining hundreds of lines of assembly with subtle ABI differences between Linux, macOS, and Windows.

There had to be a simpler way.

**A note on scope:** This is not a standards-lawyer-approved coroutine implementation, nor is it a drop-in replacement for C++20 coroutines. It's a pragmatic, POSIX-oriented approach that trades a small amount of theoretical purity for simplicity, debuggability, and portability across real-world platforms. It's not designed for signal-heavy or real-time contexts - it's meant for structured concurrency in things like build systems, game scripting, or async I/O multiplexing.

## The Problem with setjmp

My first thought was `setjmp` and `longjmp`. They already save and restore CPU registers - that's exactly what context switching needs. The classic pattern:

```c
if (setjmp(buf) == 0) {
    // First call - do something
} else {
    // Returned via longjmp
}
```

But there's a fundamental problem: `setjmp` saves the CPU register state, including the stack pointer, but it does *not* preserve the contents of the stack itself. When a coroutine yields, its local variables live on the stack. If execution continues on the same stack, the memory backing those locals will be reused and overwritten.

For coroutines to work, each one needs its own stack.

## Allocating Stacks

Getting a separate stack is straightforward:

```c
void *stack = mmap(NULL, 64 * 1024,
    PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
```

On Windows, `VirtualAlloc` serves the same purpose, though the semantics differ slightly and MSVC's `setjmp` requires some care.

We now have 64KB of memory that can serve as a stack. But here's where it gets tricky: how do we actually *use* this stack?

The stack pointer is a CPU register. To make function calls use our new stack, we need to change that register. And in C, there's no portable way to do this. We're forced into assembly.

## The Insight

This is where most implementations go off the deep end. They write full context-switching routines - save every register, swap stacks, restore every register. For each architecture. With different calling conventions on each OS.

But wait. We already have code that saves and restores registers: `setjmp` and `longjmp`. What if we could combine them with just *minimal* assembly?

The insight is this: we only need assembly to switch stacks once, at coroutine startup. After that, `setjmp`/`longjmp` handle everything.

Here's the entire architecture-specific code for x86_64:

```c
static void coro_switch_stack(void *new_sp, void (*func)(void)) {
    __asm__ volatile(
        "mov %0, %%rsp\n\t"
        "call *%1\n\t"
        : : "r"(new_sp), "r"(func)
        : "memory"
    );
}
```

Five lines. ARM64 is similarly short:

```c
static void coro_switch_stack(void *new_sp, void (*func)(void)) {
    __asm__ volatile(
        "mov sp, %0\n\t"
        "blr %1\n\t"
        : : "r"(new_sp), "r"(func)
        : "memory"
    );
}
```

That's the entire platform-specific surface. Everything else is portable C - or more precisely, POSIX C with glibc/BSD libc assumptions. We're relying on implementation-defined behavior of `setjmp`/`longjmp` across stack boundaries, which works reliably on real-world POSIX systems but isn't guaranteed by ISO C.

## How It Works

The key invariant that makes this safe: once a coroutine starts, it *never leaves its own stack*. All subsequent yields and resumes happen via `setjmp`/`longjmp` within that stack. The stack pointer saved by `setjmp` always points into the coroutine's allocated region.

The mechanism is a trampoline function. Conceptually, the trampoline turns "call a function on a new stack" into "enter a new world where `setjmp` now sees that stack as home."

When we start a coroutine:

1. Save the caller's context with `setjmp`
2. Switch to the new stack (our tiny bit of assembly)
3. Call a trampoline that invokes the user's function
4. When the user calls `yield`, we `setjmp` to save the coroutine's state, then `longjmp` back to the caller
5. When the caller resumes, we `longjmp` back into the coroutine

Here's what memory looks like with two stacks:

```
    MAIN STACK                          COROUTINE STACK
    (process default)                   (mmap'd region)

    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │   [guard page]  │ <- PROT_NONE
    │    main()       │                 ├─────────────────┤
    │    locals       │                 │                 │
    ├─────────────────┤                 │                 │
    │   loop_run()    │                 │   (free space)  │
    │    locals       │                 │                 │
    ├─────────────────┤                 │                 │
    │  coro_resume()  │                 ├─────────────────┤
    │  caller_ctx ◄───────────────────────── longjmp ─────┤
    ├─────────────────┤                 ├─────────────────┤
    │                 │                 │  user_func()    │
    │                 │                 │   locals        │
    │                 │                 ├─────────────────┤
    │                 │                 │  coro_yield()   │
    │                 │                 │   coro_ctx ─────────► setjmp
    │                 │                 ├─────────────────┤
    │                 │                 │     args        │ <- copied here
    └─────────────────┘                 └─────────────────┘
          ▲                                   ▲
          │                                   │
       SP (when in main)                   SP (when in coroutine)
```

The flow of a yield and resume:

```
    resume(co)                              user_func()
        │                                       │
        ▼                                       ▼
    setjmp(caller_ctx) ◄─────────────┐     do work...
        │                            │          │
        │ (first time: returns 0)    │          ▼
        ▼                            │     YIELD
    switch_stack ──────────┐         │          │
                           │         │          ▼
                           ▼         │     setjmp(coro_ctx)
                      trampoline     │          │
                           │         │          │ (returns 0)
                           ▼         │          ▼
                      user_func()    │     longjmp(caller_ctx, 1)
                           │         │          │
                           ▼         └──────────┘
                        YIELD              (resume returns)
                                                │
                                                ▼
    - - - - - - - - - - - - - - - - - - - - - - - - - - -

    resume(co) again                        user_func()
        │                                  (continues)
        ▼                                       ▲
    setjmp(caller_ctx)                          │
        │                                       │
        │ (returns 0)                           │
        ▼                                       │
    longjmp(coro_ctx, 1) ───────────────────────┘
                                          (setjmp returns 1)
```

```c
static void coro_trampoline(void) {
    coro_t *co = get_current();
    co->func(co->arg);
    co->status = CORO_DEAD;
    longjmp(co->caller_ctx, 1);
}
```

The beauty is that after the initial stack switch, we never touch assembly again. The `setjmp` in `yield` captures the full register state on the coroutine's stack. The `longjmp` in `resume` restores it perfectly. The C runtime does all the hard work.

## The glibc Trap

This worked beautifully on my first test. Then I tried a build with optimizations enabled, and it crashed.

The culprit was glibc's `_FORTIFY_SOURCE`. When enabled, glibc replaces `longjmp` with `__longjmp_chk`, which validates that the stack pointer is within "reasonable" bounds. With multiple stacks, this check fails spectacularly.

The fix is to use `_setjmp` and `_longjmp` instead. These are the raw versions that skip the security check. As a bonus, they're faster - they don't save and restore the signal mask.

```c
#if defined(__GLIBC__)
# define coro_setjmp(buf)      _setjmp(buf)
# define coro_longjmp(buf, v)  _longjmp(buf, v)
#endif
```

This is glibc-specific behavior. Other libcs - musl, BSD libc, macOS libSystem - do not perform this check and work without special handling.

I also added a compile-time check that errors out if someone tries to build with `_FORTIFY_SOURCE` on glibc without this workaround. Better to fail loudly than debug mysterious crashes.

## Guard Pages

With multiple stacks comes the risk of stack overflow going undetected. Normally, if you overflow the main stack, you hit unmapped memory and get a segfault. With mmap'd stacks, you might just corrupt adjacent memory.

The solution is guard pages:

```c
void *stack = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
mprotect(stack, page_size, PROT_NONE);  // First page is inaccessible
```

The bottom page of each stack is marked as no-access. Overflow hits this page and crashes immediately, rather than silently corrupting memory.

## Making It Useful

Raw coroutines are interesting but not particularly useful on their own. The real power comes from combining them with an event loop.

The pattern is simple: when a coroutine needs to wait for I/O, it registers its file descriptor with epoll (or kqueue on macOS), then yields. The event loop runs other coroutines. When the fd becomes ready, the original coroutine is resumed.

```c
CORO_DEF(fetch, args, { int socket; }) {
    char buf[4096];

    WAIT_FD(args->socket);  // Yields until readable
    int n = read(args->socket, buf, sizeof(buf));

    // Process data...
}
```

From the coroutine's perspective, `WAIT_FD` looks like a blocking call. But behind the scenes, the event loop is running other coroutines while this one waits.

## The Macro Layer

One ergonomic issue remained: passing arguments to coroutines. Since each coroutine has its own stack, we can't just pass pointers to the caller's local variables - they might be gone by the time the coroutine runs.

The solution is to copy arguments onto the coroutine's stack at creation time. A set of macros makes this transparent:

```c
CORO_DEF(worker, args, { int id; char *name; }, {}, { int result; }) {
    printf("Worker %d: %s\n", args->id, args->name);
    RETURN_T(worker, .result = 42);
}

// Usage
t_future *f = RUN(loop, worker, .id = 1, .name = "test");
CORO_RET(worker) *ret = AWAIT(f);
printf("Result: %d\n", ret->result);
```

The `RUN` macro copies the argument struct onto the coroutine's stack. The coroutine receives a pointer to this copy. No lifetime issues, no manual memory management.

Yes, this is macro-heavy. The alternative is either heap-allocating argument blocks manually or forcing users into unsafe lifetime rules. The macros hide the ceremony while keeping the underlying mechanism simple.

## Signals: Here Be Dragons

Signals and coroutines have a complicated relationship. Understanding it is essential if your coroutines will interact with child processes, timers, or any signal-generating code.

### The Problem

When a signal arrives, the kernel interrupts whatever is currently running and invokes the handler on the *current stack*. With coroutines, that means:

```
    Signal arrives while coroutine is running:

    MAIN STACK                          COROUTINE STACK
    ┌─────────────────┐                 ┌─────────────────┐
    │                 │                 │  signal_handler │ <- runs HERE
    │   (suspended)   │                 ├─────────────────┤
    │                 │                 │  user_func()    │
    │                 │                 │   (interrupted) │
    └─────────────────┘                 └─────────────────┘
```

This creates several issues:

1. **Stack overflow risk** - The handler runs on the coroutine's limited stack (64KB by default). A complex handler could overflow it.

2. **Shared signal masks** - Signal masks are per-thread, not per-coroutine. If one coroutine blocks `SIGUSR1`, all coroutines have it blocked.

3. **EINTR interruptions** - Signals interrupt `epoll_wait`/`kevent` with `EINTR`. The event loop must handle this.

4. **No longjmp from handlers** - Attempting to yield or longjmp from a signal handler is undefined behavior and will likely crash.

### Solutions

**For stack protection**, use `sigaltstack` to run handlers on a dedicated stack:

```c
char *alt_stack = malloc(64 * 1024);
stack_t ss = { .ss_sp = alt_stack, .ss_size = 64*1024 };
sigaltstack(&ss, NULL);

struct sigaction sa = {
    .sa_handler = handler,
    .sa_flags = SA_ONSTACK  // Key flag
};
```

**For event loop integration**, avoid handlers entirely. On Linux, `signalfd` converts signals into file descriptor events:

```c
sigset_t mask;
sigaddset(&mask, SIGUSR1);
sigprocmask(SIG_BLOCK, &mask, NULL);  // Block traditional delivery

int sigfd = signalfd(-1, &mask, SFD_NONBLOCK);

// In coroutine - signals become regular I/O:
WAIT_FD(sigfd);
read(sigfd, &info, sizeof(info));
```

This integrates perfectly with epoll - no handler, no stack concerns, the coroutine decides when to handle the signal.

**For portable code**, use the self-pipe trick:

```c
static int signal_pipe[2];

void handler(int sig) {
    char c = sig;
    write(signal_pipe[1], &c, 1);  // async-signal-safe
}

// In coroutine:
WAIT_FD(signal_pipe[0]);
read(signal_pipe[0], &signo, 1);
```

The handler is minimal (just a write), and all real logic lives in the coroutine.

### The EINTR Fix

Signals can interrupt blocking syscalls. The event loop must retry:

```c
static void loop_poll_io(t_loop *loop) {
    int n;
    do {
        n = epoll_wait(loop->epoll_fd, events, 64, -1);
    } while (n < 0 && errno == EINTR);
    // ...
}
```

Without this, a stray `SIGCHLD` from a child process can cause the event loop to exit prematurely.

### Summary

| Approach | Stack Safe | Coroutine Control | Portable |
|----------|------------|-------------------|----------|
| sigaltstack | Yes | No | Yes |
| signalfd | N/A | Yes | Linux |
| Self-pipe | Handler only | Yes | Yes |

For a build system spawning child processes, `signalfd` (or `pidfd` for process-specific events) is ideal. For portable code, the self-pipe trick works everywhere.

## Was It Worth It?

The final implementation is around 500 lines of C. It runs on Linux, macOS, and Windows. It supports x86_64 and ARM64. The total architecture-specific assembly is about 10 lines.

Compare this to traditional approaches with hundreds of lines of assembly per platform, and the tradeoff is clear. We give up a small amount of performance (the overhead of `setjmp`/`longjmp` versus hand-optimized assembly), but gain maintainability and portability.

If you need maximum performance or language-level guarantees, hand-written assembly or compiler-supported coroutines may be worth the cost. But if you want coroutines you can read, debug, and maintain - coroutines where you can step through the context switch in gdb and understand exactly what's happening - this approach hits a sweet spot that's surprisingly hard to beat.
