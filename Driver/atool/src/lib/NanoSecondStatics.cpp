#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>
#include <nata/Statistics.h>

#include <nata/NanoSecond.h>

#ifndef __Kilo
#define __Kilo	1000LL
#endif // ! __Kilo
#ifndef __Mega
#define __Mega	1000000LL
#endif // ! __Mega
#ifndef __Giga
#define __Giga	1000000000LL
#endif // ! __Giga





namespace NanoSecondStatics {


    uint64_t minSleep = 0LL;

    static bool isInInitialize = false;

    static Mutex initLock;
    static bool isInitialized = false;

#ifdef NATA_API_WIN32API
    static void w32_initialize();
#endif // NATA_API_WIN32API





    inline void
    calibrate(int n) {
        NanoSecond start;
        NanoSecond end;
        Statistics<uint64_t> st;

        for (int i = 0; i < n; i++) {
            start = CURRENT_TIME_IN_NANOS;
            NanoSecond::nanoSleep(1LL);
            end = CURRENT_TIME_IN_NANOS;
            st.record(end - start);
        }

        minSleep = (uint64_t)st.average();

        dbgMsg("min sleep = " PF64(u) " nsec.\n", minSleep);
    }


    inline void
    initialize(int n) {
        ScopedLock l(&initLock);
        if (isInitialized == false) {
            isInInitialize = true;
#ifdef NATA_API_WIN32API
            w32_initialize();
#endif // NATA_API_WIN32API
            calibrate(n);
            isInInitialize = false;
            isInitialized = true;
        }
    }





#if defined(NATA_API_POSIX)


    static void posix_nanoSleep(uint64_t nsec);
    static uint64_t posix_getCurrentTimeInNanos(void);

    __nanoSleeperProcT sleepProc = posix_nanoSleep;
    __getCurrentTimeInNanosProcT getTimeProc = posix_getCurrentTimeInNanos;
    

    static inline void
    posix_nanoSleep(uint64_t nsec) {
        timespecT t, r;

        NanoSecond::nanosecToTimespec(t, nsec);

        Retry:
        errno = 0;
        if (nanosleep(&t, &r) == 0) {
            return;
        } else {
            if (errno == EINTR) {
                t = r;
                goto Retry;
            } else {
                perror("nanosleep");
                fatal("Must not happen.\n");
            }
        }
    }


    static inline uint64_t
    posix_getCurrentTimeInNanos(void) {
        timespecT ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
            uint64_t ret;
            NanoSecond::timespecToNanosec(ret, ts);
            return ret;
        } else {
            perror("clock_gettime");
            return ULLONG_MAX;
        }
    }





#elif defined(NATA_API_WIN32API)

    static bool HRFreqUsable = false;
    static uint64_t tickInNanos = __Mega;
    static uint64_t highResTickBase = ULLONG_MAX;

    static void w32_initialize();
    static void w32_spinNanoSleep(uint64_t nsec);
    static void w32_nanoSleep(uint64_t nsec);

    static uint64_t w32_getHighResTick(void);
    static uint64_t w32_getCurrentTimeInNanos(void);
    static uint64_t w32_getCurrentTimeInNanosHighRes(void);

    __nanoSleeperProcT sleepProc = w32_nanoSleep;
    __getCurrentTimeInNanosProcT getTimeProc = w32_getCurrentTimeInNanos;


    static inline void
    w32_nanoSleep(uint64_t nsec) {
        Sleep((DWORD)(nsec / __Mega));
    }


    static inline void
    w32_spinNanoSleep(uint64_t nsec) {
        uint64_t now0, now1, elps;

        DWORD msec = (DWORD)(nsec / __Mega);
        if (msec > 0) {
            now0 = NanoSecond::getCurrentTimeInNanos();
            /*
             * Wait msec.
             */
            Sleep(msec);
            /*
             * What time is it now?
             */
            now1 = NanoSecond::getCurrentTimeInNanos();
            elps = now1 - now0;
        } else {
            elps = 0LL;
        }

        if (elps < nsec) {
            /*
             * The spin time.
             */
            uint64_t nTicks = 
                (((nsec - elps) * 10LL + 5) / tickInNanos) / 10LL;

            uint64_t sT = w32_getHighResTick();
            uint64_t eT = sT + nTicks;
            volatile uint64_t c = sT;

            do {
                c = w32_getHighResTick();
            } while (c <= eT);
        }
    }


    static inline uint64_t
    w32_getHighResTick(void) {
        uint64_t now;
        (void)QueryPerformanceCounter((LARGE_INTEGER *)&now);
        if (highResTickBase < now) {
            return now;
        } else {
            /*
             * Counter seems overflowed, VERY RARE CASE THOUGH.
             */
            ScopedLock l(&initLock);

            uint64_t ret = ULLONG_MAX - highResTickBase;
            ret += now;
            highResTickBase = now;
            return ret;
        }
    }


    static inline uint64_t
    w32_getCurrentTimeInNanos(void) {
#if defined(NATA_OS_WINDOWSCE)
        /*
         * Not yet.
         *	Use GetSystemTime() & SystemTimeToFileTime() on WinCE
         *	6.0 and older.
         *	Use GetSystemTimeAsFileTime() on WinCE 7.0 and later.
         */
        fatal("not yet implemented.\n");
        return 0LL;
#else
        uint64_t ret;
        timeb64T curTime;
#if defined(__MINGW64__) || defined(__MINGW32__)
#ifdef USE_TIMEB64
        _ftime64(&curTime);
#else
        _ftime(&curTime);
#endif // USE_TIMEB64
#else
#ifdef USE_TIMEB64
        _ftime64_s(&curTime);
#else
        _ftime(&curTime);
#endif // USE_TIMEB64
#endif // __MINGW64__ || __MINGW32__
        NanoSecond::timeb64ToNanosec(ret, curTime);
        return ret;
#endif // NATA_OS_WINDOWSCE
    }


    static inline uint64_t
    w32_getCurrentTimeInNanosHighRes(void) {
        uint64_t retBase = w32_getCurrentTimeInNanos();
        uint64_t nanos = w32_getHighResTick() * tickInNanos;
        return retBase + (nanos % __Mega);
    }


    static inline void
    w32_initialize(void) {
#if defined(NATA_OS_WINDOWSCE)
        fatal("not yet implemented.\n");
#else
        if (timeBeginPeriod(1) != TIMERR_NOERROR) {
            fatal("Can't set timer resolution to minimum.\n");
        }

        uint64_t freq;

        if (QueryPerformanceFrequency((LARGE_INTEGER *)&freq) == TRUE) {
            double dt = (double)__Giga / (double)freq;
            tickInNanos = (uint64_t)(dt + 0.5);
            HRFreqUsable = true;

            sleepProc = w32_spinNanoSleep;
            getTimeProc = w32_getCurrentTimeInNanosHighRes;

            (void)QueryPerformanceCounter((LARGE_INTEGER *)&highResTickBase);

            dbgMsg("freq " PF64(u) " Hz\n", freq);
            dbgMsg("tick " PF64(u) " ns\n", tickInNanos);
        }
#endif // NATA_OS_WINDOWSCE
    }

#else

#error Unknown/Non-supported API.

#endif // NATA_API_POSIX, NATA_API_WIN32API
}
