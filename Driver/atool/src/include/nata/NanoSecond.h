/*
 * $Id: NanoSecond.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NANOSECOND_H__
#define __NANOSECOND_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>
#include <nata/Statistics.h>


#define CURRENT_TIME_IN_NANOS	ULLONG_MAX
#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
#define USE_TIMEB64
#endif // NATA_API_WIN32API && !NATA_OS_WINDOWSCE


#define __Kilo	1000LL
#define __Mega	1000000LL
#define __Giga	1000000000LL





namespace NanoSecondStatics {
    typedef struct timespec timespecT;
    typedef struct timeval timevalT;

#if defined(NATA_API_WIN32API)
#if !defined(NATA_OS_WINDOWSCE)
#ifdef USE_TIMEB64
    typedef struct __timeb64 timeb64T;
#else
    typedef struct _timeb timeb64T;
#endif // USE_TIMEB64
#endif // ! NATA_OS_WINDOWSCE
#endif // NATA_API_WIN32API

    typedef void (*__nanoSleeperProcT)(uint64_t nsec);
    typedef uint64_t (*__getCurrentTimeInNanosProcT)(void);

    extern __nanoSleeperProcT sleepProc;
    extern __getCurrentTimeInNanosProcT getTimeProc;

    extern uint64_t minSleep;

    extern void initialize(int n = 1000000);
    extern void calibrate(int n = 1000000);

}





class NanoSecond {


private:
    __rcsId("$Id: NanoSecond.h 86 2012-07-30 05:33:07Z m-hirano $");


    typedef struct timespec timespecT;
    typedef struct timeval timevalT;
#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
#ifdef USE_TIMEB64
    typedef struct __timeb64 timeb64T;
#else
    typedef struct _timeb timeb64T;
#endif // USE_TIMEB64
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE

    timespecT mTS;
    uint64_t mNSec;





    static inline uint64_t
    pNsecToUsec(uint64_t nsec) {
        return ((nsec / 100LL) + 5LL) / 10LL;
    }


    static inline uint64_t
    pUsecToNsec(uint64_t usec) {
        return usec * __Kilo;
    }


    static inline uint64_t
    pNsecToMsec(uint64_t nsec) {
        return ((nsec / 100000LL) + 5LL) / 10LL;
    }


    static inline uint64_t
    pMsecToNsec(uint64_t msec) {
        return msec * __Mega;
    }


    static inline void
    pTimespecToNsec(uint64_t &to, const timespecT &from) {
        to = 
            (uint64_t)(from.tv_sec) * __Giga +
            (uint64_t)(from.tv_nsec);
    }


    static inline void
    pNsecToTimespec(timespecT &to, uint64_t &from) {
        uint64_t sec = from / __Giga;
        uint64_t nsec = from % __Giga;
        to.tv_sec = (time_t)sec;
        to.tv_nsec = (long)nsec;
    }


    static inline void
    pTimevalToNsec(uint64_t &to, const timevalT &from) {
#if 0
        uint64_t usec = 
            (uint64_t)(from.tv_sec) * __Mega +
            (uint64_t)(from.tv_usec);
        to = pUsecToNsec(usec);
#else
        to =
            (uint64_t)(from.tv_sec) * __Giga +
            (uint64_t)(from.tv_usec) * __Kilo;
#endif
    }


    static inline void
    pNsecToTimeval(timevalT &to, uint64_t &from) {
        uint64_t orgUsec = pNsecToUsec(from);
        uint64_t sec = orgUsec / __Mega;
        uint64_t usec = orgUsec % __Mega;
        to.tv_sec = (timeval_sec_t)sec;
        to.tv_usec = (timeval_usec_t)usec;
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    static inline void
    pTimeb64ToNsec(uint64_t &to, const timeb64T &from) {
        to = 
            (uint64_t)from.time * __Giga +
            (uint64_t)from.millitm * __Mega;
    }


    static inline void
    pNsecToTimeb64(timeb64T &to, uint64_t &from) {
        uint64_t orgMsec = pNsecToMsec(from);
        uint64_t sec = orgMsec / __Kilo;
        uint64_t msec = orgMsec % __Kilo;
        to.time = (__time64_t)sec;
        to.millitm = (unsigned short)msec;
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE


    static inline void
    pTimevalToTimespec(timespecT &to, const timevalT &from) {
        uint64_t nsec;
        pTimevalToNsec(nsec, from);
        pNsecToTimespec(to, nsec);
    }


    static inline void
    pTimespecToTimeval(timevalT &to, const timespecT &from) {
        uint64_t nsec;
        pTimespecToNsec(nsec, from);
        pNsecToTimeval(to, nsec);
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    static inline void
    pTimeb64ToTimespec(timespecT &to, const timeb64T &from) {
        uint64_t nsec;
        pTimeb64ToNsec(nsec, from);
        pNsecToTimespec(to, nsec);
    }


    static inline void
    pTimespecToTimeb64(timeb64T &to, const timespecT &from) {
        uint64_t nsec;
        pTimespecToNsec(nsec, from);
        pNsecToTimeb64(to, nsec);
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE


    static inline uint64_t
    pGetCurrentTimeInNanos(void) {
#if defined(NATA_API_POSIX)
        return NanoSecondStatics::getTimeProc();
#elif defined(NATA_API_WIN32API)
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
        return NanoSecondStatics::getTimeProc();
#endif // NATA_OS_WINDOWSCE
#else
#error Unknown/Non-supported API.
#endif // NATA_API_POSIX, NATA_API_WIN32API
    }


    static inline void
    pSleep(uint64_t nsec) {
        if (NanoSecondStatics::minSleep < nsec) {
            nsec = (uint64_t)
                ((double)(nsec - NanoSecondStatics::minSleep) * 0.997);
        } else {
            nsec = 1;
        }
#if defined(NATA_API_POSIX) || defined(NATA_API_WIN32API)
        NanoSecondStatics::sleepProc(nsec);
#else
#error Unknown/Non-supported API.
#endif // NATA_API_POSIX, NATA_API_WIN32API
    }


    inline void
    pSleepSelf(void) {
        pSleep(mNSec);
    }


public:


    inline
    NanoSecond(void) {
        mTS.tv_sec = (time_t)0;
        mTS.tv_nsec = (long)0;
        mNSec = 0LL;
    }


    inline
    NanoSecond(uint64_t nsec) {
        if (nsec == CURRENT_TIME_IN_NANOS) {
            mNSec = pGetCurrentTimeInNanos();
            pNsecToTimespec(mTS, mNSec);
        } else {
            mNSec = nsec;
            pNsecToTimespec(mTS, nsec);
        }
    }


    inline
    NanoSecond(timespecT &t) {
        mTS = t;
        pTimespecToNsec(mNSec, mTS);
    }


    inline
    NanoSecond(timevalT &t) {
        pTimevalToNsec(mNSec, t);
        pNsecToTimespec(mTS, mNSec);
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    inline
    NanoSecond(timeb64T &t) {
        pTimeb64ToNsec(mNSec, t);
        pNsecToTimespec(mTS, mNSec);
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE


    inline
    NanoSecond(const NanoSecond &obj) {
        mTS = obj.mTS;
        mNSec = obj.mNSec;
    }





    // assign obj
    inline NanoSecond &
    operator = (const NanoSecond &obj) {
        mTS = obj.mTS;
        mNSec = obj.mNSec;
        return *this;
    }


    // assign uint64_t
    inline NanoSecond &
    operator = (uint64_t nsec) {
        if (nsec == CURRENT_TIME_IN_NANOS) {
            mNSec = pGetCurrentTimeInNanos();
            pNsecToTimespec(mTS, mNSec);
        } else {
            mNSec = nsec;
            pNsecToTimespec(mTS, nsec);
        }
        return *this;
    }


    // assign timespec
    inline NanoSecond &
    operator = (const timespecT &ts) {
        mTS = ts;
        pTimespecToNsec(mNSec, ts);
        return *this;
    }


    // assign timeval
    inline NanoSecond &
    operator = (const timevalT &tv) {
        pTimevalToNsec(mNSec, tv);
        pNsecToTimespec(mTS, mNSec);
        return *this;
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    // assign timeb64
    inline NanoSecond &
    operator = (const timeb64T &tv) {
        pTimeb64ToNsec(mNSec, tv);
        pNsecToTimespec(mTS, mNSec);
        return *this;
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE





    // cast to uint64_t
    inline
    operator uint64_t() {
        return mNSec;
    }


    // cast to timespec
    inline
    operator timespecT() {
        return mTS;
    }


    // cast to timeval
    inline
    operator timevalT() {
        timevalT ret;
        pNsecToTimeval(ret, mNSec);
        return ret;
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    // cast to timeb64
    inline
    operator timeb64T() {
        timeb64T ret;
        pNsecToTimeb64(ret, mNSec);
        return ret;
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE





    static inline uint64_t
    getCurrentTimeInNanos(void) {
        return pGetCurrentTimeInNanos();
    }


    static inline void
    nanosecToTimespec(timespecT &to, uint64_t &from) {
        pNsecToTimespec(to, from);
    }


    static inline void
    timespecToNanosec(uint64_t &to, timespecT &from) {
        pTimespecToNsec(to, from);
    }


    static inline void
    timespecToTimeval(timevalT &to, timespecT &from) {
        pTimespecToTimeval(to, from);
    }


    static inline void
    timevalToTimespec(timespecT &to, timevalT &from) {
        pTimevalToTimespec(to, from);
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    static inline void
    timespecToTimeb64(timeb64T &to, const timespecT &from) {
        pTimespecToTimeb64(to, from);
    }


    static inline void
    timeb64ToTimespec(timespecT &to, const timeb64T &from) {
        pTimeb64ToTimespec(to, from);
    }


    static inline void
    nanosecToTimeb64(timeb64T &to, uint64_t &from) {
        pNsecToTimeb64(to, from);
    }


    static inline void
    timeb64ToNanosec(uint64_t &to, const timeb64T &from) {
        pTimeb64ToNsec(to, from);
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE


    static inline void
    nanoSleep(uint64_t nsec) {
        pSleep(nsec);
    }


    static inline void
    calibrate(int n = 100) {
        NanoSecondStatics::calibrate(n);
    }


    static inline void
    initialize(int n = 100) {
        NanoSecondStatics::initialize(n);
    }





    inline timespecT
    toTimespec(timespecT &t) {
        t = mTS;
        return t;
    }


    inline timevalT
    toTimeval(timevalT &t) {
        pNsecToTimeval(t, mNSec);
        return t;
    }


#if defined(NATA_API_WIN32API) && !defined(NATA_OS_WINDOWSCE)
    inline timeb64T
    toTimeb64(timeb64T &t) {
        pNsecToTimeb64(t, mNSec);
        return t;
    }
#endif // NATA_API_WIN32API && ! NATA_OS_WINDOWSCE


    inline uint64_t
    toNanosec(uint64_t &t) {
        t = mNSec;
        return t;
    }


    inline void
    sleep(void) {
        pSleepSelf();
    }

};


#undef __Kilo
#undef __Mega
#undef __Giga


#endif // __NANOSECOND_H__
