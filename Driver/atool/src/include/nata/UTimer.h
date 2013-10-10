/*
 * $Id: UTimer.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __UTIMER_H__
#define __UTIMER_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>



class UTimer {



private:
    __rcsId("$Id: UTimer.h 86 2012-07-30 05:33:07Z m-hirano $");

    struct timeval mStart;
    struct timeval mStop;
    struct timeval mOrigin;



public:
    UTimer(void) {
        (void)gettimeofday(&mStart, NULL);
        mOrigin = mStop = mStart;
    }

    UTimer(const UTimer &obj) {
        mStart = obj.mStart;
        mStop = obj.mStart;
        mOrigin = obj.mOrigin;
    };

    void
    start(void) {
        (void)gettimeofday(&mStart, NULL);
    }


    void
    restart(void) {
        start();
    }


    double
    elapsed(void) {
        (void)gettimeofday(&mStop, NULL);

        return (double)((mStop.tv_sec * 1000000.0 + mStop.tv_usec) -
                       (mStart.tv_sec * 1000000.0 + mStart.tv_usec));
    }


    double
    elapsedFromOrigin(void) {
        (void)gettimeofday(&mStop, NULL);

        return (double)((mStop.tv_sec * 1000000.0 + mStop.tv_usec) -
                       (mOrigin.tv_sec * 1000000.0 + mOrigin.tv_usec));
    }


    double
    now(void) {
        (void)gettimeofday(&mStop, NULL);

        return (double)(mStop.tv_sec * 1000000.0 + mStop.tv_usec);
    }


    double
    started(void) {
        return (double)(mStart.tv_sec * 1000000.0 + mStart.tv_usec);
    }

    double
    origin(void) {
        return (double)(mOrigin.tv_sec * 1000000.0 + mOrigin.tv_usec);
    }

    void
    sleep(uint64_t usec) {
        if (usec > 0) {
            struct timespec t, t1;

            t.tv_sec = (time_t)(usec / 1000000LL);
            t.tv_nsec = (long)((usec % 1000000LL) * 1000LL);

            Retry:
            errno = 0;
            if (nanosleep(&t, &t1) < 0) {
                if (errno == EINTR) {
                    t = t1;
                    goto Retry;
                }
            }
        }
    }
};


#endif // ! __UTIMER_H__
