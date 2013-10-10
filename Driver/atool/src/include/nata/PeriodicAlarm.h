/*
 * $Id: PeriodicAlarm.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __PERIODICALARM_H__
#define __PERIODICALARM_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/nata_perror.h>

#include <nata/Thread.h>
#include <nata/ScopedLock.h>
#include <nata/BoundedBlockingQueue.h>
#include <nata/SynchronizedMap.h>
#include <nata/NanoTimer.h>


class PeriodicAlarmHandle: public BoundedBlockingQueue<uint64_t> {


public:
    inline
    PeriodicAlarmHandle(size_t n = BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH):
        BoundedBlockingQueue<uint64_t>(n) {
    }


private:
    PeriodicAlarmHandle(const PeriodicAlarmHandle &obj);
    PeriodicAlarmHandle operator = (const PeriodicAlarmHandle &obj);



public:
    inline bool
    wait(uint64_t &notifiedTime, int64_t timedout = -1, 
         BlockingContainer::ContainerStatus *sPtr = NULL) {
        return get(notifiedTime, timedout, sPtr);
    }


    inline bool
    notify(int64_t timedout = -1,
           BlockingContainer::ContainerStatus *sPtr = NULL) {
        uint64_t curTime = NanoSecond::getCurrentTimeInNanos();
        return put(curTime, timedout, sPtr);
    }


    inline bool
    terminate(int64_t timedout = -1,
              BlockingContainer::ContainerStatus *sPtr = NULL) {
        return put(0LL, timedout, sPtr);
    }
};


#define FRAME_UNIT_IN_NANOS	16666666LL


class PeriodicAlarm {


private:


    __rcsId("$Id: PeriodicAlarm.h 86 2012-07-30 05:33:07Z m-hirano $");


private:


    class PeriodicAlarmTable:
        public SynchronizedMap<PeriodicAlarmHandle *, bool> {

    private:
        typedef std::map<PeriodicAlarmHandle *, bool>::iterator
        PAHIterator;

    public:
        inline size_t
        getHandles(PeriodicAlarmHandle *pahPtrs[], size_t maxPtrs) {
            size_t nAdd = 0;
            PeriodicAlarmHandle *pahPtr = NULL;

            lock();
            {
                PAHIterator it;
                PAHIterator endIt = end();

                for (it = begin(); it != endIt; it++) {
                    pahPtr = it->first;
                    if (nAdd < maxPtrs) {
                        pahPtrs[nAdd] = pahPtr;
                        nAdd++;
                    } else {
                        break;
                    }
                }
            }
            unlock();

            return nAdd;
        }
    };


    class PeriodicThread: public Thread {
    private:
        PeriodicAlarm *mPAPtr;

    private:
        inline int
        run(void) {
            int ret = 0;
            NanoTimer nt;
            uint64_t elapsed;
            uint64_t interval = mPAPtr->interval();

            while (mPAPtr->isStopping() == false) {

                nt.start();
                {
                    ScopedLock l(&(mPAPtr->mLock));
                    if (mPAPtr->pNotifyAll() == false) {
                        ret = 1;
                        break;
                    }
                    mPAPtr->pDeleteHandles();
                }
                elapsed = nt.measure();

                if (interval > elapsed) {
                    NanoSecond::nanoSleep(interval - elapsed);
                }
            }
            return ret;
        }

    public:
        inline
        PeriodicThread(PeriodicAlarm *paPtr):
            mPAPtr(paPtr) {
        }
    };
    friend class PeriodicThread;





private:


    PeriodicAlarmTable mPATbl;
    PeriodicAlarmTable mDelPATbl;
    Mutex mLock;
    PeriodicThread *mPTPtr;

    uint64_t mIntervalNSec;
    bool mIsStopping;





private:


    inline void
    pPutHandle(PeriodicAlarmHandle *pahPtr) {
        mPATbl.put(pahPtr, true);
    }


    inline void
    pRemoveHandle(PeriodicAlarmHandle *pahPtr) {
        bool dummy;
        mPATbl.remove(pahPtr, dummy);
        mDelPATbl.put(pahPtr, true);
    }


    inline void
    pDeleteHandles(void) {
        size_t nPAHs = 0;
        PeriodicAlarmHandle *pahPtrs[1024];
        bool dummy;

        nPAHs = mDelPATbl.getHandles(pahPtrs, 1024);
        for (size_t i = 0; i < nPAHs; i++) {
            mDelPATbl.remove(pahPtrs[i], dummy);
            delete pahPtrs[i];
        }
    }


    inline bool
    pNotifyAll(int64_t timeout = -1) {
        size_t nPAHs = 0;
        PeriodicAlarmHandle *pahPtrs[1024];
        int nErrors = 0;

        nPAHs = mPATbl.getHandles(pahPtrs, 1024);
        for (size_t i = 0; i < nPAHs; i++) {
            if (pahPtrs[i]->notify(timeout, NULL) == false) {
                nErrors++;
            }
        }

        return (nErrors == 0) ? true : false;
    }


    inline bool
    pStop(void) {
        size_t i;
        size_t nPAHs = 0;
        size_t nDelPAHs = 0;
        PeriodicAlarmHandle *pahPtrs[1024];
        bool dummy;
        bool ret = false;

        {
            ScopedLock l(&mLock);

            if (mIsStopping == false) {
                mIsStopping = true;

                nPAHs = mPATbl.getHandles(pahPtrs, 1024);
                for (i = 0; i < nPAHs; i++) {
                    (void)pahPtrs[i]->terminate();
                }
                for (i = 0; i < nPAHs; i++) {
                    pahPtrs[i]->stop();
                }
            } else {
                return mIsStopping;
            }
        }

        //
        // Wait all allarms are released. Note that we don't lock the
        // mLock here.
        //
        while (nPAHs > nDelPAHs) {
            nDelPAHs = mDelPATbl.getHandles(pahPtrs, 1024);
            dbgMsg("Alarm released " PFSz(u) "/" PFSz(u) "\n",
                   nDelPAHs, nPAHs);
            NanoSecond::nanoSleep(1000000LL);
        }

        {
            ScopedLock l(&mLock);

            for (i = 0; i < nDelPAHs; i++) {
                mDelPATbl.remove(pahPtrs[i], dummy);
                delete pahPtrs[i];
            }

            mDelPATbl.clear();
            mPATbl.clear();
        }
        ret = mIsStopping;

        if (mPTPtr != NULL) {
            ret = mPTPtr->wait();
        }

        return ret;
    }


public:


    inline
    PeriodicAlarm(uint64_t intervalNSec = FRAME_UNIT_IN_NANOS):
        // mPATbl,
        // mDelPATbl,
        // mLock,
        mPTPtr(NULL),
        mIntervalNSec(intervalNSec),
        mIsStopping(false) {
        mPTPtr = new PeriodicThread(this);
        if (mPTPtr == NULL) {
            fatal("Can't create a periodic thread.\n");
        }
    }
        

    inline
    ~PeriodicAlarm(void) {
        pStop();
        ScopedLock l(&mLock);
        deleteIfNotNULL(mPTPtr);
        mPTPtr = NULL;
    }


private:
    PeriodicAlarm(const PeriodicAlarm &obj);
    PeriodicAlarm operator = (const PeriodicAlarm &obj);


public:


    inline bool
    start() {
        bool ret = false;
        if (mPTPtr != NULL) {
            ret = mPTPtr->start();
        }
        return ret;
    }


    inline bool
    stop(void) {
        return pStop();
    }


    inline bool
    wait(void) {
        bool ret = stop();
        if (mPTPtr != NULL) {
            ret = mPTPtr->wait();
        }
        return ret;
    }


    inline bool
    isStopping(void) {
        ScopedLock l(&mLock);
        return mIsStopping;
    }


    inline uint64_t
    interval(void) {
        return mIntervalNSec;
    }


    inline PeriodicAlarmHandle *
    acquireAlarm(size_t n = BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH) {
        PeriodicAlarmHandle *ret = new PeriodicAlarmHandle(n);

        ScopedLock l(&mLock);
        pPutHandle(ret);

        return ret;
    }


    inline void
    releaseAlarm(PeriodicAlarmHandle *pahPtr) {
        if (pahPtr != NULL) {
            ScopedLock l(&mLock);
            pRemoveHandle(pahPtr);
        }
    }


};


#endif // ! __PERIODICALARM_H__
