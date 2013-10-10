/* 
 * $Id: Completion.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __COMPLETION_H__
#define __COMPLETION_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/WaitCondition.h>
#include <nata/ScopedLock.h>

#include <nata/nata_perror.h>





class Completion {


private:
    __rcsId("$Id: Completion.h 86 2012-07-30 05:33:07Z m-hirano $");


    typedef bool	(*CheckProcT)(void *ptr);
    typedef void	(*WakeProcT)(void *ptr);

    WaitCondition mCond;
    Mutex *mLockPtr;
    void *mCtx;
    CheckProcT	mCPPtr;
    WakeProcT mWPPtr;
    bool mDummy;
    bool mIsDeleting;


    inline bool
    pIsComplete(void) {
        bool ret = false;

        if (mCPPtr != NULL) {
            ret = mCPPtr(mCtx);
        } else {
            ret = mDummy;
            if (ret == true) {
                mDummy = false;
            }
        }
        return ret;
    }


    inline void
    pDone(void) {
        if (mWPPtr != NULL) {
            mWPPtr(mCtx);
        } else {
            mDummy = true;
        }
    }



public:


    Completion(Mutex *lockPtr) :
        // mCond,
        mLockPtr(lockPtr),
        mCtx(NULL),
        mCPPtr(NULL),
        mWPPtr(NULL),
        mDummy(false),
        mIsDeleting(false) {
    }


    virtual
    ~Completion(void) {
        mIsDeleting = true;
        mCond.wakeAll();
    }


private:
    Completion(const Completion &obj);
    Completion operator = (const Completion &obj);





public:


    inline bool
    wait(bool doLock = false) {
        bool ret = false;

        if (doLock == true) {
            mLockPtr->lock();
        }

        ReCheck:
        if (mIsDeleting != true) {
            if ((ret = pIsComplete()) == false) {
                ret = mCond.wait(mLockPtr);
                goto ReCheck;
            }
        }

        if (doLock == true) {
            mLockPtr->unlock();
        }

        return ret;
    }


    inline bool
    timedwait(uint64_t uSec, bool doLock = false) {
        bool ret = false;

        if (doLock == true) {
            mLockPtr->lock();
        }

        if (mIsDeleting != true) {
            if ((ret = pIsComplete()) == false) {
                ret = mCond.timedwait(mLockPtr, uSec);
            }
        }

        if (doLock == true) {
            mLockPtr->unlock();
        }

        return ret;
    }


    inline void
    wake(void) {
        ScopedLock l(mLockPtr);
        pDone();
        mCond.wake();
    }


    inline void
    wakeAll(void) {
        ScopedLock l(mLockPtr);
        pDone();
        mCond.wakeAll();
    }


    inline void
    setContext(void *ctx) {
        mCtx = ctx;
    }


    inline void *
    getContext(void) {
        return mCtx;
    }


    inline Mutex *
    getMutex(void) {
        return mLockPtr;
    }


    void
    setCheckProc(CheckProcT cp) {
        mCPPtr = cp;
    }


    void
    setWakeProc(WakeProcT wp) {
        mWPPtr = wp;
    }
};


#endif // ! __COMPLETION_H__
