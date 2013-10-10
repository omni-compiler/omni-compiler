/* 
 * $Id: WaitCondition.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __WAITCONDITION_H__
#define __WAITCONDITION_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/NanoSecond.h>

#include <nata/Mutex.h>

#include <nata/nata_perror.h>





class WaitCondition {



private:
    __rcsId("$Id: WaitCondition.h 86 2012-07-30 05:33:07Z m-hirano $");

    pthread_cond_t mCond;
    pid_t mCreatorPid;
    bool mIsDeleting;





    inline void
    mInitialize(void) {
	int st;
	errno = 0;
	if ((st = pthread_cond_init(&mCond, NULL)) != 0) {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_cond_init");
	    errno = sErrno;
	    fatal("Can't create a conditional variable.\n");
	}
    }
        




public:
    WaitCondition(void) :
	mCreatorPid(getpid()),
	mIsDeleting(false) {
        mInitialize();
    }
    ~WaitCondition(void) {
	mIsDeleting = true;
        (void)pthread_cond_broadcast(&mCond);
	// if the caller was forked, not destroy.
	if (getpid() == mCreatorPid) {
	    (void)pthread_cond_destroy(&mCond);
	}
    }


private:
    WaitCondition(const WaitCondition &obj);
    WaitCondition operator = (const WaitCondition &obj);



public:
    inline void
    reinitialize(void) {
        pid_t pid = getpid();
        if (pid != mCreatorPid) {
            mInitialize();
            mIsDeleting = false;
            mCreatorPid = pid;
        }
    }

    inline bool
    wait(Mutex *mPtr) {
	int st;
	errno = 0;
	bool ret = ((st = pthread_cond_wait(&mCond, &(mPtr->mLock))) == 0) ?
	    true : false;
	if (ret != true) {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_cond_wait");
	    errno = sErrno;
	}
	return ret;
    }

    inline bool
    timedwait(Mutex *mPtr, uint64_t uSec) {
	int st;
	struct timespec to;

        NanoSecond ns = CURRENT_TIME_IN_NANOS;
        ns = ns + (uSec * 1000LL);
        to = (struct timespec)ns;

	errno = 0;
	bool ret = ((st = pthread_cond_timedwait(&mCond,
						 &(mPtr->mLock),
						 &to)) == 0) ? true : false;
	if (ret != true) {
	    if (errno != ETIMEDOUT && st != ETIMEDOUT) {
		int sErrno = errno;
		errno = st;
		perror("pthread_cond_timedwait");
		errno = sErrno;
	    }
	}
	return ret;
    }

    inline bool
    wake(void) {
	int st;
	errno = 0;
	bool ret = ((st = pthread_cond_signal(&mCond)) == 0) ? true : false;
	if (ret != true) {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_cond_signal");
	    errno = sErrno;
	}
	return ret;
    }

    inline bool
    wakeAll(void) {
	int st;
	errno = 0;
	bool ret = ((st = pthread_cond_broadcast(&mCond)) == 0) ? true : false;
	if (ret != true) {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_cond_broadcast");
	    errno = sErrno;
	}
	return ret;
    }

    inline bool
    isDeleting(void) {
	return mIsDeleting;
    }
};


#endif // ! __WAITCONDITION_H__
