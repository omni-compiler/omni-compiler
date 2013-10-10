/* 
 * $Id: Thread.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __THREAD_H__
#define __THREAD_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#ifdef NATA_API_POSIX
#ifdef NATA_OS_LINUX
#include <sys/syscall.h>
#include <sys/prctl.h>
#endif // NATA_OS_LINUX
#endif // NATA_API_POSIX

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>

#include <nata/nata_perror.h>


#define __THREAD_DEBUG__


class Thread;


namespace ThreadStatics {
    extern void *	sThreadEntryPoint(void *ptr);
    extern void		sCancelHandler(void *ptr);
    extern bool		sIsOnHeap(void *addr);
#ifdef NATA_API_POSIX
    extern void		sBlockAllSignals(void);
    extern void		sUnblockAllSignals(void);
    extern sigset_t	sGetFullSignalSet(void);
#ifdef NATA_OS_LINUX
    extern pid_t	sGetLinuxThreadId(void);
    extern void		sSetLinuxThreadName(const char *name);
#endif // NATA_OS_LINUX
#endif // NATA_API_POSIX
    extern bool		sAddThreadName(uint32_t tId, Thread *tPtr);
    extern void		sRemoveThreadName(uint32_t tId);
    extern const char *	sFindThreadName(uint32_t tId);
}



class Thread {
    friend void *	ThreadStatics::sThreadEntryPoint(void *ptr);
    friend void 	ThreadStatics::sCancelHandler(void *ptr);



private:
    __rcsId("$Id: Thread.h 86 2012-07-30 05:33:07Z m-hirano $");



// Public enumerations.


public:
    typedef enum {
        State_Unknown = 0,
        State_Created,
        State_Starting,
        State_Started,
        State_Stopping,
        State_Stopped,
        State_Exiting,
        State_Exited
    } ThreadStatus;



// Protected enumrations.


protected:
    typedef enum {
        Cancel_Asynchronous = 0,
        Cancel_Deferred
    } ThreadCancelType;



// Privates.


private:
    pthread_t mTid;
#if defined(NATA_API_WIN32API)
    DWORD mWinTid;
#endif // NATA_API_WIN32API
    char *mName;
    size_t mStackSize;
    pthread_attr_t mTAttr;

    volatile ThreadStatus mSt;
    Mutex mStatusLock;

    pid_t mCreatorPid;

    int mExitCode;

    volatile bool mIsInStartupCriticalSection;
    Mutex mInStartupCriticalSectionLock;
    WaitCondition mInStartupCriticalSectionWait;

    volatile bool mIsActive;
    Mutex mActiveLock;
    WaitCondition mActiveWait;

    volatile bool mDetachFailure;
    volatile bool mIsDeleting;
    Mutex mDeleteLock;

    volatile bool mIsPthreadCancelCalled;
    Mutex mCancelLock;

    volatile bool mIsCanceled;

    volatile bool mIsWaited;

    volatile bool mDoAutoDelete;



// Private methods.


    inline void
    pSynchronizeStart(void) {
        // Startup sycnhronization.
        mInStartupCriticalSectionLock.lock();
        ReCheck:
        if (mIsInStartupCriticalSection == true) {
            mInStartupCriticalSectionWait.wait(
                &(mInStartupCriticalSectionLock));
            goto ReCheck;
        }
        mInStartupCriticalSectionLock.unlock();
    }


    inline void
    pSetState(ThreadStatus s) {
        mStatusLock.lock();
        mSt = s;
        mStatusLock.unlock();
    }


    inline bool
    pIsMyThread(void) {
#if defined(NATA_API_POSIX)
        return (mTid == pthread_self()) ? true : false;
#elif defined(NATA_API_WIN32API)
        return (mWinTid == GetCurrentThreadId()) ? true : false;
#else
#error Unknown/Non-supported API.
#endif // NATA_API_POSIX, NATA_API_WIN32API
    }


    inline bool
    pIsMyProcess(void) {
        return (mCreatorPid == getpid()) ? true : false;
    }


    inline void
    pExit(int code) {
#ifdef __THREAD_DEBUG__
        dbgMsg("Enter.\n");
#endif // __THREAD_DEBUG__

        bool doPtExit = false;
        bool doDelete = false;
        pSetState(State_Exiting);

        mActiveLock.lock();
	if (mIsActive == true) {
	    mExitCode = code;

            if (mDoAutoDelete == true &&
                ThreadStatics::sIsOnHeap((void *)this) == true) {
                doDelete = true;
            }

	    if (mIsCanceled == false) {
                doPtExit = true;
            }

	    mIsActive = false;
            mActiveWait.wakeAll();

            pSetState(State_Exited);

#ifdef __THREAD_DEBUG__
            dbgMsg("exiting, %s.\n",
                   mIsCanceled == true ? "by cancel" : "normaly");
#endif // __THREAD_DEBUG__
        }
        mActiveLock.unlock();

        ThreadStatics::sRemoveThreadName(threadId());

        if (doDelete == true) {
#ifdef __THREAD_DEBUG__
            dbgMsg("auto deletion.\n");
#endif // __THREAD_DEBUG__
            delete this;
        }

#ifdef __THREAD_DEBUG__
        dbgMsg("Leave.\n");
#endif // __THREAD_DEBUG__

        if (doPtExit == true) {
            pthread_exit(NULL);
        }
    }


    inline bool
    pCancel(void) {
#ifdef __THREAD_DEBUG__
        dbgMsg("Enter.\n");
#endif // __THREAD_DEBUG__

        pSynchronizeStart();

        mCancelLock.lock();
        if (mIsPthreadCancelCalled == false) {
            int st;
            errno = 0;
            if ((st = pthread_cancel(mTid)) != 0) {
                int sErrno = errno;
                errno = st;
                perror("pthread_cancel");
                errno = sErrno;
            }
            mIsPthreadCancelCalled = true;
            pSetState(State_Stopping);
        }
        mCancelLock.unlock();

#ifdef __THREAD_DEBUG__
        dbgMsg("Leave.\n");
#endif // __THREAD_DEBUG__
        return mIsPthreadCancelCalled;
    }



// Protected.


protected:


// Protected methods.


    inline void
    setCancellationType(ThreadCancelType t) {
        switch (t) {
            case Cancel_Asynchronous: {
                (void)pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
                break;
            }
            case Cancel_Deferred: {
                (void)pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
                break;
            }
            default: {
                break;
            }
        }
    }


    inline void
    checkCancellation(void) {
	pthread_testcancel();
    }



// Protected virtual methods.


    virtual int
    run(void) {
	return 0;
    }



// Publics.


public:


// Public class methods.


    static inline void
    enableCancellation(void) {
        (void)pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    }


    static inline void
    disableCancellation(void) {
        (void)pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
    }


    static inline void
    yield(void) {
        (void)sched_yield();
    }


    static inline const char *
    findThreadName(uint32_t tId) {
        return ThreadStatics::sFindThreadName(tId);
    }


#ifdef NATA_API_POSIX
#ifdef NATA_OS_LINUX
    static inline pid_t
    getLinuxThreadId(void) {
        return ThreadStatics::sGetLinuxThreadId();
    }
#endif // NATA_OS_LINUX
#endif // NATA_API_POSIX


// Public methods.


    inline const char *
    getName(void) {
        return (const char *)mName;
    }


    inline void
    setName(const char *name) {
        (void)free((void *)mName);
        mName = strdup(name);
    }


    inline bool
    setStackSize(size_t s) {
        bool ret = false;
        mActiveLock.lock();
        if (mIsActive != true) {
#if PTHREAD_STACK_MIN > 0
            if (s < PTHREAD_STACK_MIN) {
                s = PTHREAD_STACK_MIN;
            }
#endif // PTHREAD_STACK_MIN > 0
            mStackSize = s;

            int sErrno;
            int st;
            errno = 0;
            if ((st = pthread_attr_setstacksize(&mTAttr, mStackSize)) != 0) {
                sErrno = errno;
                errno = st;
                perror("pthread_attr_setstacksize");
                errno = sErrno;
                goto Done;
            } else {
                ret = true;
            }
        }
        Done:
        mActiveLock.unlock();
        return ret;
    }


    inline bool
    start(bool waitIfRunning = false, bool doAutoDelete = false) {
	bool ret = false;

        mInStartupCriticalSectionLock.lock();
        mIsInStartupCriticalSection = true;

	mActiveLock.lock();

	ReCheck:
	if (mIsActive == true) {
	    if (waitIfRunning == false) {
		goto Done;
	    } else {
		mActiveWait.wait(&mActiveLock);
		if (mIsDeleting == true) {
		    goto Done;
		}
		goto ReCheck;
	    }
	}

	mExitCode = -1;
        mDoAutoDelete = doAutoDelete;
	mDetachFailure = false;
	mIsDeleting = false;
        mIsPthreadCancelCalled = false;
	mIsCanceled = false;
        mIsWaited = false;
        mTid = INVALID_THREAD_ID;
#ifdef NATA_API_WIN32API
        mWinTid = 0;
#endif // NATA_API_WIN32API

	int st;
	int sErrno;

	errno = 0;
	if ((st = pthread_create(&mTid,
                                 (mStackSize > 0) ? &mTAttr : NULL,
                                 ThreadStatics::sThreadEntryPoint,
                                 this)) != 0) {
	    sErrno = errno;
	    errno = st;
	    perror("pthraed_create");
	    errno = sErrno;
	    goto Done;
	}

        errno = 0;
        if ((st = pthread_detach(mTid)) != 0) {
            sErrno = errno;
            errno = st;
            perror("pthread_detach");
            errno = sErrno;
            mDetachFailure = true;
            goto Done;
	}

        // Both the thread creation and detaching are succeeded.
	mIsActive = true;
        pSetState(State_Starting);
	ret = true;

	Done:
	mActiveLock.unlock();

        mIsInStartupCriticalSection = false;
        mInStartupCriticalSectionWait.wakeAll();
        mInStartupCriticalSectionLock.unlock();

	return ret;
    }


    inline bool
    wait(void) {
        if (mIsWaited == false) {

            // mIsActive is modified in pExit().
            mActiveLock.lock();
            ReCheck:
            if (mIsActive == true) {
                mActiveWait.wait(&mActiveLock);
                goto ReCheck;
            }
            mActiveLock.unlock();

            mIsWaited = true;
        }

	return mIsWaited;
    }


    inline bool
    cancel(void) {
        return pCancel();
    }


    inline bool
    stop(bool doWait = true) {
	bool ret = false;

        mActiveLock.lock();
        if (mIsActive == true) {
            ret = cancel();
        } else {
            ret = true;
        }
        mActiveLock.unlock();

        if (ret == true) {
            if (doWait == true) {
                return wait();
            } else {
                return ret;
            }
        } else {
            return ret;
        }
    }


    int
    exitCode(void) {
	return mExitCode;
    }


    uint32_t
    threadId(void) {
#if defined(NATA_API_POSIX)
	return (uint32_t)mTid;
#elif defined(NATA_API_WIN32API)
        return (uint32_t)mWinTid;
#else
#error Unknown/Non-supported API.
#endif // NATA_API_POSIX, NATA_API_WIN32API
    }


    ThreadStatus
    status(void) {
        ThreadStatus ret = State_Unknown;
        mStatusLock.lock();
        ret = mSt;
        mStatusLock.unlock();

        return ret;
    }


#ifdef NATA_API_POSIX
    static inline void
    blockAllSignals(void) {
        ThreadStatics::sBlockAllSignals();
    }


    static inline void
    unblockAllSignals(void) {
        ThreadStatics::sUnblockAllSignals();
    }


    static inline sigset_t
    getFullSignalSet(void) {
        return ThreadStatics::sGetFullSignalSet();
    }
#endif // NATA_API_POSIX



// Constructors and destructor.


    Thread(void) :
        mTid(INVALID_THREAD_ID),
#ifdef NATA_API_WIN32API
        mWinTid(0),
#endif // NATA_API_WIN32API
        mName(NULL),
        mStackSize(0),
        // mTAttr,

        mSt(State_Unknown),
        // mStatusLock,

        mCreatorPid(getpid()),
        mExitCode(-1),

        mIsInStartupCriticalSection(false),
        // mInStartupCriticalSectionLock,
        // mInStartupCriticalSectionWait,

        mIsActive(false),
        // mActiveLock,
        // mActiveWait,

        mDetachFailure(false),
        mIsDeleting(false),
        // mDeleteLock,

        mIsPthreadCancelCalled(false),
        // mCancelLock,

        mIsCanceled(false),

        mIsWaited(false),

        mDoAutoDelete(false) {
        (void)pthread_attr_init(&mTAttr);
        pSetState(State_Created);
    }


    virtual
    ~Thread(void) {
#ifdef __THREAD_DEBUG__
        dbgMsg("Enter.\n");
#endif // __THREAD_DEBUG__
        pSynchronizeStart();

	mDeleteLock.lock();
	if (mIsDeleting != true) {
	    mIsDeleting = true;

	    if (pIsMyProcess() == true &&
                pIsMyThread() == false) {
		stop();
            }

            mActiveLock.lock();
            mIsActive = false;
            mActiveWait.wakeAll();
            mActiveLock.unlock();
            (void)pthread_attr_destroy(&mTAttr);
            (void)free((void *)mName);
	}
	mDeleteLock.unlock();

#ifdef __THREAD_DEBUG__
        dbgMsg("Leave.\n");
#endif // __THREAD_DEBUG__
    }


private:
    Thread(const Thread &obj);
    Thread operator = (const Thread &obj);
};


#endif // __THREAD_H__
