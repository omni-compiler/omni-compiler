#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>
#include <nata/SynchronizedMap.h>
#include <nata/Thread.h>

#include <nata/nata_perror.h>





#ifdef NATA_API_POSIX
extern int end;
#endif /* NATA_API_POSIX */





namespace ThreadStatics {


    __rcsId("$Id: ThreadStatics.cpp 86 2012-07-30 05:33:07Z m-hirano $")





    typedef SynchronizedMap<uint32_t, char *> sThreadNameTable;





    static sThreadNameTable sThdTbl;





#ifdef NATA_API_POSIX


    static bool sIsInitialized = false;
    static sigset_t sFullSignalSet;


    static inline void
    sInitializeSignalMask(void) {
        (void)sigfillset(&sFullSignalSet);        
#if defined(SIGRTMIN) && defined(SIGRTMAX)
        //
        // Exclude realtime signals (also for Linux's thread
        // cancellation).
        //
        {
            int i;
            int sMin = SIGRTMIN;
            int sMax = SIGRTMAX;
            for (i = sMin; i < sMax; i++) {
                (void)sigdelset(&sFullSignalSet, i);
            }
        }
#endif // SIGRTMIN && SIGRTMAX
        //
        // And exclude everything that seems for thread-related.
        //
#ifdef SIGWAITING
        (void)sigdelset(&sFullSignalSet, SIGWAITING);
#endif // SIGWAITING
#ifdef SIGLWP
        (void)sigdelset(&sFullSignalSet, SIGLWP);
#endif // SIGLWP
#ifdef SIGFREEZE
        (void)sigdelset(&sFullSignalSet, SIGFREEZE);
#endif // SIGFREEZE
#ifdef SIGCANCEL
        (void)sigdelset(&sFullSignalSet, SIGCANCEL);
#endif // SIGCANCEL
    }


    static inline void
    sChildAfterFork(void) {
        sThdTbl.childAfterFork();
    }


    class ThreadInitializer {
    public:
        ThreadInitializer(void) {
            if (sIsInitialized == false) {
                sInitializeSignalMask();
                (void)pthread_atfork(NULL,
                                     NULL,
                                     sChildAfterFork);                
                sIsInitialized = true;
            }
        }

    private:
        ThreadInitializer(const ThreadInitializer &obj);
        ThreadInitializer operator = (const ThreadInitializer &obj);
    };


    static ThreadInitializer sTi;


    void
    sBlockAllSignals(void) {
        (void)pthread_sigmask(SIG_SETMASK, &sFullSignalSet, NULL);
    }


    void
    sUnblockAllSignals(void) {
        sigset_t empty;
        (void)sigemptyset(&empty);
        (void)pthread_sigmask(SIG_SETMASK, &empty, NULL);
    }


    sigset_t
    sGetFullSignalSet(void) {
        return sFullSignalSet;
    }


#ifdef NATA_OS_LINUX
    pid_t
    sGetLinuxThreadId(void) {
        return (pid_t)syscall(SYS_gettid);
    }


    void
    sSetLinuxThreadName(const char *name) {
        if (isValidString(name) == true) {
            char buf[16];
            (void)snprintf(buf, sizeof(buf), "%s", name);
            (void)prctl(PR_SET_NAME, (unsigned long)buf,
                        ULONG_MAX, ULONG_MAX, ULONG_MAX);
        }
    }
#endif // NATA_OS_LINUX


#endif // NATA_API_POSIX





    bool
    sAddThreadName(uint32_t tId, Thread *tPtr) {
        const char *name = tPtr->getName();
        if (isValidString(name) == true) {
            return sThdTbl.put(tId, strdup(name));
        }
        return true;
    }


    void
    sRemoveThreadName(uint32_t tId) {
        char *dum = NULL;
        (void)sThdTbl.remove(tId, dum);
        (void)free((void *)dum);
    }


    const char *
    sFindThreadName(uint32_t tId) {
        char *ret = NULL;
        (void)sThdTbl.get(tId, ret);
        return ret;
    }


    bool
    sIsOnHeap(void *addr) {
#if defined(NATA_API_POSIX)
        return ((((uintptr_t)&end) <= ((uintptr_t)addr)) &&
                (((uintptr_t)addr) < ((uintptr_t)sbrk((intptr_t)0)))) ?
            true : false;
#elif defined(NATA_API_WIN32API)
        /*
         * Not yet.
         */
        (void)addr;
        return false;
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    }


    void
    sCancelHandler(void *ptr) {
        Thread *tPtr = (Thread *)ptr;

#ifdef __THREAD_DEBUG__
        dbgMsg("Enter.\n");
#endif // __THREAD_DEBUG__
        if (tPtr != NULL) {
            tPtr->mCancelLock.lock();
            tPtr->mIsCanceled = true;
            tPtr->pSetState(Thread::State_Stopped);
            tPtr->mCancelLock.unlock();
            tPtr->pExit(-1);
        }
#ifdef __THREAD_DEBUG__
        dbgMsg("Leave.\n");
#endif // __THREAD_DEBUG__
    }


    void *
    sThreadEntryPoint(void *ptr) {
        Thread *tPtr = (Thread *)ptr;

#ifdef NATA_API_POSIX
        sBlockAllSignals();
#endif // NATA_API_POSIX

        if (tPtr != NULL) {
            uint32_t tId = tPtr->threadId();
            sAddThreadName(tId, tPtr);

#ifdef NATA_API_POSIX
#ifdef NATA_OS_LINUX
            sSetLinuxThreadName(tPtr->getName());
#endif // NATA_OS_LINUX
#endif // NATA_API_POSIX

#ifdef __THREAD_DEBUG__
            dbgMsg("Enter.\n");
#endif // __THREAD_DEBUG__

            tPtr->pSynchronizeStart();

            int ret = -1;
            volatile bool cleanFinish = false;

            tPtr->disableCancellation();
            {
                if (tPtr->mDetachFailure == true) {
                    tPtr->enableCancellation();
                    goto Done;
                }

                tPtr->pSetState(Thread::State_Started);
#ifdef NATA_API_WIN32API
                tPtr->mWinTid = GetCurrentThreadId();
#endif /* NATA_API_WIN32API */
                
            }
            tPtr->enableCancellation();

            pthread_cleanup_push(sCancelHandler, ptr);
            {
                ret = tPtr->run();
                cleanFinish = true;
            }
            pthread_cleanup_pop((cleanFinish == true) ? 0 : 1);

            Done:
            tPtr->pExit(ret);
        }

#ifdef __THREAD_DEBUG__
        dbgMsg("Leave.\n");
#endif // __THREAD_DEBUG__
        return NULL;
    }


}
