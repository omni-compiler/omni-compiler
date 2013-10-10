/* 
 * $Id: SignalThread.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __SIGNALTHREAD_H__
#define __SIGNALTHREAD_H__

#include <nata/nata_rcsid.h>
#include <nata/nata_includes.h>


#ifdef NATA_API_POSIX


#include <nata/nata_macros.h>
#include <nata/Thread.h>
#include <nata/SynchronizedMap.h>
#include <nata/ProcessJanitor.h>
#include <nata/ScopedLock.h>
#include <nata/nata_perror.h>



class SignalThread: public Thread {



private:
    __rcsId("$Id: SignalThread.h 86 2012-07-30 05:33:07Z m-hirano $");





    Mutex mLock;
    SynchronizedMap<int, sighandler_t> mTbl;
    sigset_t mSS;


    SignalThread(const SignalThread &obj);
    SignalThread operator = (const SignalThread &obj);





    inline sighandler_t
    mDeleteHandler(int sig) {
        sighandler_t oH = NULL;        
        if (sig != SIGCHLD) {
            (void)mTbl.remove(sig, oH);
            (void)sigdelset(&mSS, sig);
        }
        return oH;
    }


    inline sighandler_t
    mAddHandler(int sig, sighandler_t proc) {
        sighandler_t oH = NULL;

        if (sig != SIGCHLD) {
            if (proc != NULL) {
                (void)mTbl.get(sig, oH);
                (void)mTbl.put(sig, proc);
                (void)sigaddset(&mSS, sig);
            } else {
                oH = mDeleteHandler(sig);
            }
        }

        return oH;
    }


    inline sighandler_t
    mFindHandler(int sig) {
        sighandler_t oH = NULL;
        (void)mTbl.get(sig, oH);
        return oH;
    }





public:


    inline sighandler_t
    setHandler(int sig, sighandler_t proc) {
        ScopedLock l(&mLock);
        return mAddHandler(sig, proc);
    }


    inline sighandler_t
    unsetHandler(int sig) {
        ScopedLock l(&mLock);
        return mDeleteHandler(sig);
    }


    inline sighandler_t
    findHandler(int sig) {
        ScopedLock l(&mLock);
        return mFindHandler(sig);
    }


    inline void
    ignore(int sig) {
        ScopedLock l(&mLock);

        //
        // Add the signal to the set. Thus the thread can wait the
        // signal and just sink it.
        //
        (void)sigaddset(&mSS, sig);
    }


    inline void
    reset(int sig) {
        ScopedLock l(&mLock);

        //
        // Delete the signal from the set. The thread unbloked the
        // signal and can't wait the signal, thus the signal is caught
        // by default handler.
        //
        if (sig != SIGCHLD) {
            (void)sigdelset(&mSS, sig);
        }
    }





    SignalThread(void) :
        Thread() {
        Process::blockAllSignals();
	(void)sigemptyset(&mSS);
        (void)sigaddset(&mSS, SIGCHLD);
        setName("signal thread");
    }





    int
    run(void) {
	int sig;
        sighandler_t sProc;

        Thread::unblockAllSignals();
        (void)sigaddset(&mSS, SIGCHLD);	// for failsafe.
        (void)pthread_sigmask(SIG_SETMASK, &mSS, NULL);

	while (1) {

	    errno = 0;
	    if (sigwait(&mSS, &sig) == -1) {
		if (errno != EAGAIN) {
		    perror("sigwait");
		}
		continue;
	    }

            dbgMsg("Got signal %d\n", sig);

            if (sig == SIGCHLD) {
                ProcessJanitor::requestWait(PROCESS_WAIT_ANY);
            } else {
                sProc = NULL;
                if (mTbl.get(sig, sProc) == true &&
                    sProc != NULL) {
                    sProc(sig);
                }
            }

        }

	return 0;
    }

};


#endif // NATA_API_POSIX


#endif // ! __SIGNALTHREAD_H__
