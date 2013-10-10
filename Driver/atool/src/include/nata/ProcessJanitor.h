/* 
 * $Id: ProcessJanitor.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __PROCESSJANITOR_H__
#define __PROCESSJANITOR_H__

#include <nata/nata_rcsid.h>
#include <nata/nata_includes.h>
#include <nata/nata_macros.h>


#ifdef NATA_API_POSIX


#include <nata/Thread.h>
#include <nata/BoundedBlockingQueue.h>
#include <nata/SynchronizedMap.h>
#include <nata/Completion.h>
#include <nata/ScopedLock.h>

#include <nata/nata_perror.h>


#define PROCESS_WAIT_ANY	0



class ProcessJanitor: public Thread {


private:
    __rcsId("$Id: ProcessJanitor.h 86 2012-07-30 05:33:07Z m-hirano $");





    class mProcessCompletion: public Completion {


    private:


        ProcessJanitor *mPJPtr;
        int mExitStatus;
        pid_t mPid;
        bool mIsDone;


        static void
        done(void *ptr) {
            mProcessCompletion *pcPtr = (mProcessCompletion *)ptr;
            if (pcPtr != NULL) {
                ProcessJanitor *pjPtr = pcPtr->mPJPtr;
                if (pjPtr != NULL) {
                    if (pjPtr->mIsDeleting == false) {
                        int sTmp;
                        if (pjPtr->findExitStatus(pcPtr->mPid, sTmp) == true) {
                            pcPtr->mExitStatus = sTmp;
                        } else {
                            pcPtr->mExitStatus += 1;
                        }
                        pcPtr->mIsDone = true;
                        return;
                    }
                }
            }
            fatal("Invalid parameter(s).\n");
        }
        friend void done(void *ptr);


        static bool
        isComplete(void *ptr) {
            mProcessCompletion *pcPtr = (mProcessCompletion *)ptr;
            if (pcPtr != NULL) {
                return pcPtr->mIsDone;
            }
            fatal("Invalid parameter.\n");
        }
        friend bool isComplete(void *ptr);


    public:


        mProcessCompletion(pid_t pid, Mutex *mtxPtr, ProcessJanitor *pjPtr) :
            Completion(mtxPtr),
            mPJPtr(pjPtr),
            mExitStatus(INT_MIN),
            mPid(pid),
            mIsDone(false) {
            setCheckProc(mProcessCompletion::isComplete);
            setWakeProc(mProcessCompletion::done);
            setContext((void *)this);
        }


        inline int
        getExitStatus(void) {
            return mExitStatus;
        }


    };





// command queue, for waitpid(2).
    typedef BoundedBlockingQueue<int> IntQ;


// pid_t -> int, returned status of a process waited by waitpid(2).
    typedef SynchronizedMap<pid_t, int>	ExitStatusTable;


// pid_t -> Completion, database for status report request.
    class ReportRequestTable:
    public SynchronizedMap<pid_t, mProcessCompletion *> {


    private:


        ReportRequestTable(const ReportRequestTable &obj);
        ReportRequestTable operator = (const ReportRequestTable &obj);


        static void
        deleteHook(mProcessCompletion *clPtr, void *arg) {
            (void)arg;
            delete clPtr;
        }


    public:


        ReportRequestTable(void) :
            SynchronizedMap<pid_t, mProcessCompletion *>() {
            setDeleteHook(deleteHook, NULL);
        }


    };
    friend class mProcessCompletion;



private:


    ExitStatusTable *mStTblPtr;
    ReportRequestTable *mRrTblPtr;
    IntQ *mQPtr;

    bool mIsDeleting;


    bool	addExitStatus(pid_t pid, int state);
    void	deleteExitStatus(pid_t pid);
    bool	findExitStatus(pid_t, int &state);

    bool	addReportRequest(pid_t pid, Mutex *mtxPtr);
    void	deleteReportRequest(pid_t pid);
    bool	findReportRequest(pid_t pid, mProcessCompletion * &clPtr);

    void	waitChildren(void);


    ProcessJanitor(const ProcessJanitor &obj);
    ProcessJanitor operator = (const ProcessJanitor &obj);


protected:
    int		run(void);


public:
    ProcessJanitor(void);
    ~ProcessJanitor(void);

    static bool		requestWait(int cmd);
    static bool		waitExit(pid_t pid, int &state);
    static bool		requestReport(pid_t pid, Mutex *mtxPtr);

    static void		initialize(void);
    static void		finalize(void);
};


#endif // NATA_API_POSIX


#endif // ! __PROCESSJANITOR_H__
