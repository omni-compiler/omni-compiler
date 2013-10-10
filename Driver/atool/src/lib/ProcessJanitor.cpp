#include <nata/nata_rcsid.h>
__rcsId("$Id: ProcessJanitor.cpp 86 2012-07-30 05:33:07Z m-hirano $")

#include <nata/libnata.h>


#ifdef NATA_API_POSIX


#include <nata/ProcessJanitor.h>



static pthread_mutex_t ctorLock = PTHREAD_MUTEX_INITIALIZER;
volatile static bool aProcJntrIsAlive = false;

static ProcessJanitor *pjPtr = NULL;

#define abortIfNULL(...) { \
    if (pjPtr == NULL) { \
	fatal("Call ProcessJanitor::initialize() BEFORE call me.\n"); \
    } \
}



// Constructor
ProcessJanitor::ProcessJanitor(void) :
    mStTblPtr(new ExitStatusTable()),
    mRrTblPtr(new ReportRequestTable()),
    mQPtr(new IntQ()),
    mIsDeleting(false) {

    (void)pthread_mutex_lock(&ctorLock);

    if (aProcJntrIsAlive == true) {
	(void)pthread_mutex_unlock(&ctorLock);
	fatal("The ProcessJanitor must be a singleton.\n");
    }
    aProcJntrIsAlive = true;

    setName("process janitor");

    (void)pthread_mutex_unlock(&ctorLock);
}


// Destructor
ProcessJanitor::~ProcessJanitor(void) {
    mStTblPtr->clear();
    delete mStTblPtr;

    
    mRrTblPtr->clear();
    delete mRrTblPtr;

    mQPtr->clear();
    delete mQPtr;
}



bool
ProcessJanitor::addExitStatus(pid_t pid, int state) {
    bool ret = mStTblPtr->put(pid, state);
    return ret;
}


void
ProcessJanitor::deleteExitStatus(pid_t pid) {
    int state;
    (void)mStTblPtr->remove(pid, state);
}


bool
ProcessJanitor::findExitStatus(pid_t pid, int &state) {
    return mStTblPtr->get(pid, state);
}


bool
ProcessJanitor::addReportRequest(pid_t pid, Mutex *mtxPtr) {
    bool ret = false;
    mProcessCompletion *clPtr = NULL;

    if (mRrTblPtr->get(pid, clPtr) != true) {
        clPtr = new mProcessCompletion(pid, mtxPtr, pjPtr);
        if (clPtr != NULL &&
            mRrTblPtr->put(pid, clPtr) == true) {
            ret = true;
        }
    } else {
        ret = true;
    }

    return ret;
}


void
ProcessJanitor::deleteReportRequest(pid_t pid) {
    mProcessCompletion *clPtr = NULL;
    (void)mRrTblPtr->remove(pid, clPtr);
    deleteIfNotNULL(clPtr);
}


bool
ProcessJanitor::findReportRequest(pid_t pid, mProcessCompletion *&clPtr) {
    return mRrTblPtr->get(pid, clPtr);
}


void
ProcessJanitor::waitChildren(void) {
    int stat;
    pid_t pid;
    mProcessCompletion *clPtr;

    while (true) {
	errno = 0;
	pid = waitpid(-1, &stat, WNOHANG);
	if (pid <= 0) {
	    if (pid < 0) {
		if (errno != ECHILD) {
		    perror("waitpid");
		}
	    }
            break;
	} else {
            if (WIFEXITED(stat) || WIFSIGNALED(stat)) {
                clPtr = NULL;
                if (findReportRequest(pid, clPtr) == true) {
                    addExitStatus(pid, stat);
                    if (clPtr != NULL) {
                        clPtr->wake();
                        //
                        // In this case, deletion of the status and
                        // the request are done in waitExit()
                        //
                    } else {
                        deleteExitStatus(pid);
                        deleteReportRequest(pid);
                    }
                }
	    }
	}
    }
}



bool
ProcessJanitor::requestWait(int cmd) {
    abortIfNULL();
    return pjPtr->mQPtr->put(cmd);
}


bool
ProcessJanitor::waitExit(pid_t pid, int &state) {
    dbgMsg("Enter.\n");

    abortIfNULL();

    mProcessCompletion *clPtr = NULL;
    bool ret = false;

    if (pjPtr->mIsDeleting == true) {
        goto Done;
    }

    if (pjPtr->findReportRequest(pid, clPtr) == false) {
        goto Done;
    }

    if (clPtr != NULL) {
        ret = clPtr->wait();
        if (ret == true) {
            state = clPtr->getExitStatus();
        }
    }

    Done:
    pjPtr->deleteExitStatus(pid);
    pjPtr->deleteReportRequest(pid);

    dbgMsg("Leave.\n");
    return ret;
}


bool
ProcessJanitor::requestReport(pid_t pid, Mutex *mtxPtr) {
    abortIfNULL();
    return pjPtr->addReportRequest(pid, mtxPtr);
}



int
ProcessJanitor::run(void) {
    int dummy;
    while (mIsDeleting == false) {
	if (mQPtr->get(dummy, 5000000) == true) {
            dbgMsg("Wait requested.\n");
	    waitChildren();
	} else {
	    // Timedout. Wait any children anyway.
            if (mIsDeleting == false) {
                // dbgMsg("Wait any children.\n");
                waitChildren();
            }
	}
    }

    return 0;
}


void
ProcessJanitor::initialize(void) {
    if (pjPtr == NULL) {
	pjPtr = new ProcessJanitor();
	pjPtr->start(false);
    }
}


void
ProcessJanitor::finalize(void) {
    deleteIfNotNULL(pjPtr);
}


#endif // NATA_API_POSIX
