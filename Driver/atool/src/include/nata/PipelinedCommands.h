/* 
 * $Id: PipelinedCommands.h 196 2012-09-05 05:01:12Z m-hirano $
 */
#ifndef __PIPELINEDCOMMANDS_H__
#define __PIPELINEDCOMMANDS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/IOPipers.h>
#include <nata/FullSimplexedCommand.h>

#include <vector>
#include <string>

#include <nata/nata_perror.h>





class PipelinedCommands {





private:


    __rcsId("$Id: PipelinedCommands.h 196 2012-09-05 05:01:12Z m-hirano $");





private:


    class mCommandParams {
    private:
        const char *mPath;
        char **mArgv;
        size_t mNArgv;
        const char *mPwd;
        const char *mTeeInFile;
        const char *mTeeOutFile;
        const char *mTeeErrFile;

        mCommandParams(const mCommandParams &obj);
        mCommandParams operator = (const mCommandParams &obj);


    public:
        mCommandParams(const char *pwd, const char *path, char * const *argv,
                       const char *teeInFile = NULL,
                       const char *teeOutFile = NULL,
                       const char *teeErrFile = NULL) :
            mPath(NULL),
            mArgv(NULL),
            mNArgv(-INT_MAX),
            mPwd(NULL),
            mTeeInFile(NULL),
            mTeeOutFile(NULL),
            mTeeErrFile(NULL) {
            mPath = strdup(path);
            if (isValidString(pwd) == true) {
                mPwd = strdup(pwd);
            } else {
                mPwd = NULL;
            }

            size_t n = 0;
            char * const *cpArgv = argv;
            while (*(cpArgv++) != NULL) {
                n++;
            }
            mArgv = (char **)malloc(sizeof(char *) * (n + 1));
            if (mArgv == NULL) {
                fatal("Can't allocate an array for command line options.");
            }

            size_t i;
            for (i = 0; i < n; i++) {
                mArgv[i] = strdup(argv[i]);
            }
            mArgv[n] = NULL;
            mNArgv = n;

            if (isValidString(teeInFile) == true) {
                mTeeInFile = strdup(teeInFile);
            }
            if (isValidString(teeOutFile) == true) {
                mTeeOutFile = strdup(teeOutFile);
            }
            if (isValidString(teeErrFile) == true) {
                mTeeErrFile = strdup(teeErrFile);
            }
        }


        ~mCommandParams(void) {
            (void)free((void *)mPath);
            (void)free((void *)mPwd);
            size_t i;

            for (i = 0; i < mNArgv; i++) {
                (void)free((void *)mArgv[i]);
            }
            (void)free(mArgv);

            free((void *)mTeeInFile);
            free((void *)mTeeOutFile);
            free((void *)mTeeErrFile);
        }


        inline const char *
        getPath(void) {
            return mPath;
        }


        inline const char *
        getPwd(void) {
            return mPwd;
        }


        inline char * const *
        getArgv(void) {
            return (char * const *)mArgv;
        }


        inline const char *
        getTeeInFile(void) {
            return mTeeInFile;
        }


        inline const char *
        getTeeOutFile(void) {
            return mTeeOutFile;
        }


        inline const char *
        getTeeErrFile(void) {
            return mTeeErrFile;
        }


    };


    typedef struct {
        int inFd;
        int outFd;
        int errFd;
        const char *teeInFile;
        const char *teeOutFile;
        const char *teeErrFile;
    } mFdTupleT;


    Mutex mLock;
    std::vector<FullSimplexedCommand *> mCmds;
    std::vector<mCommandParams *> mParams;

    int mInFd;
    int mOutFd;
    int mErrFd;

    int mHeadFd;
    int mTailFd;

    IOPipers *mIOPPtr;

    bool mIsStarted;
    bool mIsWaited;
    bool mWaitResult;

    PipelinedCommands(const PipelinedCommands &obj);
    PipelinedCommands operator = (const PipelinedCommands &obj);





    inline const char *
    mFd2Path(int fd) {
        const char *ret = NULL;
#if defined(NATA_OS_LINUX) || defined(NATA_OS_CYGWIN)
        if (fd >= 0) {
            char path[PATH_MAX];
            struct stat st;

            snprintf(path, sizeof(path), "/proc/%d/fd/%d",
                     (int)getpid(), fd);
            if (lstat(path, &st) == 0 &&
                isBitSet(st.st_mode, S_IFLNK) == true) {
                char realPath[PATH_MAX];
                ssize_t l = readlink(path, realPath, sizeof(realPath));
                if (l >= 0) {
                    l = (ssize_t)(((size_t)l < (sizeof(realPath) - 1)) ?
                                  l : sizeof(realPath) - 1);
                    realPath[l] = '\0';
                    ret = strdup(realPath);
                }
            }
        }
#endif // NATA_OS_LINUX || NATA_OS_CYGWIN
        return ret;
    }


    inline FILE *
    mFDopen(int fd, const char *mode) {
        FILE *ret = fdopen(fd, mode);
        if (ret != NULL) {
            (void)setvbuf(ret, NULL, _IONBF, 0);
        }
        return ret;
    }


    inline void
    mAddCommand(const char *pwd, const char *path, char * const *argv,
                const char *teeInFile = NULL,
                const char *teeOutFile = NULL,
                const char *teeErrFile = NULL) {
        mCommandParams *pPtr = new mCommandParams(pwd, path, argv,
                                                  teeInFile,
                                                  teeOutFile,
                                                  teeErrFile);
        mParams.push_back(pPtr);
        FullSimplexedCommand *cPtr =
            new FullSimplexedCommand(pPtr->getPwd(),
                                     pPtr->getPath(),
                                     pPtr->getArgv());
        mCmds.push_back(cPtr);
    }


    inline const char *
    mStringifyCommands(void) {
        char *ret = NULL;
        std::string buf = "";
        size_t i;
        FullSimplexedCommand *cPtr;
        char * const *argv;

        for (i = 0; i < mCmds.size(); i++) {
            cPtr = mCmds[i];
            argv = cPtr->getArguments();
            while (*argv != NULL) {
                buf += *(argv++);
                buf += " ";
            }
            buf += "| ";
        }
        ret = strdup(buf.c_str());
        size_t len = strlen(ret);
        ret[len - 3] = '\0';

        if (mInFd > 0 || mOutFd > 1) {
            buf = ret;
            free((void *)ret);
            ret = NULL;
            const char *fdPath;

            if (mInFd >= 0 &&
                isValidString(fdPath = mFd2Path(mInFd)) == true) {
                buf += " < ";
                buf += fdPath;
                free((void *)fdPath);
            }
            if (mOutFd >= 0 &&
                isValidString(fdPath = mFd2Path(mOutFd)) == true) {
                buf += " > ";
                buf += fdPath;
                free((void *)fdPath);
            }

            ret = strdup(buf.c_str());
        }

        return (const char *)ret;
    }


    inline bool
    mStart(bool silent = false) {
        bool ret = false;
        size_t i;
        mCommandParams *mPtr;
        FullSimplexedCommand *cPtr;
        IOPiper *iPtr;
        std::vector<mFdTupleT> fdTuples;
        mFdTupleT t0, t1;

        if (mIsStarted == true) {
            return false;
        }

        mIsWaited = false;
        mWaitResult = false;

        for (i = 0; i < mCmds.size(); i++) {
            cPtr = mCmds[i];
            mPtr = mParams[i];
            if (cPtr->start() != true) {
                goto Done;
            }
            t0.inFd = cPtr->childInFd();
            t0.outFd = cPtr->childOutFd();
            t0.errFd = cPtr->childErrFd();
            t0.teeInFile = mPtr->getTeeInFile();
            t0.teeOutFile = mPtr->getTeeOutFile();
            t0.teeErrFile = mPtr->getTeeErrFile();
            fdTuples.push_back(t0);
        }

        mIOPPtr = new IOPipers(silent);

        for (i = 0; i < (fdTuples.size() - 1); i++) {
            t0 = fdTuples[i];
            t1 = fdTuples[i + 1];
            iPtr = mIOPPtr->addPiper(t0.outFd, t1.inFd, true, true);
            iPtr->tee(t0.teeOutFile);
#if 0
            fprintf(stderr, "%d: err: %d -> %d\n", (int)i,
                    t0.errFd, mErrFd);
#endif
            if (mErrFd >= 0) {
                iPtr = mIOPPtr->addPiper(t0.errFd, mErrFd, true, false);
                iPtr->tee(t0.teeErrFile);
            }
        }

        t0 = fdTuples[0];
        mHeadFd = t0.inFd;
        if (mInFd >= 0) {
            iPtr = mIOPPtr->addPiper(mInFd, mHeadFd, false, true);
            iPtr->tee(t0.teeInFile);
        }

        t1 = fdTuples[fdTuples.size() - 1];
        mTailFd = t1.outFd;
        if (mOutFd >= 0) {
            iPtr = mIOPPtr->addPiper(mTailFd, mOutFd, true, false);
            iPtr->tee(t1.teeOutFile);
        }

#if 0
        fprintf(stderr, "last: err: %d -> %d\n", t1.errFd, mErrFd);
#endif
        if (mErrFd >= 0) {
            mIOPPtr->addPiper(t1.errFd, mErrFd, true, false);
        }

        ret = mIOPPtr->start();

        Done:
        if (ret == true) {
            mIsStarted = true;
        }

        return ret;
    }


    inline bool
    mWait(void) {
        if (mIsWaited == false) {
            if (mIsStarted == true) {
                size_t i;
                int nErrors = 0;
                FullSimplexedCommand *cPtr;

                if (mIOPPtr != NULL) {
                    mIOPPtr->wait();
                    if (mIOPPtr->exitCode() != 0) {
                        nErrors++;
                    }
                    delete mIOPPtr;
                    mIOPPtr = NULL;
                }

                for (i = 0; i < mCmds.size(); i++) {
                    cPtr = mCmds[i];       
                    cPtr->wait();
                    if (cPtr->getExitCode() != 0) {
                        nErrors++;
                    }
                }

                mWaitResult = (nErrors == 0) ? true : false;
                mIsStarted = false;
                mIsWaited = true;
            }
        }
        return mWaitResult;
    }


    inline size_t
    mGetCommandsNumber(void) {
        return mCmds.size();
    }


    inline const char * const *
    mGetCommandArguments(size_t i) {
        char * const *ret = NULL;
        if (i < mGetCommandsNumber()) {
            FullSimplexedCommand *cPtr = mCmds[i];
            ret = cPtr->getArguments();
        }
        return (const char * const *)ret;
    }


    inline Process::ProcessFinalStateT
    mGetCommandFinalState(size_t i) {
        Process::ProcessFinalStateT ret = Process::Process_Finally_Unknown;
        if (i < mGetCommandsNumber()) {
            FullSimplexedCommand *cPtr = mCmds[i];
            ret = cPtr->getFinalState();
        }
        return ret;
    }
    

    inline int
    mGetCommandExitCode(size_t i) {
        int ret = EXITCODE_IMPOSSIBLE;
        if (i < mGetCommandsNumber()) {
            FullSimplexedCommand *cPtr = mCmds[i];
            if (cPtr->getFinalState() == Process::Process_Finally_Exited) {
                ret = cPtr->getExitCode();
            }
        }
        return ret;
    }


    inline int
    mGetCommandExitSignal(size_t i) {
        int ret = EXITSIGNAL_IMPOSSIBLE;
        if (i < mGetCommandsNumber()) {
            FullSimplexedCommand *cPtr = mCmds[i];
            if (cPtr->getFinalState() == Process::Process_Finally_Signaled) {
                ret = cPtr->getExitSignal();
            }
        }
        return ret;
    }


    inline bool
    mCommandDumpedCore(size_t i) {
        bool ret = false;
        if (i < mGetCommandsNumber()) {
            FullSimplexedCommand *cPtr = mCmds[i];
            if (cPtr->getFinalState() == Process::Process_Finally_Signaled) {
                ret = cPtr->isCoreDumped();
            }
        }
        return ret;
    }


public:


    PipelinedCommands(int inFd = -1, int outFd = -1, int errFd = -1) :
        // mLock,
        // mCmds,
        // mParams,
        mInFd(inFd),
        mOutFd(outFd),
        mErrFd(errFd),
        mHeadFd(-1),
        mTailFd(-1),
        mIOPPtr(NULL),
        mIsStarted(false),
        mIsWaited(false),
        mWaitResult(false) {
    }


    ~PipelinedCommands(void) {
        size_t i;

        for (i = 0; i < mCmds.size(); i++) {
            delete mCmds[i];
        }

        for (i = 0; i < mParams.size(); i++) {
            delete mParams[i];
        }

        if (mIOPPtr != NULL) {
            delete mIOPPtr;
        }
    }


    inline void
    setInFd(int fd) {
        mInFd = fd;
    }


    inline void
    setOutFd(int fd) {
        mOutFd = fd;
    }


    inline void
    setErrFd(int fd) {
        mErrFd = fd;
    }


    inline int
    getInputFd(void) {
        return mHeadFd;
    }


    inline FILE *
    getInputFILE(void) {
        return mFDopen(getInputFd(), "w");
    }


    inline int
    getOutputFd(void) {
        return mTailFd;
    }


    inline FILE *
    getOutputFILE(void) {
        return mFDopen(getOutputFd(), "r");
    }


    inline bool
    start(bool doWait = true, bool silent = false) {
        bool ret = false;
        ScopedLock l(&mLock);
        ret = mStart(silent);
        if (doWait == true) {
            ret = mWait();
        }
        return ret;
    }


    inline bool
    wait(void) {
        ScopedLock l(&mLock);
        return mWait();
    }


    inline void
    addCommand(const char *pwd, const char *path, char * const *argv,
               const char *teeInFile = NULL,
               const char *teeOutFile = NULL,
               const char *teeErrFile = NULL) {
        mAddCommand(pwd, path, argv, teeInFile, teeOutFile, teeErrFile);
    }


    inline const char *
    stringifyCommands(void) {
        return mStringifyCommands();
    }


    inline void
    printCommands(FILE *fd) {
        const char *cmd = mStringifyCommands();
        if (isValidString(cmd) == true) {
            fprintf(fd, "%s\n", cmd);
            (void)fflush(fd);
            free((void *)cmd);
        }
    }


    inline size_t
    getCommandsNumber(void) {
        ScopedLock l(&mLock);
        return mGetCommandsNumber();
    }


    inline const char * const *
    getCommandArguments(size_t i) {
        ScopedLock l(&mLock);
        return mGetCommandArguments(i);
    }


    inline Process::ProcessFinalStateT
    getCommandFinalState(size_t i) {
        ScopedLock l(&mLock);
        return mGetCommandFinalState(i);
    }


    inline int
    getCommandExitCode(size_t i) {
        ScopedLock l(&mLock);
        return mGetCommandExitCode(i);
    }


    inline int
    getCommandExitSignal(size_t i) {
        ScopedLock l(&mLock);
        return mGetCommandExitSignal(i);
    }


    inline bool
    commandDumpedCore(size_t i) {
        ScopedLock l(&mLock);
        return mCommandDumpedCore(i);
    }


};


#endif // ! __PIPELINEDCOMMANDS_H__
