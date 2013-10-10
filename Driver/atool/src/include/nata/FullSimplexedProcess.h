#ifndef __FULLSIMPLEXEDPROCESS_H__
#define __FULLSIMPLEXEDPROCESS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Process.h>

#include <nata/nata_perror.h>





class FullSimplexedProcess: public Process {





private:


    __rcsId("$Id: FullSimplexedProcess.h 111 2012-08-07 13:04:22Z m-hirano $");





protected:


    virtual int
    run(void) {
        return 0;
    }





private:


    int
    runChild(void) {
        return run();
    }


    int
    runParent(void) {
        // do nothig.
        return 0;
    }


    FullSimplexedProcess(const FullSimplexedProcess &obj);
    FullSimplexedProcess operator = (const FullSimplexedProcess &obj);


    inline FILE *
    mFDopen(int fd, const char *mode) {
        FILE *ret = fdopen(fd, mode);
        if (ret != NULL) {
            (void)setvbuf(ret, NULL, _IONBF, 0);
        }
        return ret;
    }





public:


    FullSimplexedProcess(const char *pwd = NULL) :
        Process(pwd, "", "", "") {
    }


    inline int
    childInFd(void) {
        return Process::childInFd();
    }


    inline int
    childOutFd(void) {
        return Process::childOutFd();
    }


    inline int
    childErrFd(void) {
        return Process::childErrFd();
    }


    inline FILE *
    childInFILE(void) {
        return mFDopen(childInFd(), "w");
    }


    inline FILE *
    childOutFILE(void) {
        return mFDopen(childOutFd(), "r");
    }


    inline FILE *
    childErrFILE(void) {
        return mFDopen(childErrFd(), "r");
    }


    inline bool
    start(void) {
        return Process::start(Process::Process_Sync_ParentCares,
                              Process::Process_IPC_Pipe);
    }


};


#endif // __FULLSIMPLEXEDPROCESS_H__
