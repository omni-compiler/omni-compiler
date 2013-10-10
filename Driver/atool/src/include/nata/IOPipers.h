#ifndef __IOPIPERS_H__
#define __IOPIPERS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/IOPiper.h>

#include <vector>

#include <nata/nata_perror.h>





class IOPipers: public Thread {





private:


    __rcsId("$Id: IOPipers.h 135 2012-08-14 11:02:37Z m-hirano $");





private:


    std::vector<IOPiper *> mPipers;
    bool mIsSilent;

    int
    run(void) {
        int nErrors = 0;
        size_t i;
        IOPiper *pPtr;

        for (i = 0; i < mPipers.size(); i++) {
            pPtr = mPipers[i];
            pPtr->start();
        }

        for (i = 0; i < mPipers.size(); i++) {
            pPtr = mPipers[i];
            pPtr->wait();
            if (pPtr->exitCode() != 0) {
                nErrors++;
            }
        }

        return (nErrors == 0) ? 0 : 1;
    }


public:


    IOPipers(bool silent = false) :
        Thread(),
        mIsSilent(silent) {
        setName("I/O Pipers");
    }


    ~IOPipers(void) {
        size_t i;
        for (i = 0; i < mPipers.size(); i++) {
            delete mPipers[i];
        }
    }


    inline IOPiper *
    addPiper(int fromFd, int toFd,
             bool closeFrom = false, bool closeTo = false,
             const char *dumpFile = NULL) {
        IOPiper *ipPtr = new IOPiper(fromFd, toFd, closeFrom, closeTo);
        mPipers.push_back(ipPtr);
        ipPtr->tee(dumpFile);
        if (mIsSilent == true) {
            ipPtr->setSilent();
        }
        return ipPtr;
    }
        

private:
    IOPipers(const IOPipers &obj);
    IOPipers operator = (const IOPipers &obj);
        

};


#endif // __IOPIPERS_H__
