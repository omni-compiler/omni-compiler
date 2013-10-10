#ifndef __IOPIPER_H__
#define __IOPIPER_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Thread.h>

#include <nata/nata_perror.h>





class IOPiper: public Thread {





private:


    __rcsId("$Id: IOPiper.h 135 2012-08-14 11:02:37Z m-hirano $");





private:


    int mFromFd;
    int mToFd;
    bool mCloseFrom;
    bool mCloseTo;
    const char *mDumpFile;
    int mDumpFd;
    bool mIsSilent;


     inline int
     mOpenDumpFile(const char *file) {
         int ret = -INT_MAX;
         if (isValidString(file) == true) {
             ret = ::open(file, O_WRONLY | O_CREAT | O_TRUNC, 0600);
             if (ret < 0 && mIsSilent == false) {
                 char errbuf[PATH_MAX];
                 snprintf(errbuf, sizeof(errbuf), "open(\"%s\")", mDumpFile);
                 perror(errbuf);
             }
         }
         return ret;
     }


    inline void
    mCloseDumpFile(int &fd) {
        if (fd >= 0) {
            (void)::close(fd);
            fd = -INT_MAX;
        }
    }


    inline bool
    mRead(int &fd, char *buf, ssize_t &n) {
        bool ret = false;
        if (fd >= 0) {
            errno = 0;
            ssize_t rN = ::read(fd, (void *)buf, (size_t)n);
            if (rN > 0) {
                ret = true;
            } else if (rN < 0) {
                if (mIsSilent == false) {
                    perror("read");
                }
            }
            n = rN;
        }
        return ret;
    }


     inline bool
     mWrite(int &fd, char *buf, ssize_t &n) {
         bool ret = false;
         if (fd >= 0) {
             errno = 0;
             ssize_t wN = (siolen_t)nata_WriteInt8(fd, (int8_t *)buf,
                                                   (iolen_t)n);
             if (wN < 0 || wN != n) {
                 if (fd == mDumpFd) {
                     if (mIsSilent == false) {
                         char errbuf[PATH_MAX];
                         snprintf(errbuf, sizeof(errbuf), "write(\"%s\")",
                                  mDumpFile);
                     }
                     (void)::close(fd);
                     fd = -INT_MAX;
                 } else {
                     if (mIsSilent == false) {
                         ::perror("write");
                     }
                 }
                 goto Done;
             }
             ret = true;
         }
         Done:
         return ret;
    }


    int
    run(void) {
        int ret = 1;
        bool doLoop = true;
        ssize_t n;
        char buf[8192];

        mDumpFd = mOpenDumpFile(mDumpFile);

        while (doLoop == true) {
            n = sizeof(buf);
            if ((doLoop = mRead(mFromFd, buf, n)) == false) {
                if (n == 0) {
                    ret = 0;
                }
                break;
            }

            if ((doLoop = mWrite(mToFd, buf, n)) == true) {
                (void)mWrite(mDumpFd, buf, n);
            } else {
                break;
            }
        }

        mCloseDumpFile(mDumpFd);

        if (mCloseFrom == true) {
            (void)close(mFromFd);
        }
        if (mCloseTo == true) {
            (void)close(mToFd);
        }

        return ret;
    }


public:
    IOPiper(int fromFd, int toFd,
            bool closeFrom = false, bool closeTo = false) :
        Thread(), 
        mFromFd(fromFd),
        mToFd(toFd),
        mCloseFrom(closeFrom),
        mCloseTo(closeTo),
        mDumpFile(NULL),
        mDumpFd(-INT_MAX),
        mIsSilent(false) {
        char buf[32];
        snprintf(buf, sizeof(buf), "I/O Piper %d -> %d", fromFd, toFd);
        setName(buf);
    }


    ~IOPiper(void) {
        free((void *)mDumpFile);
        mCloseDumpFile(mDumpFd);
    }


    inline void
    setSilent(void) {
        mIsSilent = true;
    }


    inline void
    tee(const char *file) {
        if (isValidString(file) == true) {
            free((void *)mDumpFile);
            mDumpFile = strdup(file);
        }
    }


private:
    IOPiper(const IOPiper &obj);
    IOPiper operator = (const IOPiper &obj);


};


#endif // __IOPIPER_H__
