/* 
 * $Id: IOHandle.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __IOHANDLE_H__
#define __IOHANDLE_H__

#define IN_NATA_SAFE_SYSCALL

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>
#include <nata/nata_macros.h>
#include <nata/nata_safe_syscall.h>
#include <nata/nata_util.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>
#include <nata/ScopedLock.h>

#include <nata/nata_perror.h>


namespace IOHandleStatics {
    static const char *modeStr[] = { NULL, "r", "w", "rw", NULL };
}



class IOHandle {



private:
    __rcsId("$Id: IOHandle.h 86 2012-07-30 05:33:07Z m-hirano $");



public:
    typedef enum {
        Type_Unknown = 0,

        Type_Tty,
        Type_Pty,

        Type_StreamSocketInet,
        Type_StreamSocketUNIX,

        Type_DatagramSocketInet,
        Type_DatagramSocketUNIX,

        Type_Pipe,

        Type_Generic
    } HandleType;


    typedef enum {
        Direction_Unknown = 0,
	Direction_ReadOnly,
	Direction_WriteOnly,
        Direction_ReadWrite
    } IODirection;



private:

    HandleType mType;		// Handle type.
    IODirection mDirection;	// IO direction.
    bool mIsNonBlockingIO;	// the handle is set to blocking mode
    				// or not.
    Mutex mLock;		// a lock.

    int mFd;			// a file descriptor.
				// -1 at initialization.
    FILE *mStdFd;		// a file descriptor (FILE *).
    				// NULL at initialization.

    bool mIsDeleting;
    bool mIsClosed;



    inline const char *
    mDirection2Mode(IODirection dir) {
        using IOHandleStatics::modeStr;
        int i = (int)dir;
        return (i >= 0 && i <= (int)Direction_ReadWrite) ? 
            modeStr[i] : NULL;
    }

    inline void
    mFlush(void) {
        if (mStdFd != NULL) {
            if (mDirection == Direction_WriteOnly ||
                mDirection == Direction_ReadWrite) {
                (void)::fflush(mStdFd);
            }
        }
    }

    inline void
    mClose(int exceptThis = -1) {
        if (mStdFd != NULL) {
            mFlush();

            int chkFd = ::fileno(mStdFd);
            (void)::fclose(mStdFd);
            mStdFd = NULL;

            // Note:
            //
            //  If mFd == chkFd, ::fclose(mStdFd) implies
            //	::close(mFd). So we must avoid ::close(mFd) again,
            //	since there are possibilities that someone ::open() a
            //	new file after ::fclose(mStdFd) and the new file's
            //	descriptor could be identical to mFd.

            if (chkFd != mFd) {
                (void)::close(mFd);
            }
        } else if (mFd >= 0) {
            if (exceptThis != mFd) {
                (void)::close(mFd);
                mFd = -1;
            }
        }
    }

    inline bool
    mFdopen(int fd, IODirection dir) {
        const char *mode = mDirection2Mode(dir);
        if (isValidString(mode) == false) {
            return false;
        }
        FILE *stdFd = fdopen(fd, mode);
        if (stdFd == NULL) {
            perror("fdopen");
            return false;
        }

        mClose(fd);
        mStdFd = stdFd;
        mFd = fd;

        return true;
    }



protected:
    inline void
    lock(void) {
        mLock.lock();
    }

    inline void
    unlock(void) {
        mLock.unlock();
    }

    inline bool
    setHandle(int fd, bool doAllocateStdio = false) {
        if (fd >= 0) {
            ScopedLock l(&mLock);
            mClose(fd);
            mFd = fd;
            mIsNonBlockingIO = nata_IsNonBlock(mFd);
            if (doAllocateStdio == true) {
                return mFdopen(fd, mDirection);
            }
            return true;
        }
        return false;
    }

    inline bool
    setHandle(FILE *stdFd) {
        if (stdFd != NULL) {
            ScopedLock l(&mLock);
            int fd = ::fileno(stdFd);
            mClose(fd);
            mFd = fd;
            mIsNonBlockingIO = nata_IsNonBlock(mFd);
            return true;
        }
        return false;
    }

    inline void
    setNonBlockingIO(bool setNonBlock) {
        if (mFd >= 0) {
            if (mIsNonBlockingIO == setNonBlock) {
                return;
            }

            ScopedLock l(&mLock);

            int s = ::fcntl(mFd, F_GETFL, 0);
            if (s < 0) {
                perror("fcntl");
                return;
            }
            if (setNonBlock == true) {
                s |= O_NONBLOCK;
            } else {
                s &= ~(O_NONBLOCK);
            }
            if (::fcntl(mFd, F_SETFL, s) < 0) {
                perror("fcntl");
                return;
            }
            mIsNonBlockingIO = setNonBlock;
        }
    }

    inline ssize_t
    rawReadInt8(int8_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_ReadInt8(mFd, buf, len);
    }

    inline ssize_t
    rawWriteInt8(const int8_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_WriteInt8(mFd, buf, len);
    }

    inline ssize_t
    rawReadInt16(int16_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_ReadInt16(mFd, buf, len);
    }

    inline ssize_t
    rawWriteInt16(const int16_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_WriteInt16(mFd, buf, len);
    }

    inline ssize_t
    rawReadInt32(int32_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_ReadInt32(mFd, buf, len);
    }

    inline ssize_t
    rawWriteInt32(const int32_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_WriteInt32(mFd, buf, len);
    }

    inline ssize_t
    rawReadInt64(int64_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_ReadInt64(mFd, buf, len);
    }

    inline ssize_t
    rawWriteInt64(const int64_t *buf, size_t len) {
        ScopedLock l(&mLock);
        return nata_WriteInt64(mFd, buf, len);
    }



public:
    IOHandle(void) :
        mType(Type_Unknown),
        mDirection(Direction_Unknown),
        // mIsNonBlockingIO,
        // mLock,
        mFd(-1),
        mStdFd(NULL),
        mIsDeleting(false),
        mIsClosed(false) {
    }

    IOHandle(int fd,
             IOHandle::IODirection dir,
             bool doAllocateStdio = false) :
        mType(Type_Unknown),
        mDirection(dir),
        // mIsNonBlockingIO,
        // mLock,
        mFd(fd),
        mStdFd(NULL),
        mIsDeleting(false),
        mIsClosed(false) {
        setHandle(fd, doAllocateStdio);
    }


private:
    IOHandle(const IOHandle &obj);
    IOHandle operator = (const IOHandle &obj);



public:


    inline void
    close(void) {
        ScopedLock l(&mLock);

        if (mIsClosed == false) {
            mClose();
            mIsClosed = true;
        }
    }

    inline bool
    allocateStdio(void) {
        ScopedLock l(&mLock);

        if (mStdFd != NULL) {
            return true;
        } else {
            return mFdopen(mFd, mDirection);
        }
    }

};


#endif // ! __IOHANDLE_H__
