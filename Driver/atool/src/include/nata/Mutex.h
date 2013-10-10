/* 
 * $Id: Mutex.h 195 2012-09-05 04:48:15Z m-hirano $
 */
#ifndef __MUTEX_H__
#define __MUTEX_H__


#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/nata_perror.h>


#ifdef linux
#define PTHREAD_MUTEX_DEFAULT		PTHREAD_MUTEX_FAST_NP
#define PTHREAD_MUTEX_RECURSIVE		PTHREAD_MUTEX_RECURSIVE_NP
#define PTHREAD_MUTEX_ERRORCHECK	PTHREAD_MUTEX_ERRORCHECK_NP
#endif // linux





class WaitCondition;


class Mutex {
    friend class WaitCondition;	// WaitCondition peeks mLock directly.


public:


    typedef enum {
        Mutex_Type_Unknown = -(INT_MAX),
        Mutex_Type_Default = (int)(PTHREAD_MUTEX_DEFAULT),
        Mutex_Type_Recursive = (int)(PTHREAD_MUTEX_RECURSIVE),
        Mutex_Type_ErrorCheck = (int)(PTHREAD_MUTEX_ERRORCHECK)
    } MutexTypeT;


private:
    __rcsId("$Id: Mutex.h 195 2012-09-05 04:48:15Z m-hirano $");


    pthread_mutex_t mLock;
    pthread_mutexattr_t mAttr;
#if defined(NATA_API_POSIX)
    volatile pthread_t mLockedBy;
#elif defined(NATA_API_WIN32API)
    volatile DWORD mWinLockedBy;
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    pid_t mCreatorPid;
    MutexTypeT mType;





    inline void
    mInitialize(void) {
        int st;

	errno = 0;
#ifndef NATA_OS_CYGWIN
        if ((st = pthread_mutexattr_init(&mAttr)) != 0) {
            int sErrno = errno;
	    errno = st;
	    perror("pthread_mutexattr_init");
	    errno = sErrno;
	    fatal("Can't initialize a mutex attribute.\n");
        }
#else
        //
        // FIXME:
        //	The cygwin has been fu*ked up around pthread thingies.
        //
        (void)pthread_mutexattr_init(&mAttr);
#endif // ! NATA_OS_CYGWIN

	errno = 0;
        if ((st = pthread_mutexattr_settype(&mAttr,
                                            (int)mType)) != 0) {
            int sErrno = errno;
	    errno = st;
	    perror("pthread_mutexattr_settype");
	    errno = sErrno;
	    fatal("Can't initialize a mutex attribute.\n");
        }

        errno = 0;
	if ((st = pthread_mutex_init(&mLock, &mAttr)) != 0) {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_mutex_init");
	    errno = sErrno;
	    fatal("Can't create a mutex.\n");
	}
    }





public:


    Mutex(MutexTypeT type = Mutex_Type_Default) :
        // mLock,
        // mAttr,
#if defined(NATA_API_POSIX)
        mLockedBy(INVALID_THREAD_ID),
#elif defined(NATA_API_WIN32API)
        mWinLockedBy((DWORD)-1),
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
        mCreatorPid(getpid()),
        mType(type) {
        mInitialize();
    }


    ~Mutex(void) {
	// if the caller was forked, not destroy.
	if (getpid() == mCreatorPid) {
	    (void)pthread_mutex_destroy(&mLock);
            (void)pthread_mutexattr_destroy(&mAttr);
	}
    }


private:
    Mutex(const Mutex &obj);
    Mutex operator = (const Mutex &obj);



public:


    inline void
    reinitialize(void) {
        pid_t pid = getpid();
        if (pid != mCreatorPid) {
            mInitialize();
            mCreatorPid = pid;
#if defined(NATA_API_POSIX)
            mLockedBy = INVALID_THREAD_ID;
#elif defined(NATA_API_WIN32API)
            mWinLockedBy = (DWORD)-1;
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
        }
    }


    inline bool
    lock(void) {
	int st;
	errno = 0;
	bool ret = ((st = pthread_mutex_lock(&mLock)) == 0) ? true : false;
	if (ret == true) {
#if defined(NATA_API_POSIX)
	    mLockedBy = pthread_self();
#elif defined(NATA_API_WIN32API)
            mWinLockedBy = GetCurrentThreadId();
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
	} else {
	    int sErrno = errno;
	    errno = st;
	    perror("pthread_mutex_lock");
	    errno = sErrno;
            abort();
	}
	return ret;
    }


    inline bool
    unlock(void) {
#if defined(NATA_API_POSIX)
        mLockedBy = INVALID_THREAD_ID;
#elif defined(NATA_API_WIN32API)
        mWinLockedBy = (DWORD)-1;
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
        int st;
        errno = 0;
        bool ret = ((st = pthread_mutex_unlock(&mLock)) == 0) ? true : false;
        if (ret != true) {
            int sErrno = errno;
            errno = st;
            perror("pthread_mutex_unlock");
            errno = sErrno;
            abort();
        }
        return ret;
    }


    inline bool
    trylock(void) {
	int st;
	errno = 0;
	bool ret = ((st = pthread_mutex_trylock(&mLock)) == 0) ? true : false;
	if (ret == true) {
#if defined(NATA_API_POSIX)
	    mLockedBy = pthread_self();
#elif defined(NATA_API_WIN32API)
            mWinLockedBy = GetCurrentThreadId();
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
	} else {
	    if (errno != EBUSY && st != EBUSY) {
		int sErrno = errno;
		errno = st;
		perror("pthread_mutex_lock");
		errno = sErrno;
	    }
	}
	return ret;
    }


#if defined(NATA_API_POSIX)
    inline pthread_t
    lockedBy(void) {
	return mLockedBy;
    }
#elif defined(NATA_API_WIN32API)
    inline DWORD
    lockedBy(void) {
	return mWinLockedBy;
    }
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */


    inline bool
    isLockedByMe(void) {
#if defined(NATA_API_POSIX)
	return (mLockedBy == pthread_self()) ? true : false;
#elif defined(NATA_API_WIN32API)
        return (mWinLockedBy == GetCurrentThreadId()) ? true : false;
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    }


    inline MutexTypeT
    getType(void) {
        return mType;
    }


};


#endif // ! __MUTEX_H__
