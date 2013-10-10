/* 
 * $Id: BoundedBlockingQueue.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __BOUNDEDBLOCKINGQUEUE_H__
#define __BOUNDEDBLOCKINGQUEUE_H__

#include <nata/nata_rcsid.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>
#include <nata/ScopedLock.h>
#include <nata/BlockingContainer.h>


#include <queue>


template<typename T>
class BoundedBlockingQueue: public BlockingContainer {



private:
    __rcsId("$Id: BoundedBlockingQueue.h 86 2012-07-30 05:33:07Z m-hirano $");


    typedef void (*__ElementDestructProcT)(T obj, void *arg);

    std::queue<T> mQ;

    size_t mMaxLen;

    WaitCondition mMaxWait;

    Mutex mLock;
    WaitCondition mQWait;

    __ElementDestructProcT mDestructor;
    void *mDestructArg;

    volatile bool mIsStopping;


protected:
    inline void
    setDeleteHook(__ElementDestructProcT p, void *a = NULL) {
        mDestructor = p;
        mDestructArg = a;
    }


private:
    inline void
    pDeleteHook(T obj) {
        if (mDestructor != NULL) {
            (mDestructor)(obj, mDestructArg);
        }
    }


public:
    inline
    BoundedBlockingQueue(
        size_t maxLen = BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH) :
        // mQ,
        mMaxLen((maxLen <= 0) ?
                BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH : maxLen),
        mDestructor(NULL),
        mDestructArg(NULL),
        mIsStopping(false) {
        (void)rcsid();
    }


    inline virtual
    ~BoundedBlockingQueue(void) {
	stop();
    }


private:
    BoundedBlockingQueue(const BoundedBlockingQueue &obj);
    BoundedBlockingQueue operator = (const BoundedBlockingQueue &obj);





public:
    inline bool
    put(T tObj, int64_t waitUSec = -1,
        BlockingContainer::ContainerStatus *sPtr = NULL) {
	bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

	ReCheck:
	if (mIsStopping == true) {
            pDeleteHook(tObj);
            st = BlockingContainer::Status_Container_No_Longer_Valid;
	    goto BailOut;
	}

	// Fetch current queue length.
	if (mQ.size() < mMaxLen) {
	    // Push the object.
	    mQ.push(tObj);
            st = BlockingContainer::Status_OK;
	    ret = true;
	} else {
	    if (waitUSec < 0) {
		// Wait until someone dequeue at least an object.
		mMaxWait.wait(&mLock);
		goto ReCheck;
	    } else {
		if (mMaxWait.timedwait(&mLock, (uint64_t)waitUSec) == true) {
		    // Someone dequeu at least an object.
		    goto ReCheck;
		} else {
		    // Timedout.
                    st = BlockingContainer::Status_Timedout;
		    goto BailOut;
		}
	    }
	}

	BailOut:
	if (ret == true) {
	    // Wake waiters waiting for an object up.
	    mQWait.wakeAll();
	}
        if (sPtr != NULL) {
            *sPtr = st;
        }

	return ret;
    }


    inline bool
    get(T &tObj, int64_t waitUSec = -1,
        BlockingContainer::ContainerStatus *sPtr = NULL) {
	bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

	ReCheck:
	if (mIsStopping == true) {
            st = BlockingContainer::Status_Container_No_Longer_Valid;
	    goto BailOut;
	}

	// Fetch current queue length.
	if (mQ.size() > 0) {
	    // At least an object is available. Fetch an object and
	    // dequeue it.
	    tObj = (T)(mQ.front());
	    mQ.pop();
            st = BlockingContainer::Status_OK;
	    ret = true;
	} else {
	    if (waitUSec < 0) {
		// No object is available. Wait until new one comes.
		mQWait.wait(&mLock);
		goto ReCheck;
	    } else {
		// No object is available. Wait until new one comes or
		// wait waitUSec.
		if (mQWait.timedwait(&mLock, (uint64_t)waitUSec) == true) {
		    goto ReCheck;
		} else {
		    // Timedout.
                    st = BlockingContainer::Status_Timedout;
		    goto BailOut;
		}
	    }
	}

	BailOut:
	if (ret == true) {
	    // Wake waiters waiting for a queue slot up.
	    mMaxWait.wakeAll();
	}
        if (sPtr != NULL) {
            *sPtr = st;
        }

	return ret;
    }


    inline bool
    peek(T &tObj, int64_t waitUSec = -1,
         BlockingContainer::ContainerStatus *sPtr = NULL) {
	bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

	ReCheck:
	if (mIsStopping == true) {
            st = BlockingContainer::Status_Container_No_Longer_Valid;
	    goto BailOut;
	}

	// Fetch current queue length.
	if (mQ.size() > 0) {
	    // At least an object is available. Fetch an object and
	    // dequeue it.
	    tObj = (T)(mQ.front());
            st = BlockingContainer::Status_OK;
	    ret = true;
	} else {
	    if (waitUSec < 0) {
		// No object is available. Wait until new one comes.
		mQWait.wait(&mLock);
		goto ReCheck;
	    } else {
		// No object is available. Wait until new one comes or
		// wait waitUSec.
		if (mQWait.timedwait(&mLock, (uint64_t)waitUSec) == true) {
		    goto ReCheck;
		} else {
		    // Timedout.
                    st = BlockingContainer::Status_Timedout;
		    goto BailOut;
		}
	    }
	}

	BailOut:
	if (ret == true) {
	    // Wake waiters waiting for a queue slot up.
	    mMaxWait.wakeAll();
	}
        if (sPtr != NULL) {
            *sPtr = st;
        }

	return ret;
    }


    void
    stop(void) {
	bool doWakeUp = false;

        ScopedLock l(&mLock);

	if (mIsStopping == false) {
	    mIsStopping = true;

	    mMaxLen = 0;
	    while (mQ.empty() != true) {
		T tObj = (T)(mQ.front());
		pDeleteHook(tObj);
		mQ.pop();
	    }

	    doWakeUp = true;
	}

	if (doWakeUp == true) {
	    mMaxWait.wakeAll();
	    mQWait.wakeAll();
	}
    }


    bool
    isStopping(void) {
        ScopedLock l(&mLock);
	return mIsStopping;
    }


    void
    clear(void) {
        ScopedLock l(&mLock);
	while (mQ.empty() != true) {
	    T tObj = (T)(mQ.front());
	    pDeleteHook(tObj);
	    mQ.pop();
	}
	mMaxWait.wakeAll();
	mQWait.wakeAll();
    }


    size_t
    size(void) {
        ScopedLock l(&mLock);
	return mQ.size();
    }


    bool
    empty(void) {
        ScopedLock l(&mLock);
	return mQ.empty();
    }
};


#endif // ! __BOUNDEDBLOCKINGQUEUE_H__
