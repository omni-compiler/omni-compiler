/*
 * $Id: CircularBuffer.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __CIRCULARBUFFER_H__
#define __CIRCULARBUFFER_H__


#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>
#include <nata/ScopedLock.h>
#include <nata/BlockingContainer.h>

#include <nata/nata_perror.h>


template <typename T>
class CircularBuffer: public BlockingContainer {



private:
    __rcsId("$Id: CircularBuffer.h 86 2012-07-30 05:33:07Z m-hirano $");


    typedef bool (*__ElementAllocateProcT)(T *ptr, void *arg);
    typedef void (*__ElementDestructProcT)(T ptr, void *arg);

    Mutex mLock;
    WaitCondition mReaderWait;
    WaitCondition mWriterWait;

    T *mBuf;			// The buffer itself.

    __ElementAllocateProcT mAllocator;
    void *mAllocateArg;
    __ElementDestructProcT mDestructor;
    void *mDestructArg;

    int64_t mBufSize;		// A size of mBuf (a # of elements).
    int64_t mGuardSize;		// A # of "guarded" elements.
    int64_t mRIdx;		// An absolute read index.
    int64_t mWIdx;		// An absolute write index.

    int64_t mActualRoomSize;
    
    bool mIsStopping;
    bool mIsSetup;





protected:


    inline void
    setAllocationHook(__ElementAllocateProcT p, void *arg = NULL) {
        ScopedLock l(&mLock);
        mAllocator = p;
        mAllocateArg = arg;
    }


    inline void
    setDestructionHook(__ElementDestructProcT p, void *arg = NULL) {
        ScopedLock l(&mLock);
        mDestructor = p;
        mDestructArg = arg;
    }


    inline void
    allocateElements(void) {
        pSetup();
    }


    inline void
    setup(void) {
        pSetup();
    }





private:


    inline void
    pSetup(void) {
        ScopedLock l(&mLock);
        if (mIsSetup == false) {
            for (int64_t i = 0; i < mBufSize; i++) {
                if (pConstructHook(mBuf[i]) == false) {
                    fatal("Can't allocate an element.\n");
                }
            }
            mIsSetup = true;
        }
    }


    inline bool
    pConstructHook(T &anElem) {
        if (mAllocator != NULL) {
            T newElem;
            bool ret;
            if ((ret = (mAllocator)(&newElem, mAllocateArg)) == true) {
                anElem = newElem;
            }
            return ret;
        } else {
            return true;
        }
    }


    inline void
    pDestructHook(T &anElem) {
        if (mDestructor != NULL) {
            (mDestructor)(anElem, mDestructArg);
        }
    }


    inline void
    adjustIndices(void) {
        if (mWIdx < 0) {
            //
            // I KNOW that this couldn't be happened in the most
            // situation, but just for the case:
            //

            mWIdx = mBufSize;
            mRIdx = mRIdx % mBufSize;

            // Note that this makes keep mWIdx > mRIdx ALWAYS.
        }
    }





public:


    CircularBuffer(uint64_t n = CIRCULARBUFFER_DEFALUT_MAX_LENGTH,
                   uint64_t g = 1) :
        // mLock,
        // mReaderWait,
        // mWriterWait,
        mBuf(NULL),
        mAllocator(NULL),
        mDestructor(NULL),
        // mBufSize(0),
        mGuardSize(0),
        mRIdx(0),
        mWIdx(0),
        mActualRoomSize(0),
        mIsStopping(false),
        mIsSetup(false) {

        if (n > LLONG_MAX) {
            fatal("Too many elements to initialize.\n");
        } else {
            mBufSize = (int64_t)n;
        }

        if (g >= n) {
            fatal("Too many guarded elements.\n");
        } else {
            mGuardSize = (int64_t)g;
        }

        mBuf = new T[mBufSize];

        mActualRoomSize = mBufSize - mGuardSize;
    }


    virtual
    ~CircularBuffer(void) {
        stop();
    }


private:
    CircularBuffer(const CircularBuffer &obj);
    CircularBuffer operator = (const CircularBuffer &obj);





public:


    inline bool
    getWriterElement(T &obj, int64_t waitUSec = -1,
                     BlockingContainer::ContainerStatus *sPtr = NULL) {
        bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

        ReCheck:
        if (mIsStopping == false) {
            adjustIndices();
        
            //
            // Now mWIdx >= mRIdx.
            //
            if ((mWIdx - mRIdx) < mActualRoomSize) {
                //
                // INCLUDES the case (mWIdx - mRIdx) equals to zero.
                //
                obj = mBuf[mWIdx % mBufSize];
                mWIdx++;
                //
                // Let the readers awake.
                //
                mReaderWait.wakeAll();
                ret = true;
                st = BlockingContainer::Status_OK;
                goto BailOut;
            } else {
                //
                // To avoid overwriting, wait the reader reads.
                //
                if (waitUSec < 0) {
                    //
                    // Let the writer sleep.
                    //
                    mWriterWait.wait(&mLock);
                    goto ReCheck;
                } else {
                    if (mWriterWait.timedwait(&mLock, (uint64_t)waitUSec) ==
                        true) {
                        goto ReCheck;
                    } else {
                        st = BlockingContainer::Status_Timedout;
                        goto BailOut;
                    }
                }
            }
        } else {
            st = BlockingContainer::Status_Container_No_Longer_Valid;
        }

        BailOut:
        if (sPtr != NULL) {
            *sPtr = st;
        }
        return ret;
    }


    inline bool
    getReaderElement(T &obj, int64_t waitUSec = -1,
                     BlockingContainer::ContainerStatus *sPtr = NULL) {
        bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

        ReCheck:
        if (mIsStopping == false) {
            adjustIndices();

            //
            // Now mWIdx >= mRIdx.
            //
            if ((mWIdx - mRIdx) > mGuardSize) {
                obj = mBuf[mRIdx % mBufSize];
                mRIdx++;
                //
                // Let the writers awake.
                //
                mWriterWait.wakeAll();
                ret = true;
                st = BlockingContainer::Status_OK;
                goto BailOut;
            } else {
                //
                // Wait the writer writes.
                //
                if (waitUSec < 0) {
                    //
                    // Let the reader sleep.
                    //
                    mReaderWait.wait(&mLock);
                    goto ReCheck;
                } else {
                    if (mReaderWait.timedwait(&mLock, (uint64_t)waitUSec) ==
                        true) {
                        goto ReCheck;
                    } else {
                        st = BlockingContainer::Status_Timedout;
                        goto BailOut;
                    }
                }
            }
        } else {
            st = BlockingContainer::Status_Container_No_Longer_Valid;
        }

        BailOut:
        if (sPtr != NULL) {
            *sPtr = st;
        }
        return ret;
    }


    inline bool
    peekReaderElement(T &obj, int64_t waitUSec = -1,
                      BlockingContainer::ContainerStatus *sPtr = NULL) {
        bool ret = false;
        BlockingContainer::ContainerStatus st = 
            BlockingContainer::Status_Any_Failure;

        ScopedLock l(&mLock);

        ReCheck:
        if (mIsStopping == false) {
            adjustIndices();

            //
            // Now mWIdx >= mRIdx.
            //
            if ((mWIdx - mRIdx) > mGuardSize) {
                obj = mBuf[mRIdx % mBufSize];
                //
                // Let the writers awake.
                //
                mWriterWait.wakeAll();
                ret = true;
                st = BlockingContainer::Status_OK;
                goto BailOut;
            } else {
                //
                // Wait the writer writes.
                //
                if (waitUSec < 0) {
                    //
                    // Let the reader sleep.
                    //
                    mReaderWait.wait(&mLock);
                    goto ReCheck;
                } else {
                    if (mReaderWait.timedwait(&mLock, (uint64_t)waitUSec) ==
                        true) {
                        goto ReCheck;
                    } else {
                        st = BlockingContainer::Status_Timedout;
                        goto BailOut;
                    }
                }
            }
        } else {
            st = BlockingContainer::Status_Container_No_Longer_Valid;
        }

        BailOut:
        if (sPtr != NULL) {
            *sPtr = st;
        }
        return ret;
    }


    inline void
    stop(void) {
        ScopedLock l(&mLock);

        if (mIsStopping != true) {
            mIsStopping = true;

            for (int64_t i = 0; i < mBufSize; i++) {
                pDestructHook(mBuf[i]);
            }
            delete [] mBuf;

            mWriterWait.wakeAll();
            mReaderWait.wakeAll();
        }
    }


    bool
    isStopping(void) {
        ScopedLock l(&mLock);
	return mIsStopping;
    }


    inline uint64_t
    maxSize(void) {
        return (uint64_t)mBufSize;
    }


    inline uint64_t
    writableSize(void) {
        ScopedLock l(&mLock);

        adjustIndices();
        int64_t ret = mActualRoomSize - (mWIdx - mRIdx);
        if (ret < 0) {
            ret = 0;
        }
        return (uint64_t)ret;
    }


    inline uint64_t
    readableSize(void) {
        ScopedLock l(&mLock);

        adjustIndices();
        int64_t ret = mWIdx - mRIdx - mGuardSize;
        if (ret < 0) {
            ret = 0;
        }
        return (uint64_t)ret;
    }
};


#endif // ! __CIRCULARBUFFER_H__
