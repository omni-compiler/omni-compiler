/*
 * $Id: ChunkedBuffer.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __CHUNKEDBUFFER_H__
#define __CHUNKEDBUFFER_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/WaitCondition.h>
#include <nata/ScopedLock.h>
#include <nata/BoundedBlockingQueue.h>

#include <nata/nata_perror.h>

#include <utility>

template <typename T>
class ChunkedBuffer: public BlockingContainer {


private:
    typedef T * (*__ChunkAllocateProcT)(size_t s, void *arg);
    typedef void (*__ChunkDestructProcT)(T *ptr, void *arg);

    class aChunk: public std::pair<T *, size_t> {
    private:
        aChunk(const aChunk &obj);
        aChunk operator = (const aChunk &obj);

        __ChunkDestructProcT mDestructor;
        void *mDestructorArg;

        inline void
        pDeleteHook(T *p) {
            if (mDestructor != NULL) {
                (mDestructor)(p, mDestructorArg);
            }
        }

    public:
        inline
        aChunk(T *p, size_t s,
               __ChunkDestructProcT dProc = NULL,
               void *arg = NULL) :
            std::pair<T *, size_t>(p, s),
            mDestructor(dProc),
            mDestructorArg(arg) {
        }

        inline
        ~aChunk(void) {
            if (this->first != NULL) {
                pDeleteHook(this->first);
            }
        }

        inline T *
        data(void) {
            return this->first;
        }

        inline size_t
        size(void) {
            return this->second;
        }
    };
        

    class aChunkQueue: public BoundedBlockingQueue<aChunk *> {
    private:
        static inline void
        deleteHook(aChunk *p, void *arg) {
            (void)arg;
            if (p != NULL) {
                delete p;
            }
        }
        aChunkQueue(const aChunkQueue &obj);
        aChunkQueue operator = (const aChunkQueue &obj);

    public:
        inline
        aChunkQueue(size_t maxLen =
                    BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH) :
            BoundedBlockingQueue<aChunk *>(maxLen) {
            setDeleteHook(deleteHook, this);
        }
    };


protected:
    aChunkQueue *mQPtr;

private:
    Mutex mLock;
    Mutex mTotalSizeLock;
    WaitCondition mTotalSizeWait;
    __ChunkAllocateProcT mCreator;
    void *mCreatorArg;
    __ChunkDestructProcT mDestructor;
    void *mDestructorArg;
    aChunk *mCurChunk;
    T *mCurChunkPtr;
    size_t mCurChunkSize;
    size_t mCurChunkOffset;
    size_t mCurChunkLeftSize;
    size_t mTotalSize;
    bool mIsDeleting;
    bool mIsStopping;


    inline size_t
    pUpdateTotalSize(size_t d) {
        ScopedLock l(&mTotalSizeLock);
        mTotalSize += d;
        mTotalSizeWait.wakeAll();
        return mTotalSize;
    }


    inline bool
    pWaitUntilTotalSizeGreaterEqual(size_t &desiredSize,
                                    int64_t timedOut = -1) {
        bool ret = false;
        ScopedLock l(&mTotalSizeLock);

        if (mIsStopping == true) {
            return false;
        }

        int aCace = (timedOut > 0) ? 1 :
            ((timedOut == 0) ? 0 : -1);

        switch (aCace) {

            case 1: {
                //
                // Wait specified time.
                //
                uint64_t waitStart;
                int64_t rest = timedOut;

                if (mTotalSize < desiredSize) {

                    ReWait:
                    if (mIsStopping == false) {
                        waitStart = NanoSecond::getCurrentTimeInNanos();
                        if (mTotalSizeWait.timedwait(&mTotalSizeLock,
                                                     rest) == true) {
                            if (mTotalSize < desiredSize) {
                                //
                                // Not yet.
                                //
                                rest -= 
                                    ((NanoSecond::getCurrentTimeInNanos() - 
                                      waitStart) * 1000LL);
                                if (rest > 0 && mIsStopping == false) {
                                    goto ReWait;
                                }
                            } else {
                                //
                                // Got desired number of data.
                                //
                                ret = true;
                            }
                        }
                        // else we just miss the train or the train
                        // never comes. What a shame.
                    }

                } else {
                    //
                    // We already have enuff data.
                    //
                    ret = true;
                }

                break;
            }

            case 0: {
                //
                // One shot polling.
                //
                if (mTotalSize >= desiredSize) {
                    ret = true;
                }
                break;
            }

            case -1: {
                //
                // Wait forever.
                //
                ReCheck:
                if (mTotalSize < desiredSize && mIsStopping == false) {
                    mTotalSizeWait.wait(&mTotalSizeLock);
                    goto ReCheck;
                } else {
                    ret = true;
                }

                break;
            }

        }

        if (mTotalSize < desiredSize) {
            desiredSize = mTotalSize;
        }

        return ret;
    }


    inline size_t
    pTotalSize(void) {
        ScopedLock l(&mTotalSizeLock);
        return mTotalSize;
    }


    inline void
    pUpdateCurChunkInfo(size_t d) {
        mCurChunkOffset += d;
        mCurChunkLeftSize -= d;;
        pUpdateTotalSize(-d);
        if (mCurChunkLeftSize == 0) {
            // we'd better free chunks up eagerly, so do it very
            // here.
            delete mCurChunk;
            mCurChunk = NULL;
            mCurChunkSize = 0;
            mCurChunkLeftSize = 0;
        }
    }


    inline bool
    pGetNextChunk(int64_t timedOut = -1,
                  BlockingContainer::ContainerStatus *sPtr = NULL) {
        aChunk *newP;
        bool ret = false;

        if (mCurChunk != NULL) {
            delete mCurChunk;
            mCurChunk = NULL;
        }
        if ((ret = mQPtr->get(newP, timedOut, sPtr)) == true) {
            mCurChunk = newP;
            mCurChunkPtr = mCurChunk->data();
            mCurChunkSize = mCurChunk->size();
            mCurChunkLeftSize = mCurChunkSize;
            ret = true;
        } else {
            mCurChunkPtr = NULL;
            mCurChunkSize = 0;
            mCurChunkLeftSize = 0;
        }
        mCurChunkOffset = 0;

        return ret;
    }


    inline bool
    pPut(T *ptr,
         size_t bufSize,
         int64_t timedOut = -1,
         BlockingContainer::ContainerStatus *sPtr = NULL) {

        //
        // Note that it is not necessary to lock the mLock here since
        // we only need to update the total buffer size and it is done
        // by pUpdateTotalSize() that has own lock.
        //

        T *newPtr = ptr;
        if (mCreator != NULL) {
            newPtr = (mCreator)(bufSize, mCreatorArg);
            if (newPtr != NULL) {
                (void)memcpy((void *)newPtr, 
                             (void *)ptr,
                             bufSize * sizeof(T));
            } else {
                if (sPtr != NULL) {
                    *sPtr = BlockingContainer::Status_Any_Failure;
                }
                return false;
            }
        }
        aChunk *ap = new aChunk(newPtr, bufSize, mDestructor, mDestructorArg);
        bool ret = mQPtr->put(ap, timedOut, sPtr);
        if (ret == true) {
            pUpdateTotalSize(bufSize);
        }

        return ret;
    }


    inline bool
    pGet(T &data,
         int64_t timedOut = -1,
         BlockingContainer::ContainerStatus *sPtr = NULL) {
        bool ret = false;

        // Note:
        //	The logic below is kinda misundestandable but it is
        //	for making branch prediction unit exploit its
        //	effciency maximumlly.

        //
        // We need the lock bety here.
        //
        ScopedLock l(&mLock);

        if (mCurChunk != NULL) {
            if (mCurChunkOffset < mCurChunkSize) {
                // the data is stil available in the current chunk.
                GotAChunk:
                data = mCurChunkPtr[mCurChunkOffset];
                pUpdateCurChunkInfo(1);
                ret = true;
            } else {
                // need a new chunk.
                GetNextChunk:
                if ((ret = pGetNextChunk(timedOut, sPtr)) == true) {
                    // got a new chunk.
                    goto GotAChunk;
                } else {
                    // no new chunk was acquired, means we are
                    // shutting down or timedout.
                    data = 0;
                }
            }
        } else {
            goto GetNextChunk;
        }

        return ret;
    }


    inline bool
    pGet(T *dstBuf,
         size_t &dstBufSize,
         int64_t timedOut = -1,
         BlockingContainer::ContainerStatus *sPtr = NULL) {
        bool ret = false;
        bool isTimedout = false;

        size_t desiredSize = dstBufSize;
        size_t dTmp = dstBufSize;
#if 0
        size_t availableSize;
#endif
        int64_t fetchTimedOut;

        if (pWaitUntilTotalSizeGreaterEqual(dTmp, timedOut) == false) {
            if (mIsStopping == true) {
                if (sPtr != NULL) {
                    *sPtr =
                        BlockingContainer::Status_Container_No_Longer_Valid;
                }
                return false;
            }

            if (timedOut == -1) {

                if (sPtr != NULL) {
                    *sPtr = BlockingContainer::Status_Any_Failure;
                }
                return false;

            } else if (timedOut > 0) {
#if 0
                dbgMsg("after wait %f msec.: desired %u, available %u.\n",
                       (double)timedOut / 1000.0,
                       desiredSize, dTmp);
#endif
                if (dTmp == 0) {

                    dstBufSize = 0;
                    if (sPtr != NULL) {
                        *sPtr = BlockingContainer::Status_Timedout;
                    }
                    return false;

                } else if (dTmp < desiredSize) {
                    isTimedout = true;
                }

            }
        }

        //
        // We need the lock bety here.
        //
        ScopedLock l(&mLock);

        fetchTimedOut = (timedOut > 0) ? 0 : timedOut;

        if (mCurChunk != NULL) {
            GotAChunk:
            if (desiredSize <= mCurChunkLeftSize) {
                // The easiest case.
                (void)memcpy((void *)dstBuf,
                             (void *)&(mCurChunkPtr[mCurChunkOffset]),
                             desiredSize * sizeof(T));
                pUpdateCurChunkInfo(desiredSize);
                ret = true;
            } else {
                size_t maxCopySize = pTotalSize();
#if 0
                availableSize = maxCopySize;
#endif
                maxCopySize = (desiredSize > maxCopySize) ? 
                    maxCopySize : desiredSize;

                //
                // Final timedout check. There could be a chance that
                // the total size is increased.
                //
                if (timedOut > 0) {
#if 0
                    dbgMsg("in loop: desired %u > available %u.\n",
                           desiredSize, availableSize);
#endif
                    if (maxCopySize == desiredSize) {
                        isTimedout = false;
                    } else {
                        isTimedout = true;
                    }
                }

                size_t cpSize = 0;
                size_t copiedSize = 0;

                while (copiedSize < maxCopySize && 
                       mIsStopping == false) {
                    if (mCurChunkLeftSize > 0) {
                        if ((copiedSize + mCurChunkLeftSize) < maxCopySize) {
                            cpSize = mCurChunkLeftSize;
                        } else {
                            // must be the last time.
                            cpSize = maxCopySize - copiedSize;
                        }

                        (void)memcpy((void *)&(dstBuf[copiedSize]),
                                     (void *)&(mCurChunkPtr[mCurChunkOffset]),
                                     cpSize * sizeof(T));
                        pUpdateCurChunkInfo(cpSize);
                        copiedSize += cpSize;
                    } else {
                        if (pGetNextChunk(fetchTimedOut, sPtr) == true) {
                            continue;
                        } else {
                            break;
                        }
                    }
                }
                dstBufSize = copiedSize;
                if (mIsStopping == false) {
                    if (copiedSize == desiredSize) {
                        ret = true;
                    }
                }
#if 0
                dbgMsg("desiredSize %u, dstBufSize: %u, maxCopySize: %u, "
                       "copiedSize %u, "
                       "ret = %s, timedout = %s\n",
                       desiredSize, dstBufSize, maxCopySize, copiedSize,
                       booltostr(ret), booltostr(isTimedout));
#endif
            }
        } else {
            if (pGetNextChunk(fetchTimedOut, sPtr) == true) {
                goto GotAChunk;
            }
        }

        if (sPtr != NULL) {
            if (ret == false) {
                if (mIsStopping == true) {
                    *sPtr = 
                        BlockingContainer::Status_Container_No_Longer_Valid;
                } else if (isTimedout == true) {
                    *sPtr = BlockingContainer::Status_Timedout;
                } else {
                    *sPtr = BlockingContainer::Status_Any_Failure;
                }
            } else {
                *sPtr = BlockingContainer::Status_OK;
            }
        }

        return ret;
    }

    inline bool
    pClear(void) {
        ScopedLock lock(&mLock);

        {
            ScopedLock totalLock(&mTotalSizeLock);
            mTotalSize = 0;
            mTotalSizeWait.wakeAll();
        }

        mQPtr->clear();

        mCurChunkOffset = 0;
        mCurChunkLeftSize = 0;
        if (mCurChunk != NULL) {
            delete mCurChunk;
            mCurChunk = NULL;
        }
        mCurChunkSize = 0;

        return true;
    }

    inline bool
    pStop(void) {
        if (mIsStopping == false) {
            mIsStopping = true;

            mTotalSizeLock.lock();
            mTotalSizeWait.wakeAll();
            mTotalSizeLock.unlock();

            if (mQPtr != NULL) {
                mQPtr->stop();
            }
            if (mCurChunk != NULL) {
                delete mCurChunk;
                mCurChunk = NULL;
            }
        }
        return mIsStopping;
    }


public:
    ChunkedBuffer(size_t maxLen = BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH) :
        // mLock
        // mTotalSizeLock,
        // mTotalSizeWait,
        mQPtr(new aChunkQueue(maxLen)),
        mCreator(NULL),
        mCreatorArg(NULL),
        mDestructor(NULL),
        mDestructorArg(NULL),
        mCurChunk(NULL),
        mCurChunkPtr((T *)NULL),
        mCurChunkSize(0),
        mCurChunkOffset(0),
        mCurChunkLeftSize(0),
        mTotalSize(0),
        mIsDeleting(false),
        mIsStopping(false) {
    }


    ~ChunkedBuffer(void) {
        ScopedLock l(&mLock);
        pStop();
        if (mIsDeleting == false) {
            mIsDeleting = true;
            if (mQPtr != NULL) {
                delete mQPtr;
            }
        }
    }


private:
    ChunkedBuffer(const ChunkedBuffer &obj);
    ChunkedBuffer operator = (const ChunkedBuffer &obj);


protected:
    inline void
    setDestructionHook(__ChunkDestructProcT proc, void *arg) {
        ScopedLock l(&mLock);
        mDestructor = proc;
        mDestructorArg = arg;
    }


    inline void
    setAllocationHook(__ChunkAllocateProcT proc, void *arg) {
        ScopedLock l(&mLock);
        mCreator = proc;
        mCreatorArg = arg;
    }


public:
    inline bool
    stop(void) {
        ScopedLock l(&mLock);
        return pStop();
    }


    inline bool
    put(T *buf,
        size_t bufSize,
        int64_t timedOut = -1,
        BlockingContainer::ContainerStatus *sPtr = NULL) {
        return pPut(buf, bufSize, timedOut, sPtr);
    }


    inline bool
    get(T &data,
        int64_t timedOut = -1,
        BlockingContainer::ContainerStatus *sPtr = NULL) {
        return pGet(data, timedOut, sPtr);
    }


    inline bool
    get(T *dstBuf,
        size_t &dstBufSize,
        int64_t timedOut = -1,
        BlockingContainer::ContainerStatus *sPtr = NULL) {
        return pGet(dstBuf, dstBufSize, timedOut, sPtr);
    }


    inline size_t
    size(void) {
        return pTotalSize();
    }

    inline bool
    clear(void) {
        return pClear();
    }

    inline size_t
    elementSize(void) {
        return sizeof(T);
    }
};


#endif // ! __CHUNKEDBUFFER_H__
