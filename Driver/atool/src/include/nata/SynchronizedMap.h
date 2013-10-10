/*
 * $Id: SynchronizedMap.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __SYNCHRONIZEDMAP_H__
#define __SYNCHRONIZEDMAP_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>

#include <map>





template <typename __KeyT, typename __ValT>
class SynchronizedMap {



private:
    __rcsId("$Id: SynchronizedMap.h 86 2012-07-30 05:33:07Z m-hirano $");

    typedef typename std::map<__KeyT, __ValT> __theMap;
    typedef typename std::map<__KeyT, __ValT>::iterator __theMapIterator;
    typedef typename std::pair<__theMapIterator, bool> __theMapInsertionResult;
    typedef typename std::pair<__KeyT, __ValT> __theMapInsertionDatum;

    typedef void (*__ValRemoveProcT)(__ValT ptr, void *arg);
    typedef void (*__ValDestructPrccT)(__ValT ptr, void *arg);
    typedef void (*__ClearProcT)(void *arg);


    __theMap mMap;
    Mutex mLock;

    __ValRemoveProcT mRemoveHook;
    void *mRemoveArg;

    __ValDestructPrccT mDeleteHook;
    void *mDeleteArg;

    __ClearProcT mClearHook;
    void *mClearArg;

    bool mIsDeleting;





private:


    inline void
    pRemoveHook(__ValT val) {
        if (mRemoveHook != NULL) {
            (mRemoveHook)(val, mRemoveArg);
        }
    }


    inline void
    pDeleteHook(__ValT val) {
        if (mDeleteHook != NULL) {
            (mDeleteHook)(val, mDeleteArg);
        }
    }


    inline void
    pClearHook(void) {
        if (mClearHook != NULL) {
            (mClearHook)(mClearArg);
        }

        __theMapIterator it;
        __theMapIterator itEnd = end();

        for (it = begin(); it != itEnd; it++) {
            pDeleteHook(it->second);
        }
    }


    SynchronizedMap(const __theMap &obj);
    SynchronizedMap operator = (const __theMap &obj);





protected:


    inline void
    setRemoveHook(__ValRemoveProcT p, void *a) {
        ScopedLock l(&mLock);
        mRemoveHook = p;
        mRemoveArg = a;
    }


    inline void
    setDeleteHook(__ValDestructPrccT p, void *a) {
        ScopedLock l(&mLock);
        mDeleteHook = p;
        mDeleteArg = a;
    }


    inline void
    setClearHook(__ClearProcT p, void *a) {
        ScopedLock l(&mLock);
        mClearHook = p;
        mClearArg = a;
    }


    inline bool
    lock(void) {
        return mLock.lock();
    }


    inline bool
    unlock(void) {
        return mLock.unlock();
    }


    inline bool
    trylock(void) {
        return mLock.trylock();
    }


    inline __theMapIterator
    begin(void) {
        return mMap.begin();
    }


    inline __theMapIterator
    end(void) {
        return mMap.end();
    }


    inline __theMapIterator
    find(__KeyT key) {
        return mMap.find(key);
    }


    inline __theMapInsertionResult
    insert(__KeyT key, __ValT val) {
        return mMap.insert(__theMapInsertionDatum(key, val));
    }


    inline void
    erase(__theMapIterator it) {
        mMap.erase(it);
    }





    inline bool
    putNoLock(__KeyT key, __ValT val) {
        bool ret = false;
        __theMapInsertionResult r;

        InsertAgain:
        r = insert(key, val);
        if (r.second == true) {
            ret = true;
        } else {
            __theMapIterator it = find(key);
            if (it != end()) {
                erase(it);
                goto InsertAgain;
            } else {
                fatal("insertion failed but has no key.\n");
            }
        }

        return ret;
    }


    inline bool
    getNoLock(__KeyT key, __ValT &val) {
        bool ret = false;

        __theMapIterator it = find(key);
        if (it != end()) {
            ret = true;
            val = it->second;
        }

        return ret;
    }


    inline bool
    removeNoLock(__KeyT key, __ValT &val) {
        bool ret = false;

        __theMapIterator it = find(key);
        if (it != end()) {
            ret = true;
            val = it->second;
            erase(it);
            pRemoveHook(val);
        }

        return ret;
    }
        

    inline void
    clearNoLock(void) {
        pClearHook();
        mMap.clear();
    }


    inline bool
    containsKeyNoLock(__KeyT key) {
        bool ret = false;

        __theMapIterator it = find(key);
        if (it != end()) {
            ret = true;
        }

        return ret;
    }


    inline size_t
    sizeNoLock(void) {
        return mMap.size();
    }





public:
    SynchronizedMap(void) :
        // mMap,
        // mLock,
        mRemoveHook(NULL),
        mRemoveArg(NULL),
        mDeleteHook(NULL),
        mDeleteArg(NULL),
        mClearHook(NULL),
        mClearArg(NULL),
        mIsDeleting(false) {
        (void)rcsid();
    }


    ~SynchronizedMap(void) {
        ScopedLock l(&mLock);
        if (mIsDeleting == false) {
            mIsDeleting = true;
            clearNoLock();
        }
    }


    inline bool
    put(__KeyT key, __ValT val) {
        ScopedLock l(&mLock);
        return putNoLock(key, val);
    }


    inline bool
    get(__KeyT key, __ValT &val) {
        ScopedLock l(&mLock);
        return getNoLock(key, val);
    }


    inline bool
    remove(__KeyT key, __ValT &val) {
        ScopedLock l(&mLock);
        return removeNoLock(key, val);
    }
        

    inline void
    clear(void) {
        ScopedLock l(&mLock);
        clearNoLock();
    }


    inline bool
    containsKey(__KeyT key) {
        ScopedLock l(&mLock);
        return containsKeyNoLock(key);
    }


    inline size_t
    size(void) {
        ScopedLock l(&mLock);
        return sizeNoLock();
    }


    inline void
    childAfterFork(void) {
        mLock.reinitialize();
    }


};


#endif // ! __SYNCHRONIZEDMAP_H__
