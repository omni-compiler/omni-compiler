/*
 * $Id: SynchronizedList.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __SYNCHRONIZEDLIST_H__
#define __SYNCHRONIZEDLIST_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>

#include <map>





template <typename __ValT>
class SynchronizedList {



private:
    __rcsId("$Id: SynchronizedList.h 86 2012-07-30 05:33:07Z m-hirano $");

    typedef typename std::list<__ValT> __theList;
    typedef typename std::list<__ValT>::iterator __theListIterator;

    typedef void (*__ValRemoveProcT)(__ValT ptr, void *arg);
    typedef void (*__ValDestructPrccT)(__ValT ptr, void *arg);
    typedef void (*__ClearProcT)(void *arg);


    __theList mList;
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

        __theListIterator it;
        __theListIterator itEnd = end();

        for (it = begin(); it != itEnd; it++) {
            pDeleteHook(*it);
        }
    }


    SynchronizedList(const __theList &obj);
    SynchronizedList operator = (const __theList &obj);





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


    inline void
    lock(void) {
        mLock.lock();
    }


    inline void
    unlock(void) {
        mLock.unlock();
    }


    inline __theListIterator
    begin(void) {
        return mList.begin();
    }


    inline __theListIterator
    end(void) {
        return mList.end();
    }


    inline __theListIterator
    insert(__theListIterator it, const __ValT &val) {
        return mList.insert(it, val);
    }


    inline void
    push_back(const __ValT &val) {
        mList.push_back(val);
    }


    inline __theListIterator
    erase(__theListIterator it) {
        return mList.erase(it);
    }


    inline void
    remove(const __ValT &val) {
        mList.remove(val);
    }


    inline size_t
    size(void) {
        return mList.size();
    }





    inline bool
    addNoLock(__ValT val) {
        bool ret = false;
        push_back(val):
        return ret;
    }


    inline bool
    putNoLock(size_t pos, __ValT val) {
        size_t sz = size();
        if (pos < sz) {
            __theListIterator it = begin();
            it += pos;
            insert(it, val);
            return true;
        } else if (pos >= sz) {
            return addNoLock(val);
        }
    }


    inline bool
    getNoLock(size_t pos, __ValT &val) {
        bool ret = false;
        size_t sz = size();
        __theMapIterator it = begin();

        if (pos < sz) {
            it += pos;
            val = *it;
            ret = true;
        }

        return ret;
    }


    inline bool
    eraseNoLock(size_t pos) {
        bool ret = false;
        size_t sz = size();
        __theMapIterator it = begin();

        if (pos < sz) {
            it += pos;
            erase(it);
            ret = true;
        }

        return ret;
    }


    inline bool
    removeNoLock(__ValT val) {
        pRemoveHook(val);
        remove(val);
        return true;
    }


    inline void
    clearNoLock(void) {
        pClearHook();
        mMap.clear();
    }


    inline bool
    contains(__ValT val) {
        bool ret = false;

        __theMapIterator it = find(key);
        if (it != end()) {
            ret = true;
        }

        return ret;
    }


    inline size_t
    sizeNoLock(void) {
        return size();
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


};


#endif // ! __SYNCHRONIZEDLIST_H__
