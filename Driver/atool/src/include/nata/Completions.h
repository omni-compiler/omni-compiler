/* 
 * $Id: Completions.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __COMPLETIONS_H__
#define __COMPLETIONS_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Completion.h>
#include <nata/ScopedLock.h>

#include <nata/nata_perror.h>





class Completions {


private:
    __rcsId("$Id: Completions.h 86 2012-07-30 05:33:07Z m-hirano $");

    typedef std::vector<Completion *> __theVector;

    __theVector mV;
    Mutex mLock;


public:

    Completions(void) {
    }


    ~Completions(void) {
        mV.clear();
    }


private:
    Completions(const Completions &obj);
    Completions operator = (const Completions &obj);



public:


    inline void
    add(Completion *c) {
        ScopedLock l(&mLock);
        mV.push_back(c);
    }


    inline size_t
    size(void) {
        ScopedLock l(&mLock);
        return mV.size();
    }
    

    inline void
    wake(void) {
        ScopedLock l(&mLock);
        Completion *cPtr = NULL;
        for (__theVector::iterator i = mV.begin();
             i != mV.end();
             i++) {
            cPtr = *i;
            if (cPtr != NULL) {
                cPtr->wake();
            }
        }
    }


    inline void
    wakeAll(void) {
        ScopedLock l(&mLock);
        Completion *cPtr = NULL;
        for (__theVector::iterator i = mV.begin();
             i != mV.end();
             i++) {
            cPtr = *i;
            if (cPtr != NULL) {
                cPtr->wakeAll();
            }
        }
    }

};



#endif // ! __COMPLETIONS_H__
