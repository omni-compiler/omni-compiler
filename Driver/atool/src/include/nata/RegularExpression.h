/* 
 * $Id: RegularExpression.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __REGULAREXPRESSION_H__
#define __REGULAREXPRESSION_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>

#include <nata/nata_perror.h>


class RegularExpression {


private:
    __rcsId("$Id");


    Mutex mLock;
    const char *mExp;
    regex_t *mCompiledExp;


    RegularExpression(const RegularExpression &obj);
    RegularExpression operator = (const RegularExpression &obj);


    inline bool
    mNew(const char *exp, bool ignoreCase = false) {
        bool ret = false;
        regex_t rc;
        int flags = REG_EXTENDED | REG_NOSUB;
        int st;

        if (ignoreCase == true) {
            flags |= REG_ICASE;
        }

        st = regcomp(&rc, exp, flags);
        if (st != 0) {
            char errMsg[4096];
            (void)regerror(st, (const regex_t *)&rc,
                           errMsg, sizeof(errMsg));
            nata_MsgError("regcomp(\"%s\") error: %s\n",
                          exp, errMsg);
            goto Done;
        }

        mCompiledExp = (regex_t *)malloc(sizeof(rc));
        if (mCompiledExp != NULL) {
            (void)memcpy((void *)mCompiledExp, (void *)&rc, sizeof(rc));
            mExp = strdup(exp);
            ret = true;
        }

        Done:
        return ret;
    }


public:
    RegularExpression(const char *exp, bool ignoreCase = false) :
        // mLock,
        mExp(NULL),
        mCompiledExp(NULL) {
        if (mNew(exp, ignoreCase) == false) {
            fatal("can't initialize a regexp.");
        }
    }


    ~RegularExpression(void) {
        (void)free((void *)mExp);
        regfree(mCompiledExp);
    }


    inline bool
    match(const char *str) {
        ScopedLock l(&mLock);
        return (regexec(mCompiledExp, str, 0, NULL, 0) == 0) ? true : false;
    }


    inline const char *
    getExpression(void) {
        return mExp;
    }


};

#endif // __REGULAREXPRESSION_H__
