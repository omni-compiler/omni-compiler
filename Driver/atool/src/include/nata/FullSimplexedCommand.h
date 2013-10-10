#ifndef __FULLSIMPLEXEDCOMMAND_H__
#define __FULLSIMPLEXEDCOMMAND_H__

#include <nata/nata_rcsid.h>

#include <nata/nata_includes.h>

#include <nata/nata_macros.h>

#include <nata/FullSimplexedProcess.h>

#include <nata/nata_perror.h>





class FullSimplexedCommand: public FullSimplexedProcess {





private:


    __rcsId("$Id: FullSimplexedCommand.h 135 2012-08-14 11:02:37Z m-hirano $");





private:


    const char *mPath;
    char * const *mArgv;


    FullSimplexedCommand(const FullSimplexedCommand &obj);
    FullSimplexedCommand operator = (const FullSimplexedCommand &obj);


    int
    run(void) {
        int i;
        int n = nata_getMaxFileNo();

        for (i = 3; i < n; i++) {
            (void)::close(i);
        }

        // nata_MsgInfo("'%s' start.\n", mPath);

        ::execvp(mPath, mArgv);
        perror("execvp");
        ::_exit(1);
        // not reached.
        return 1;
    }


public:


    FullSimplexedCommand(const char *pwd, 
                         const char *path, char * const *argv) :
        FullSimplexedProcess(pwd),
        mPath(NULL),
        mArgv(NULL) {
        mPath = path;
        mArgv = argv;
    }


    inline char * const *
    getArguments(void) {
        return mArgv;
    }


};


#endif // __FULLSIMPLEXEDCOMMAND_H__
