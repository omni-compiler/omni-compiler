/*
 * $Id: PipelineBuilder.h 235 2013-09-06 03:42:51Z m-hirano $
 */
#ifndef __PIPELINEBUILER_H__
#define __PIPELINEBUILER_H__

#include <nata/nata_rcsid.h>

#include <nata/PipelinedCommands.h>

#include "CompileOptions.h"
#include "OutputFileProvider.h"
#include "paths.h"

#include <nata/nata_perror.h>





class PipelineBuilder {



private:


    __rcsId("$Id: PipelineBuilder.h 235 2013-09-06 03:42:51Z m-hirano $")





    CompileOptions *mCOPtr;
    OutputFileProvider *mOFPPtr;

    const char *mCppExe;
    const char *mFrontendExe;
    const char *mTranslatorExe;
    const char *mDecompilerExe;
    const char *mNativeExe;
    const char *mPrelinkerExe;

    const char *mOmniIncDirCppOpt;

    bool mIsCppTheCpp;	// not like "gcc", "gfortran", etc.
    bool mIsCppGccSuit;
    bool mHasNoTranslator;

    CompileOptions::CompileStageT mStage;
    bool mIsDryrun;
    bool mIsVerbose;
    const char *mOutputFile;
    const char *mPrelinkerOutputFile;

    PipelineBuilder(const PipelineBuilder &obj);
    PipelineBuilder operator = (const PipelineBuilder &obj);





#if 0
    inline void
    mDumpArgs(const char *tags, char * const argv[]) {
        fprintf(stderr, "%s", tags);

        int i = 0;
        while (*argv != NULL) {
            fprintf(stderr, "%3d:\t%s\n", i, *argv);
            argv++;
            i++;
        }
    }
#endif


    inline const char *
    mGetArgv0(const char *path) {
        const char *tmp = strrchr(path, '/');
        if (tmp == NULL) {
            return path;
        } else if (*(tmp + 1) != '\0') {
            return (const char *)(tmp + 1);
        } else {
            return NULL;
        }
    }


    inline bool
    mCheckCpp(void) {
#if 0
        if (mIsDryrun == true) {
            return true;
        }
#endif

        bool ret = false;
        const char *cppArgv0 = mGetArgv0(mCppExe);

        //
        // Check whether the mCppExe is a gcc suit.
        // 
        const char * const cpp[] = {
            cppArgv0,
            "--version",
            NULL
        };
        const char * const grep[] = {
            "grep",
            "Free Software Foundation, Inc.",
            NULL
        };

        PipelinedCommands chkCmds;
        chkCmds.addCommand(NULL, mCppExe, (char * const *)cpp);
        chkCmds.addCommand(NULL, (const char *)grep[0], (char * const *)grep);

        chkCmds.start(false, true);

        int n = 0;
        FILE *cmdOut = chkCmds.getOutputFILE();
        char buf[4096];

        while (fgets(buf, sizeof(buf), cmdOut) != NULL) {
            n++;
        }
        (void)::fclose(cmdOut);
        (void)chkCmds.wait();

        if (n > 0) {
            mIsCppGccSuit = true;
        }

        //
        // Check whether the mCppExe is "really" cpp.
        //
        // An assumption:
        //	When issueing the mCppEx w/o any arguments:
        //		a) The mCppExe get mad if it is not really a cpp.
        //		b) The other hand, the mCppExe waits data from
        //		   stdin if it is really a cpp.
        //
        const char * const cpp2[] = {
            cppArgv0,
            NULL
        };
        PipelinedCommands chk2;
        chk2.addCommand(NULL, mCppExe, (char * const *)cpp2);

        chk2.start(false, true);

        nata_Uid testData;
        FILE *cmd2In = chk2.getInputFILE();
        FILE *cmd2Out = chk2.getOutputFILE();
        bool gotTheData = false;

        (void)nata_getUid(&testData);
        size_t testDataLen = strlen((char *)testData);

        (void)fprintf(cmd2In, "%s", (char *)testData);
        (void)::fclose(cmd2In);
        while (fgets(buf, sizeof(buf), cmd2Out) != NULL) {
            if (strncmp(buf, (char *)testData, testDataLen) == 0) {
                gotTheData = true;
            }
        }
        (void)::fclose(cmd2Out);
        (void)chk2.wait();

        mIsCppTheCpp = gotTheData;

        //
        // Finally, check whether the mCppExe with -E option acts like
        // a cpp.
        //
        char filename[PATH_MAX];
        snprintf(filename, sizeof(filename), "/tmp/.%s.f",
                 (char *)testData);
        FILE *fd = ::fopen(filename, "w+");
        if (fd == NULL) {
            perror("fopen");
            fatal("Can't create a temporary file '%s'.",
                  filename);
            // not reached.
            return false;
        }
        fprintf(fd, "#ifdef __XMP_CPP_TEST__\n%s\n#endif\n",
                (char *)testData);
        (void)::fclose(fd);

        size_t idx = 0;
        char const *cpp3[6];
        cpp3[idx++] = cppArgv0;
        if (mIsCppTheCpp == false) {
            cpp3[idx++] = "-E";
            if (mIsCppGccSuit == true) {
                cpp3[idx++] = "-cpp";
            }
        }
        cpp3[idx++] = "-D__XMP_CPP_TEST__";
        cpp3[idx++] = filename;
        cpp3[idx] = NULL;

        PipelinedCommands chk3;
        chk3.addCommand(NULL, mCppExe, (char * const *)cpp3);

        chk3.start(false, true);

        gotTheData = false;
        FILE *cmd3Out = chk3.getOutputFILE();
        while (fgets(buf, sizeof(buf), cmd3Out) != NULL) {
            if (strncmp(buf, (char *)testData, testDataLen) == 0) {
                gotTheData = true;
            }
        }
        (void)::fclose(cmd3Out);
        (void)::unlink(filename);
        if (mIsCppGccSuit == true) {
            //
            // "{gcc|gfortran} -cpp -E x.f" always generate x.s,
            // so we have to unlink it too.
            //
            snprintf(buf, sizeof(buf), "/tmp/.%s.s",
                     (char *)testData);
            (void)::unlink(buf);
        }

        if (gotTheData == true) {
            ret = true;
            // fprintf(stderr, "%s works as a cpp.\n", mCppExe);
        }
#if 0
        if (mIsCppGccSuit == true) {
            fprintf(stderr, "%s is a gcc suit.\n", mCppExe);
        }
        if (mIsCppTheCpp == true) {
            fprintf(stderr, "%s is really a cpp.\n", mCppExe);
        }
#endif

        return ret;
    }


    inline size_t
    mGenCppOpts(char **&opts, CompileOptions::CompileStageT stage) {
        size_t ret = 0;
        opts = NULL;

        switch (stage) {

            case CompileOptions::Stage_CPP: {
                //
                // Generate full preprocessor option needed.
                //
                std::vector<char *> vExeOpts;

                if (mIsCppTheCpp == false) {
                    vExeOpts.push_back((char *)"-E");
                    if (mIsCppGccSuit == true) {
                        vExeOpts.push_back((char *)"-cpp");
                    }
                }
#if 0
                if (mIsCppGccSuit == false) {
                    vExeOpts.push_back((char *)mOmniIncDirCppOpt);
                } else {
                    vExeOpts.push_back((char *)"-iprefix");
                    vExeOpts.push_back((char *)"/");
                    vExeOpts.push_back((char *)"-iwithprefix");
                    vExeOpts.push_back((char *)OMNI_INCDIR);
                }
#else
                vExeOpts.push_back((char *)mOmniIncDirCppOpt);
#endif

                size_t i;
                char **cmdlineOpts = NULL;
                size_t nCmdlineOpts =
                    mCOPtr->getOptions(cmdlineOpts, 0, 0,
                                       CompileOptions::Options_CPP);
                nCmdlineOpts--;
                size_t nExeOpts = vExeOpts.size();
                size_t nAll = nExeOpts + nCmdlineOpts;
                char **args = (char **)malloc(sizeof(char *) * nAll);
                if (args != NULL) {
                    for (i = 0; i < nExeOpts; i++) {
                        args[i] = vExeOpts[i];
                    }
                    for (i = 0; i < nCmdlineOpts; i++) {
                        args[i + nExeOpts] = cmdlineOpts[i];
                    }
                    ret = nAll;
                    opts = args;
                }
                free(cmdlineOpts);

                break;
            }

            case CompileOptions::Stage_Frontend: {
                //
                // Needs only "-I"s for module/include file search.
                //
                std::vector<char *> vOpts;
                char **cmdlineOpts = NULL;
                size_t nCmdlineOpts =
                    mCOPtr->getOptions(cmdlineOpts, 0, 0,
                                       CompileOptions::Options_CPP);
                nCmdlineOpts--;
                size_t i;

                for (i = 0; i < nCmdlineOpts; i++) {
                    if (strncmp("-I", cmdlineOpts[i], 2) == 0) {
                        vOpts.push_back(cmdlineOpts[i]);
                    }
                }
                vOpts.push_back((char *)mOmniIncDirCppOpt);

                size_t n = vOpts.size();
                char **args = (char **)malloc(sizeof(char *) * n);
                if (args != NULL) {
                    for (i = 0; i < n; i++) {
                        args[i] = vOpts[i];
                    }
                    ret = n;
                    opts = args;
                }
                free(cmdlineOpts);

                break;
            }

            default: {
                break;
            }
        }

        return ret;
    }


    inline const char *
    mTeeFile(const char *sourceFile, bool reallyNeeded,
             CompileOptions::CompileStageT stage) {
        const char *ret = NULL;
        if (reallyNeeded == true) {
            const char *outFile = mCOPtr->getOutputFile();
            const char *sfx = getOutputFileSuffix(false, stage);
            const char *retTeeFile =
                CompileOptions::getDefaultOutputFilename(sourceFile, sfx);
            if (((mStage == stage || mStage == CompileOptions::Stage_CPP) &&
                 (isValidString(outFile) == false ||
                  strcmp(outFile, retTeeFile) != 0)) ||
                (mStage != stage)) {
                ret = retTeeFile;
            } else {
                free((void *)retTeeFile);
            }
        }
        return ret;
    }


    inline int
    mRunCpp(const char *inputFile) {
        bool execSucceeded = false;
        int ret = -INT_MAX;
        size_t i;
        PipelinedCommands *pPtr = new PipelinedCommands(-1, -1, 2);
        const char *sfx = strrchr(inputFile, '.');
        const char *teeFile = NULL;
        char **cppOpts = NULL;
        size_t nCppOpts = mGenCppOpts(cppOpts, CompileOptions::Stage_CPP);

        size_t nCppArgs = nCppOpts + 3;	// one for argv0 and one for
                                        // an input file, and one for
                                        // a terminator for the execv(2).

        char **cppArgs = (char **)malloc(sizeof(char *) * nCppArgs);

        if (sfx == NULL) {
            sfx = ".f";
        }

        cppArgs[0] = (char *)mGetArgv0(mCppExe);
        for (i = 0; i < nCppOpts; i++) {
            cppArgs[i + 1] = cppOpts[i];
        }
        free((void *)cppOpts);

        cppArgs[nCppArgs - 2] = (char *)inputFile;
        cppArgs[nCppArgs - 1] = NULL;

        teeFile = mTeeFile(inputFile, mCOPtr->emitCppResult(),
                           CompileOptions::Stage_CPP);
        pPtr->addCommand(NULL, mCppExe, (char * const *)cppArgs,
                         NULL, teeFile, NULL);
        free((void *)cppArgs);
        free((void *)teeFile);

        if (mStage == CompileOptions::Stage_CPP) {
            if (isValidString(mOutputFile) == true) {
                //
                // use the specified file as th final target.
                //
                ret = mOFPPtr->open(mOutputFile, false);
            } else {
                //
                // use stdout.
                //
                ret = 1;
            }
        } else {
            //
            // use default temporary file.
            //
            ret = mOFPPtr->openTemp("/tmp/.", sfx);
        }

        pPtr->setOutFd(ret);

        if (mIsVerbose == true) {
            pPtr->printCommands(stderr);
        }
        if (mIsDryrun == false) {
            execSucceeded = pPtr->start(true, true);
        } else {
            execSucceeded = true;
        }

        if (execSucceeded == false) {
            if (mCOPtr->needCleanup() == true) {
                mOFPPtr->destroy(ret);
                ret = -INT_MAX;
            }
        }

        delete pPtr;

        return ret;
    }


public:


    PipelineBuilder(CompileOptions *coPtr,
                    OutputFileProvider *ofpPtr,
                    const char *cppExe = NULL,
                    const char *frontendExe = NULL,
                    const char *translatorExe = NULL,
                    const char *decompilerExe = NULL,
                    const char *nativeExe = NULL,
                    const char *prelinkerExe = NULL) :

        mCOPtr(coPtr),
        mOFPPtr(ofpPtr),
        mCppExe(cppExe),
        mFrontendExe(frontendExe),
        mTranslatorExe(translatorExe),
        mDecompilerExe(decompilerExe),
        mNativeExe(nativeExe),
        mPrelinkerExe(prelinkerExe),
        mOmniIncDirCppOpt(NULL),
        mIsCppTheCpp(false),
        mIsCppGccSuit(false),
        mHasNoTranslator(false),
        mStage(CompileOptions::Stage_Unknown),
        mIsDryrun(false),
        mIsVerbose(false),
        mOutputFile(NULL),
        mPrelinkerOutputFile(NULL) {

        if (mCOPtr == NULL) {
            fatal("invalid CompileOptions instance.\n");
        }

        if (mOFPPtr == NULL) {
            fatal("invalid OutputFileProvider instance.\n");
        }

        if (isValidString(mCppExe) == false) {
            mCppExe = strdup(DEFAULT_CPP);
        } else {
            mCppExe = strdup(cppExe);
        }

        if (isValidString(mFrontendExe) == false) {
            mFrontendExe = strdup(DEFAULT_FRONTEND);
        } else {
            mFrontendExe = strdup(frontendExe);
        }

        if (isValidString(mTranslatorExe) == false) {
            mHasNoTranslator = true;
        } else {
            mTranslatorExe = strdup(translatorExe);
        }

        if (isValidString(mDecompilerExe) == false) {
            mDecompilerExe = strdup(DEFAULT_DECOMPILER);
        } else {
            mDecompilerExe = strdup(decompilerExe);
        }

        if (isValidString(mNativeExe) == false) {
            mNativeExe = strdup(DEFAULT_NATIVE);
        } else {
            mNativeExe = strdup(nativeExe);
        }

        if (isValidString(mPrelinkerExe) == false) {
            mPrelinkerExe = strdup(DEFAULT_PRELINKER);
        } else {
            mPrelinkerExe = strdup(prelinkerExe);
        }

        mStage = mCOPtr->getStage();
        mIsDryrun = mCOPtr->isDryrun();
        mIsVerbose = mCOPtr->isVerbose();
        mOutputFile = mCOPtr->getOutputFile();

        if (mCheckCpp() == false) {
            fatal("%s can't be used as a cpp.", mCppExe);
        }

        char tmp[PATH_MAX];
        snprintf(tmp, sizeof(tmp), "-I%s", OMNI_INCDIR);
        mOmniIncDirCppOpt = strdup(tmp);        
    }


    ~PipelineBuilder(void) {
        free((void *)mCppExe);
        free((void *)mFrontendExe);
        free((void *)mTranslatorExe);
        free((void *)mDecompilerExe);
        free((void *)mNativeExe);
        free((void *)mPrelinkerExe);

        free((void *)mPrelinkerOutputFile);

        free((void *)mOmniIncDirCppOpt);
    }


    inline PipelinedCommands *
    getXmpPipeline(const char *inputFile, int &cppTmpFd) {
        PipelinedCommands *ret = new PipelinedCommands(-1, -1, 2);
        bool withCpp = mCOPtr->isCppNeeded(inputFile);
        const char *teeFile = NULL;
        size_t i;
        int cppFd = -INT_MAX;
        const char *cppOutputFile = NULL;

        cppTmpFd = -INT_MAX;

        //
        // cpp (if needed)
        //
        if (withCpp == true) {
            cppFd = mRunCpp(inputFile);
            if (cppFd > 0) {
                if (cppFd != 1) {
                    cppOutputFile = mOFPPtr->getPath(cppFd);
                    cppTmpFd = cppFd;
                }
            }
            if (mStage == CompileOptions::Stage_CPP ||
                cppFd < 1) {
                delete ret;
                return NULL;
            }
        }

        //
        // frontend
        //
        char **cppOpts = NULL;
        size_t nCppOpts = mGenCppOpts(cppOpts, CompileOptions::Stage_Frontend);
        char **frontArgs = NULL;
        size_t nFrontArgs = -INT_MAX;

        nFrontArgs =
            mCOPtr->getOptions(frontArgs,
                               1 + nCppOpts,
                               1,
                               CompileOptions::Options_Frontend);

        frontArgs[0] = (char *)mGetArgv0(mFrontendExe);

        for (i = 0; i < nCppOpts; i++) {
            frontArgs[i + 1] = cppOpts[i];
        }
        free((void *)cppOpts);
        frontArgs[nFrontArgs - 2] = 
            (isValidString(cppOutputFile) == true) ?
            (char *)cppOutputFile : (char *)inputFile;

        teeFile = mTeeFile(inputFile, mCOPtr->emitFrontendResult(),
                           CompileOptions::Stage_Frontend);
        // mDumpArgs("frontend:\n", frontArgs);
        ret->addCommand(NULL, mFrontendExe, (char * const *)frontArgs,
                        NULL, teeFile, NULL);
        free((void *)frontArgs);
        free((void *)teeFile);
        teeFile = NULL;

        if (mStage == CompileOptions::Stage_Frontend) {
            return ret;
        }

        //
        // translator (if specified)
        //
        if (mHasNoTranslator == false) {
            char **transArgs = NULL;
            (void)mCOPtr->getOptions(
                transArgs,
                1, 0,
                CompileOptions::Options_XcodeML_Translator);
            transArgs[0] = (char *)mGetArgv0(mTranslatorExe);

            teeFile = mTeeFile(inputFile, mCOPtr->emitTranslatorResult(),
                               CompileOptions::Stage_XcodeML_Translate);
            // mDumpArgs("translator:\n", frontArgs);
            ret->addCommand(NULL, mTranslatorExe, (char * const *)transArgs,
                            NULL, teeFile, NULL);
            free((void *)transArgs);
            free((void *)teeFile);
            teeFile = NULL;
        }
        if (mStage == CompileOptions::Stage_XcodeML_Translate) {
            return ret;
        }

        //
        // decompiler
        //
        char **decompArgs = NULL;
        (void)mCOPtr->getOptions(decompArgs,
                                 1, 0,
                                 CompileOptions::Options_Decompiler);
        decompArgs[0] = (char *)mGetArgv0(mDecompilerExe);

        teeFile = mTeeFile(inputFile, mCOPtr->emitDecompilerResult(),
                           CompileOptions::Stage_Decompile);
        // mDumpArgs("decompiler:\n", decompArgs);
        ret->addCommand(NULL, mDecompilerExe, (char * const *)decompArgs,
                        NULL, teeFile, NULL);
        free((void *)decompArgs);
        free((void *)teeFile);
        teeFile = NULL;        

        return ret;
    }


    inline PipelinedCommands *
    getNativePipeline(const char *orgInputFile, const char *decompFile) {
        if ((int)mStage <= (int)CompileOptions::Stage_Decompile) {
            return NULL;
        }
        PipelinedCommands *ret = new PipelinedCommands(-1, -1, 2);

        const char *outFile = mCOPtr->getOutputFile();
        const char *defaultOutFile = NULL;

        char **nativeArgs = NULL;
        size_t nNativeArgs =
            mCOPtr->getOptionsV(nativeArgs,
                                1, 4,
                                CompileOptions::Options_CPP,
                                CompileOptions::Options_Native,
                                CompileOptions::Options_End);
        nativeArgs[0] = (char *)mGetArgv0(mNativeExe);
        const char *sfx = "";
        switch (mStage) {
            case CompileOptions::Stage_Native_Assemble: {
                nativeArgs[nNativeArgs - 5] = (char *)"-S";
                sfx = ".s";
                break;
            }
            case CompileOptions::Stage_Native_Compile:
            case CompileOptions::Stage_Native_Link: {
                nativeArgs[nNativeArgs - 5] = (char *)"-c";
                sfx = ".o";
                break;
            }

            default: {
                break;
            }
        }

        nativeArgs[nNativeArgs - 4] = (char *)decompFile;
        nativeArgs[nNativeArgs - 3] = (char *)"-o";
        if (isValidString(outFile) == true &&
            mStage != CompileOptions::Stage_Native_Link) {
            nativeArgs[nNativeArgs - 2] = (char *)outFile;
        } else {
            defaultOutFile = 
                CompileOptions::getDefaultOutputFilename(
                    orgInputFile, sfx);
            nativeArgs[nNativeArgs - 2] = (char *)defaultOutFile;
        }

        // mDumpArgs("native:\n", nativeArgs);
        ret->addCommand(NULL, mNativeExe, (char * const *)nativeArgs);
        free((void *)defaultOutFile);

        return ret;
    }


    inline PipelinedCommands *
    getPrelinkPipeline(void) {
        size_t i = 0;

        if (mStage != CompileOptions::Stage_Native_Link) {
            return NULL;
        }
        PipelinedCommands *ret = new PipelinedCommands(-1, 1, 2);

        bool needLangSpec = true;
        char **cmdlineOptions = NULL;
        size_t nCmdlineOptions =
            mCOPtr->getOptions(cmdlineOptions, 0, 0,
                               CompileOptions::Options_Prelinker);
        for (i = 0; i < nCmdlineOptions; i++) {
            if (isValidString(cmdlineOptions[i]) == true) {
                if (strcmp(cmdlineOptions[i], "--C") == 0 ||
                    strcmp(cmdlineOptions[i], "--F") == 0) {
                    needLangSpec = false;
                    break;
                }
            }
        }
        if (needLangSpec == true) {
            // Default is the fortran for now.
            mCOPtr->addOption(CompileOptions::Options_Prelinker, "--F");
        }
        free((void *)cmdlineOptions);

        char **prelinkerArgs = NULL;
        // xmp_collect_init --cc mpicc --PID $pidStr --F $*
        //	... head margin is six.
        (void)mCOPtr->getOptionsV(prelinkerArgs,
                                  3, 0,
                                  CompileOptions::Options_Prelinker,
                                  CompileOptions::Options_Objects,
                                  CompileOptions::Options_End);

        mPrelinkerOutputFile = OutputFileProvider::tempnam("/tmp/.", ".o");

        i = 0;
        prelinkerArgs[i++] = (char *)mGetArgv0(mPrelinkerExe);
        prelinkerArgs[i++] = (char *)"-o";
        prelinkerArgs[i++] = (char *)mPrelinkerOutputFile;

        // mDumpArgs("pre-link:\n", prelinkerArgs);
        ret->addCommand(NULL, mPrelinkerExe, (char * const *)prelinkerArgs);

        return ret;
    }


    inline void
    unlinkPrelinkerOutput(void) {
        OutputFileProvider::safeUnlink(mPrelinkerOutputFile);
    }


    inline PipelinedCommands *
    getNativeLinkPipeline(void) {
        if (mStage != CompileOptions::Stage_Native_Link) {
            return NULL;
        }
        PipelinedCommands *ret = new PipelinedCommands(-1, -1, 2);

        const char *outFile = mCOPtr->getOutputFile();

        char **linkerArgs = NULL;
        size_t headMargin = 1;
        if (isValidString(outFile) == true) {
            headMargin = 3;
        }
        if (isValidString(mPrelinkerOutputFile) == true) {
            headMargin++;
        }
        (void)mCOPtr->getOptionsV(linkerArgs,
                                  headMargin, 0,
                                  CompileOptions::Options_Native,
                                  CompileOptions::Options_Objects,
                                  CompileOptions::Options_Native_Linker,
                                  CompileOptions::Options_End);
        size_t i = 0;
        linkerArgs[i++] = (char *)mGetArgv0(mNativeExe);
        if (isValidString(outFile) == true) {
            linkerArgs[i++] = (char *)"-o";
            linkerArgs[i++] = (char *)outFile;
        }
        if (isValidString(mPrelinkerOutputFile) == true) {
            linkerArgs[i++] = (char *)mPrelinkerOutputFile;
        }

        // mDumpArgs("native link:\n", linkerArgs);
        ret->addCommand(NULL, mNativeExe, (char * const *)linkerArgs);

        return ret;
    }


    inline const char *
    getOutputFileSuffix(bool forNative = true,
                        CompileOptions::CompileStageT stage =
                        CompileOptions::Stage_Unknown) {
        const char *ret = NULL;

        if (stage == CompileOptions::Stage_Unknown) {
            stage = mStage;
        }

        switch (stage) {

            case CompileOptions::Stage_CPP: {
                ret = ".i";
                break;
            }
            case CompileOptions::Stage_Frontend: {
                ret = ".xml";
                break;
            }
            case CompileOptions::Stage_XcodeML_Translate: {
                if (mHasNoTranslator == true) {
                    ret = ".xml";
                } else {
                    ret = ".trans.xml";
                }
                break;
            }
            case CompileOptions::Stage_Decompile: {
                ret = ".decomp.f90";
                break;
            }
            case CompileOptions::Stage_Native_Compile: {
                ret = (forNative == true) ? ".o" : ".decomp.f90";
                break;
            }
            case CompileOptions::Stage_Native_Assemble: {
                ret = (forNative == true) ? ".s" : ".decomp.f90";
                break;
            }
            case CompileOptions::Stage_Native_Link: {
                ret = (forNative == true) ? NULL : ".decomp.f90";
                break;
            }

            default: {
                break;
            }

        }

        return ret;
    }


    inline void
    cleanupForBogusCpp(const char *sourceFile) {
        if (mIsCppGccSuit == true) {
            const char *sFile = 
                CompileOptions::getDefaultOutputFilename(
                    sourceFile, ".s");
            OutputFileProvider::safeUnlink((const char *)sFile);
            free((void *)sFile);
        }
    }


};


#endif // __PIPELINEBUILER_H__
