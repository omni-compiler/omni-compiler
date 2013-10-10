/*
 * $Id: CompileOptions.h 235 2013-09-06 03:42:51Z m-hirano $
 */
#ifndef __COMPILEOPTIONS_H__
#define __COMPILEOPTIONS_H__

#include <nata/nata_rcsid.h>

#include <nata/libnata.h>

#include <nata/RegularExpression.h>
#include <nata/CommandLineParameters.h>

#include "OutputFileProvider.h"

#include <vector>

#include <nata/nata_perror.h>





class CompileOptions: public CommandLineParameters {


private:
    __rcsId("$Id: CompileOptions.h 235 2013-09-06 03:42:51Z m-hirano $")





public:


    typedef enum {
        Options_Unknown = 0,

        Options_CPP,
        Options_Frontend,
        Options_XcodeML_Translator,
        Options_Decompiler,
        Options_Native,
        Options_Native_Linker,
        Options_Prelinker,

        Options_Sources,
        Options_Objects,

        Options_End
    } OptionsTypeT;


    typedef enum {
        Stage_Unknown = 0,

        Stage_CPP,
        Stage_Frontend,
        Stage_XcodeML_Translate,
        Stage_Decompile,
        Stage_Native_Assemble,
        Stage_Native_Compile,
        Stage_Native_Link,

        Stage_End
    } CompileStageT;


private:


    typedef std::vector<const char *> mOptionList;
    mOptionList mOpts[(size_t)Options_End];	// Options for each command.

    CompileStageT mStage;

    const char *mCppExe;
    const char *mTranslatorExe;
    const char *mNativeCompilerExe;

    const char *mOutput;
    bool mForceCpp;

    RegularExpression *mCppExpPtr;
    RegularExpression *mSrcExpPtr;
    RegularExpression *mF77SrcExpPtr;

    bool mIsNativeNeeded;
    bool mIsCompilationNeeded;

    bool mIsVerbose;
    bool mIsDryrun;
    bool mIsNoCleanup;
    bool mIsIgnoreOutput;

    bool mEmitCPPResult;
    bool mEmitFrontendResult;
    bool mEmitTranslatorResult;
    bool mEmitDecompilerResult;

    bool mOnlyPrintSrcs;

    bool mIsUsageEmitted;





// Static private methods:


    static inline void
    mSetStage(CompileOptions *m, CompileStageT s) {
        if (m->mStage == Stage_Unknown ||
            (int)m->mStage > (int)s) {
            m->mStage = s;
        }
    }


    static inline mOptionList &
    mGetOptionList(OptionsTypeT t, CompileOptions *m) {
        if ((int)t > 0 && (int)t < (int)Options_End) {
            return m->mOpts[(int)t];
        } else {
            fatal("Invalid options type (%d).", (int)t);
            // not reached.
            mOptionList *dum = new mOptionList();
            return *dum;
        }
    }


    static inline bool
    mParseSpecific(OptionsTypeT t, CompileOptions *m, const char *arg) {
        mOptionList &l = mGetOptionList(t, m);
        l.push_back(strdup(arg));
        return true;
    }


    static inline bool
    mParseGeneric(OptionsTypeT t, CompileOptions *m, const char *arg) {
        mOptionList &l = mGetOptionList(t, m);
        bool ret = false;
        const char *cnm = strchr(arg, ',');

        if (cnm != NULL && *(cnm + 1) != '\0') {
            char *tokens[1024];
            char *args = nata_TrimRight((cnm + 1), " \t\r\n");
            int nTokens = nata_GetToken(args, tokens, 1024, ",");
            int i;
            for (i = 0; i < nTokens; i++) {
                l.push_back(strdup(tokens[i]));
            }
            (void)free((void *)args);
            ret = true;
        } else {
            fprintf(stderr, "%s: invalid option \"%s\".\n",
                    m->commandName(), arg);
        }

        return ret;
    }





// Private methods:


    inline void
    mVAddOptions(OptionsTypeT t, va_list args) {
        const char *arg;
        mOptionList &l = mGetOptionList(t, this);

        while (true) {
            arg = va_arg(args, const char *);
            if (arg != NULL) {
                l.push_back(strdup(arg));
            } else {
                break;
            }
        }
    }


    inline size_t
    mVGetOptions(char **&retList,
                 size_t headMargin,
                 size_t tailMargin,
                 OptionsTypeT t0, va_list args) {
        std::vector<OptionsTypeT> types;
        OptionsTypeT t;
        int iT;
        char **ret = NULL;
        size_t n = 0;
        size_t i;
        size_t j;
        size_t idx = headMargin;

        retList = NULL;
        iT = (int)t0;
        if (iT > 0 && iT < (int)Options_End) {
            types.push_back(t0);
            mOptionList &l = mGetOptionList(t0, this);
            n += l.size();
        } else {
            goto Done;
        }

        while (true) {
            iT = va_arg(args, int);	// It is a ok to use int since
                                        // the enum value is expressed
                                        // as an int.
            if (iT > 0 && iT < (int)Options_End) {
                t = (OptionsTypeT)iT;
                mOptionList &l = mGetOptionList(t, this);
                n += l.size();
                types.push_back(t);
            } else {
                break;
            }
        }

        n += (headMargin + tailMargin + 1);
        ret = (char **)malloc(sizeof(char *) * n);
        if (ret == NULL) {
            goto Done;
        }
        (void)memset((void *)ret, 0, sizeof(char *) * n);

        for (i = 0; i < types.size(); i++) {
            mOptionList &l = mGetOptionList(types[i], this);
            for (j = 0; j < l.size(); j++) {
                ret[idx++] = (char *)(l[j]);
            }
        }

        retList = ret;

        Done:
        return n;
    }





// Parsers:


    static bool
    parseSpecificCppOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_CPP, (CompileOptions *)ctx, arg);
    }
    friend bool parseSpecificCppOptions(const char *arg, void *ctx);

    static bool
    parseGenericCppOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_CPP, (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericCppOptions(const char *arg, void *ctx);


#if 0
    static bool
    parseSpecificFrontendOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_Frontend, (CompileOptions *)ctx, arg);
    }
    friend bool parseSpecificFrontendOptions(const char *arg, void *ctx);
#endif

    static bool
    parseGenericFrontendOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_Frontend, (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericFrontendOptions(const char *arg, void *ctx);


#if 0
    static bool
    parseSpecificTranslatorOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_XcodeML_Translator,
                              (CompileOptions *)ctx, arg);
    }
    friend bool parseSpecificTranslatorOptions(const char *arg, void *ctx);
#endif

    static bool
    parseGenericTranslatorOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_XcodeML_Translator,
                             (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericTranslatorOptions(const char *arg, void *ctx);


#if 0
    static bool
    parseSpecificDecompilerOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_Decompiler, (CompileOptions *)ctx, arg);
    }
    friend bool parseSpecificDecompilerOptions(const char *arg, void *ctx);
#endif

    static bool
    parseGenericDecompilerOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_Decompiler, (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericDecompilerOptions(const char *arg, void *ctx);


    static bool
    parseSpecificLinkerOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_Native_Linker,
                              (CompileOptions *)ctx, arg);
    }
    friend bool parseSpecificLinkerOptions(const char *arg, void *ctx);

#if 0
    static bool
    parseGenericLinkerOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_Native_Linker,
                             (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericLinkerOptions(const char *arg, void *ctx);
#endif


    static bool
    parseGenericNativeOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_Native,
                             (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericNativeOptions(const char *arg, void *ctx);


    static bool
    parseGenericPrelinkerOptions(const char *arg, void *ctx) {
        return mParseGeneric(Options_Prelinker,
                             (CompileOptions *)ctx, arg);
    }
    friend bool parseGenericPrelinkerOptions(const char *arg, void *ctx);


    static bool
    parse_E(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_CPP);
        return true;
    }
    friend bool parse_E(const char *arg, void *ctx);

    static bool
    parse_Xf(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_Frontend);
        return true;
    }
    friend bool parse_Xf(const char *arg, void *ctx);

    static bool
    parse_Xt(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_XcodeML_Translate);
        return true;
    }
    friend bool parse_Xt(const char *arg, void *ctx);

    static bool
    parse_Xd(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_Decompile);
        return true;
    }
    friend bool parse_Xd(const char *arg, void *ctx);

    static bool
    parse_c(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_Native_Compile);
        return true;
    }
    friend bool parse_c(const char *arg, void *ctx);

    static bool
    parse_S(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        mSetStage(m, Stage_Native_Assemble);
        return true;
    }
    friend bool parse_S(const char *arg, void *ctx);

    static bool
    parse_cpp(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mForceCpp = true;
        return true;
    }
    friend bool parse_cpp(const char *arg, void *ctx);

    static bool
    parse_v(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mIsVerbose = true;
        return true;
    }
    friend bool parse_v(const char *arg, void *ctx);

    static bool
    parse_n(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mIsDryrun = true;
        return true;
    }
    friend bool parse_n(const char *arg, void *ctx);

    static bool
    parse_noclean(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mIsNoCleanup = true;
        return true;
    }
    friend bool parse_noclean(const char *arg, void *ctx);

    static bool
    parse_ignoreoutput(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mIsIgnoreOutput = true;
        return true;
    }
    friend bool parse_ignoreoutput(const char *arg, void *ctx);

    static bool
    parse_emit_cpp_result(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mEmitCPPResult = true;
        return true;
    }
    friend bool parse_emit_cpp_result(const char *arg, void *ctx);

    static bool
    parse_emit_frontend_result(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mEmitFrontendResult = true;
        return true;
    }
    friend bool parse_emit_frontend_result(const char *arg, void *ctx);

    static bool
    parse_emit_translator_result(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mEmitTranslatorResult = true;
        return true;
    }
    friend bool parse_emit_translator_result(const char *arg, void *ctx);

    static bool
    parse_emit_decompiler_result(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mEmitDecompilerResult = true;
        return true;
    }
    friend bool parse_emit_decompiler_result(const char *arg, void *ctx);


    static bool
    parse_only_print_sources(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mOnlyPrintSrcs = true;
        return true;
    }
    friend bool parse_only_print_sources(const char *arg, void *ctx);


    static bool
    parse_o(const char *arg, void *ctx) {
        CompileOptions *m = (CompileOptions *)ctx;
        m->mOutput = (const char *)strdup(arg);
        return true;
    }
    friend bool parse_o(const char *arg, void *ctx);


    static bool
    parse_cpp_command(const char *arg, void *ctx) {
        CompileOptions *m = (CompileOptions *)ctx;
        m->mCppExe = (const char *)strdup(arg);
        return true;
    }
    friend bool parse_cpp_command(const char *arg, void *ctx);

    static bool
    parse_translator_command(const char *arg, void *ctx) {
        CompileOptions *m = (CompileOptions *)ctx;
        m->mTranslatorExe = (const char *)strdup(arg);
        return true;
    }
    friend bool parse_translator_command(const char *arg, void *ctx);

    static bool
    parse_native_compiler_command(const char *arg, void *ctx) {
        CompileOptions *m = (CompileOptions *)ctx;
        m->mNativeCompilerExe = (const char *)strdup(arg);
        return true;
    }
    friend bool parse_nativecompiler_command(const char *arg, void *ctx);


    static bool
    parseAnyOptions(const char *arg, void *ctx) {
        return mParseSpecific(Options_Native,
                             (CompileOptions *)ctx, arg);
    }
    friend bool parseAnyOptions(const char *arg, void *ctx);


    static bool
    doUsage(const char *arg, void *ctx) {
        (void)arg;
        CompileOptions *m = (CompileOptions *)ctx;
        m->mIsUsageEmitted = true;

        fprintf(stderr, "Usage: %s [options] file...\n", m->commandName());
        fprintf(stderr, "Options:\n");

        fprintf(stderr, "\n");
        fprintf(stderr, "  [\\-]+(h|?|help)"
                "\tShow this.\n");
        fprintf(stderr, "\n");

        fprintf(stderr, "  -Wf,opt[,opt...]"
                "\tPass options directly to the frontend.\n");
        fprintf(stderr, "  -Wt,opt[,opt...]"
                "\tPass options directly to the XcodeML-based translator.\n");
        fprintf(stderr, "  -Wd,opt[,opt...]"
                "\tPass options directly to the decompiler.\n");
        fprintf(stderr, "  -Wn,opt[,opt...]"
                "\tPass options directly to the native compiler.\n");
        fprintf(stderr, "  -Wpl,opt[,opt...]"
                "\tPass options directly to the pre-linker.\n");

        fprintf(stderr, "  -Xf"
                "\t\t\tStop the process after the frontend stage.\n");
        fprintf(stderr, "  -Xt"
                "\t\t\tStop the process after the translation stage.\n");
        fprintf(stderr, "  -Xd"
                "\t\t\tStop the process after the decompilation stage.\n");

        fprintf(stderr, "  -cpp"
                "\t\t\tProcess the files by the preprocessor forcibly.\n");

        fprintf(stderr, "  -v"
                "\t\t\tVerbose mode.\n");

        fprintf(stderr, "  -n"
                "\t\t\tOnly print the commands that would be executed.\n");

        fprintf(stderr, "  -no-clean"
                "\t\tDon't remove any output files when any failures.\n");

        fprintf(stderr, "  -ignore-output"
                "\tIgnore the output filename specified by -o option.\n");

        fprintf(stderr, "  -emit-cpp-result"
                "\tEmit the CPP result to a file.\n");
        fprintf(stderr, "  -emit-frontend-result"
                "\tEmit the frontend result to a file.\n");
        fprintf(stderr, "  -emit-translator-result"
                "\tEmit the translator result to a file.\n");
        fprintf(stderr, "  -emit-decompiler-result"
                "\tEmit the decompiler result to a file.\n");

        fprintf(stderr, "  -only-print-sources"
                "\tOnly list recognized files as input.\n");

        fprintf(stderr, "  --cpp-command=cmd"
                "\t\tUse the \"cmd\" as a cpp.\n");
        fprintf(stderr, "  --translator-command=cmd"
                "\tUse the \"cmd\" as an XcodeML-based translator.\n");
        fprintf(stderr, "  --native-compiler-command=cmd"
                "\tUse the \"cmd\" as a native compiler.\n");

        fprintf(stderr, "\n");
        fprintf(stderr, "\tAny other options are treated as of the "
                "c89(1)-based compile driver,\n"
                "\tand directly passed to the native compiler if needed.\n");
        return false;
    }
    friend bool doUsage(const char *arg, void *ctx);    





public:


    CompileOptions(const char *argv0) :
        CommandLineParameters(argv0),
        // mOpts,
        mStage(Stage_Unknown),
        mCppExe(NULL),
        mTranslatorExe(NULL),
        mNativeCompilerExe(NULL),
        mOutput(NULL),
        mForceCpp(false),
        mCppExpPtr(NULL),
        mSrcExpPtr(NULL),
        mF77SrcExpPtr(NULL),
        mIsNativeNeeded(false),
        mIsCompilationNeeded(false),
        mIsVerbose(false),
        mIsDryrun(false),
        mIsNoCleanup(false),
        mIsIgnoreOutput(false),
        mEmitCPPResult(false),
        mEmitFrontendResult(false),
        mEmitTranslatorResult(false),
        mEmitDecompilerResult(false),
        mOnlyPrintSrcs(false),
        mIsUsageEmitted(false) {

        addParser("^-Wp,.*",
                  parseGenericCppOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);
        addParser("^-[DUI]..*",
                  parseSpecificCppOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wf,.*",
                  parseGenericFrontendOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wt,.*",
                  parseGenericTranslatorOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wd,.*",
                  parseGenericDecompilerOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wl,.*",
                  parseSpecificLinkerOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);
        addParser("^-[Ll]..*",
                  parseSpecificLinkerOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wn,.*",
                  parseGenericNativeOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("^-Wpl,.*",
                  parseGenericPrelinkerOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);

        addParser("-E",
                  parse_E, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-Xf",
                  parse_Xf, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-Xt",
                  parse_Xt, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-Xd",
                  parse_Xd, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-c",
                  parse_c, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-S",
                  parse_S, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-cpp",
                  parse_cpp, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-v",
                  parse_v, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-n",
                  parse_n, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-no-clean",
                  parse_noclean, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-ignore-output",
                  parse_ignoreoutput, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-emit-cpp-result",
                  parse_emit_cpp_result, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-emit-frontend-result",
                  parse_emit_frontend_result, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-emit-translator-result",
                  parse_emit_translator_result, (void *)this,
                  Parse_This, Value_None, true, false, NULL);

        addParser("-emit-decompiler-result",
                  parse_emit_decompiler_result, (void *)this,
                  Parse_This, Value_None, true, false, NULL);


        addParser("-only-print-sources",
                  parse_only_print_sources, (void *)this,
                  Parse_This, Value_None, true, false, NULL);


        addParser("-o",
                  parse_o, (void *)this,
                  Parse_Next, Value_String, true, false, NULL);


        addParser("--cpp-command=",
                  parse_cpp_command, (void *)this,
                  Parse_ThisWithValue, Value_String, true, false, NULL);

        addParser("--translator-command=",
                  parse_translator_command, (void *)this,
                  Parse_ThisWithValue, Value_String, true, false, NULL);

        addParser("--native-compiler-command=",
                  parse_native_compiler_command, (void *)this,
                  Parse_ThisWithValue, Value_String, true, false, NULL);


        addParser("^-[\\-]*\\?$",
                  doUsage, (void *)this,
                  Parse_This, Value_None, true, true, NULL);
        addParser("^-[\\-]*h$",
                  doUsage, (void *)this,
                  Parse_This, Value_None, true, true, NULL);
        addParser("^-[\\-]*help$",
                  doUsage, (void *)this,
                  Parse_This, Value_None, true, true, NULL);


        addParser("^-.*",
                  parseAnyOptions, (void *)this,
                  Parse_This, Value_None, true, true, NULL);


        mCppExpPtr = new RegularExpression((const char *)".*\\.F[0-9]*$");
        mSrcExpPtr = new RegularExpression((const char *)".*\\.[fF][0-9]*$");
        mF77SrcExpPtr = new RegularExpression((const char *)".*\\.[fF][7]*$");
    }


    ~CompileOptions(void) {
        size_t i;
        int j;

        for (j = 1; j < (int)Options_End; j++) {
            mOptionList &l = mOpts[j];

            for (i = 0; i < l.size(); i++) {
                (void)free((void *)l[i]);
            }
        }

        (void)free((void *)mCppExe);
        (void)free((void *)mTranslatorExe);
        (void)free((void *)mNativeCompilerExe);

        (void)free((void *)mOutput);

        delete mCppExpPtr;
        delete mSrcExpPtr;
        delete mF77SrcExpPtr;
    }


    inline size_t
    getOptionsV(char **&retList,
                size_t headMargin,
                size_t tailMargin,
                OptionsTypeT t0,
                ...) {
        va_list args;
        va_start(args, t0);
        va_end(args);
        return mVGetOptions(retList, headMargin, tailMargin, t0, args);
    }


    inline size_t
    getOptions(char **&retList,
               size_t headMargin,
               size_t tailMargin,
               OptionsTypeT t) {
        return getOptionsV(retList, headMargin, tailMargin, t, Options_End);
    }


    inline void
    addOptions(OptionsTypeT t, ...) {
        va_list args;
        va_start(args, t);
        va_end(args);
        mVAddOptions(t, args);
    }


    inline void
    addOption(OptionsTypeT t, const char *arg) {
        addOptions(t, arg, NULL);
    }


    inline const char *
    getOutputFile(void) {
        return mOutput;
    }


    inline const char *
    getCppCommand(void) {
        return mCppExe;
    }


    inline const char *
    getTranslatorCommand(void) {
        return mTranslatorExe;
    }


    inline const char *
    getNativeCompilerCommand(void) {
        return mNativeCompilerExe;
    }


    inline bool
    isForcedCpp(void) {
        return mForceCpp;
    }


    inline bool
    isCppNeeded(const char *filename) {
        return (mForceCpp == true ||
                mCppExpPtr->match(filename));
    }


    inline bool
    seemsF77Source(const char *filename) {
        return mF77SrcExpPtr->match(filename);
    }


    inline bool
    seemsSource(const char *filename) {
        return mSrcExpPtr->match(filename);
    }


    inline bool
    validateOptions(void) {
        bool ret = false;

        int argc;
        const char **argv = NULL;

        getUnparsedOptions(argc, argv);
        if (argc <= 0) {
            fprintf(stderr, "%s: error: no input files.\n",
                    commandName());
            goto Done;
        }

        if (mStage == Stage_Unknown) {
            mStage = Stage_Native_Link;
        }

        switch (mStage) {

            case Stage_Native_Assemble:
            case Stage_Native_Compile:
            case Stage_Native_Link: {
                mIsNativeNeeded = true;

                int i;
                const char *objFile;
                for (i = 0; i < argc; i++) {
                    objFile = NULL;
                    if (seemsSource(argv[i]) == true) {
                        mIsCompilationNeeded = true;
                        addOption(Options_Sources, strdup(argv[i]));
                        objFile = getDefaultOutputFilename(argv[i], ".o");
                    } else {
                        objFile = strdup(argv[i]);
                    }
                    addOption(Options_Objects, objFile);
                }

                break;
            }

            case Stage_CPP:
            case Stage_Frontend:
            case Stage_XcodeML_Translate:
            case Stage_Decompile: {
                int i;
                for (i = 0; i < argc; i++) {
                    if (seemsSource(argv[i]) == true) {
                        addOption(Options_Sources, strdup(argv[i]));
                    }
                }

                break;
            }

            default: {
                break;
            }

        }

        if (mStage == Stage_CPP) {
            mForceCpp = true;
        }

        if (mIsIgnoreOutput == true) {
            free((void *)mOutput);
            mOutput = NULL;
        }

        if (isValidString(mNativeCompilerExe) == true) {
            if (isValidString(mCppExe) == false) {
                mCppExe = strdup(mNativeCompilerExe);
            }
        }

        ret = true;

        Done:
        return ret;
    }


    inline size_t
    getSourceFiles(char **&retList) {
        size_t ret = getOptions(retList, 0, 0, Options_Sources);
        return --ret;
    }


    inline size_t
    getObjectFiles(char **&retList) {
        size_t ret = getOptions(retList, 0, 0, Options_Objects);
        return --ret;
    }


    inline CompileStageT
    getStage(void) {
        return mStage;
    }


    inline bool
    isVerbose(void) {
        return mIsVerbose;
    }


    inline bool
    isDryrun(void) {
        return mIsDryrun;
    }


    inline bool
    needCleanup(void) {
        return (mIsNoCleanup == false) ? true : false;
    }


    inline bool
    emitCppResult(void) {
        return mEmitCPPResult;
    }


    inline bool
    emitFrontendResult(void) {
        return mEmitFrontendResult;
    }


    inline bool
    emitTranslatorResult(void) {
        return mEmitTranslatorResult;
    }


    inline bool
    emitDecompilerResult(void) {
        return mEmitDecompilerResult;
    }


    inline bool
    onlyPrintSources(void) {
        return mOnlyPrintSrcs;
    }


    inline bool
    isUsageEmitted(void) {
        return mIsUsageEmitted;
    }


    static inline const char *
    getDefaultOutputFilename(const char *filename, const char *suffix) {
        const char *ret = NULL;

        if (isValidString(filename) == false) {
            return NULL;
        }

        char buf[PATH_MAX];
        const char *basename = OutputFileProvider::getFileBasename(filename);

        snprintf(buf, sizeof(buf), "%s%s", basename, suffix);
        ret = strdup(buf);
        free((void *)basename);

        return ret;
    }


};


#endif // __COMPILEOPTIONS_H__
