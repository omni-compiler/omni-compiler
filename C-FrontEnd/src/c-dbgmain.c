/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-dbgmain.c
 * frontend for debugging
 */

#include <time.h>
#include "c-expr.h"
#include "c-comp.h"
#include "c-lexyacc.h"
#include "c-option.h"


//macro for checking memory corruption
#if 0
#define CHECK_TYPEDESC_LIST {\
        int i, sz = 1024 * 512;\
        char *p[sz];\
        for(i = 0; i < sz; ++i) { p[i] = (char*)malloc(8); memset(p[i], 0xFF, 8); }\
        for(i = 0; i < sz; ++i) { free(p[i]); }\
        CCOL_DListNode *ite;\
        CExprOfTypeDesc *td0 = NULL;\
        CCOL_DL_FOREACH(ite, &s_typeDescList) {\
            CExprOfTypeDesc *td = EXPR_T(CCOL_DL_DATA(ite));\
            assert(EXPR_REF(td) > 0);\
            if(EXPR_CODE(td) != EC_TYPE_DESC) {\
                DBGPRINT(("invalid    :" ADDR_PRINT_FMT "\n", (uintptr_t)td));\
                DBGPRINT(("prechecked :" ADDR_PRINT_FMT "\n", (uintptr_t)td0));\
                assert(EXPR_CODE(td) == EC_TYPE_DESC);\
            }\
            td0 = td;\
        }\
    }
#else
#define CHECK_TYPEDESC_LIST
#endif


int
main(int argc, char** argv)
{
    int i;
    int dumpParesd = 0, dumpReduced = 0, dumpResolved = 0, dumpConverted = 1;
    int dumpMisc = 1, outputStderr = 0;
    char *arg;

    setTimestamp();

    for(i = 1; i < argc; ++i) {
        arg = argv[i];
        if(strcmp(arg, "-yd") == 0) {
            yydebug = 1;
        } else if(strcmp(arg, "-dp") == 0) {
            dumpParesd = 1;
        } else if(strcmp(arg, "-dr") == 0) {
            dumpReduced = 1;
        } else if(strcmp(arg, "-ds") == 0) {
            dumpResolved = 1;
        } else if(strcmp(arg, "-da") == 0) {
            dumpParesd = 1;
            dumpReduced = 1;
            dumpResolved = 1;
        } else if(strcmp(arg, "-d0") == 0) {
            dumpConverted = 0;
            dumpMisc = 0;
        } else if(strcmp(arg, "-dS") == 0) {
			s_debugSymbol = 1;
        } else if(strcmp(arg, "-stderr") == 0) {
            outputStderr = 1;
        } else {
            continue;
        }
        arg[0] = 0;
    }

    if(procOptions(argc, argv) == 0)
        return CEXIT_CODE_ERR;

#ifdef MTRACE
    mtrace();
#endif

    initStaticData();

    FILE *fpIn, *fpOut;

    if(s_inFile) {
        fpIn = fopen(s_inFile, "r");
        if(fpIn == NULL) {
            perror(CERR_501);
            return CEXIT_CODE_ERR;
        }
    } else {
        fpIn = stdin;
    }

    CExpr *expr = execParse(fpIn);

    if(s_inFile)
        fclose(fpIn);

    if(dumpParesd || s_hasError) {
        fprintf(stdout, "\n--- Parsed ---\n");
        dumpExpr(stdout, expr);
        fflush(stdout);
    }

    if(s_hasError)
        goto end;

    reduceExpr(expr);

    if(dumpReduced || s_hasError) {
        printf("--- Reduced ---\n");
        dumpExpr(stdout, expr);
        fflush(stdout);
    }

    if(s_hasError)
        goto end;

    compile(expr);

    if(dumpResolved || s_hasError) {
        fprintf(stdout, "\n--- Resolved ---\n");
        dumpExpr(stdout, expr);
        fflush(stdout);
    }

    CHECK_TYPEDESC_LIST;

    if(s_hasError)
        goto end;

    convertSyntax(expr);
    collectTypeDesc(expr);

    if(dumpConverted) {
        fprintf(stdout, "\n--- Converted ---\n");
        dumpExpr(stdout, expr);
        fflush(stdout);
    }

    CHECK_TYPEDESC_LIST;

    if(s_hasError)
        goto end;

    if(dumpMisc)
        fprintf(stdout, "\n--- XcodeML ---\n");

    convertFileIdToNameTab();

    CHECK_TYPEDESC_LIST;

    if(s_outFile) {
        fpOut = fopen(s_outFile, "w");
        if(fpOut == NULL) {
            perror(CERR_502);
            return CEXIT_CODE_ERR;
        }
    } else {
        fpOut = stdout;
    }

    outputXcodeML(fpOut, expr);

    if(s_outFile)
        fclose(fpOut);

    fflush(stdout);

    if(s_hasError)
        goto end;

    if(dumpMisc) {
        fprintf(stdout, "\n--- File ID ---\n");
        dumpFileIdTable(stdout);
        fflush(stdout);
    }

  end:

    if(s_hasError || s_hasWarn) {
        FILE *errfp = outputStderr ? stderr : stdout;
        fprintf(errfp, "\n--- Error ---\n");
        CCOL_SL_REVERSE(&s_errorList);

        if(dumpMisc)
            dumpError(errfp);
        else
            printErrors(errfp);
    }

    if(dumpMisc)
        fprintf(stdout, "\nCompleted.\n");

    freeExpr(expr);
    freeStaticData();
    fflush(stdout);
    fflush(stderr);

#ifdef MTRACE
    muntrace();
#endif

    return s_hasError ? CEXIT_CODE_ERR : CEXIT_CODE_OK;
}

