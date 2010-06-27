/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-main.c
 * C_Front main
 */

#include "c-expr.h"
#include "c-comp.h"
#include "c-lexyacc.h"
#include "c-option.h"

/**
 * C_Front main.
 */


/**
 * \brief
 * C_Front main function
 */
int
main(int argc, char** argv)
{
    if(procOptions(argc, argv) == 0)
        return CEXIT_CODE_ERR;

    setTimestamp();
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

    int convertedFileId = 0;

    CExpr *expr = execParse(fpIn);

    if(s_inFile)
        fclose(fpIn);

    if(s_hasError || expr == NULL)
        goto end;

    reduceExpr(expr);

    if(s_hasError)
        goto end;

    if(s_verbose)
        printf("compiling ...\n");

    compile(expr);

    if(s_hasError)
        goto end;

    convertSyntax(expr);
    collectTypeDesc(expr);

    if(s_hasError)
        goto end;

    if(s_outFile) {
        fpOut = fopen(s_outFile, "w");
        if(fpOut == NULL) {
            perror(CERR_502);
            return CEXIT_CODE_ERR;
        }
    } else {
        fpOut = stdout;
    }

    convertFileIdToNameTab();
    convertedFileId = 1;
    outputXcodeML(fpOut, expr);

    if(s_outFile)
        fclose(fpOut);

  end:

    if(s_hasError || s_hasWarn) {
        CCOL_SL_REVERSE(&s_errorList);
        if(convertedFileId == 0)
            convertFileIdToNameTab();
        printErrors(stderr);
    }

    fflush(stdout);
    fflush(stderr);

    if(s_verbose)
        printf("completed\n");

    return s_hasError ? CEXIT_CODE_ERR : CEXIT_CODE_OK;
}

