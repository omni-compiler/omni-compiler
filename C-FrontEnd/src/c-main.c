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
#include "c-ptree.h"


/* display parse tree before making XcodeML */
static unsigned _SW_DISP_PTREE = 000;        // switch bits for dynamic debugging
#define _DISP_PTREE_IF(flag, expr, msg)  if ((flag) & _SW_DISP_PTREE) dispParseTree(stderr, expr, msg)


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
    _DISP_PTREE_IF(001, expr, "execParse");
    //    dispParseTree(stderr, expr, "execParse");

    if(s_inFile)
        fclose(fpIn);

    if(s_hasError || expr == NULL)
        goto end;

    reduceExpr(expr);
    _DISP_PTREE_IF(002, expr, "reduceExpr");

    if(s_hasError)
        goto end;

    if(s_verbose)
        printf("compiling ...\n");

    compile(expr);
    _DISP_PTREE_IF(004, expr, "compile");

    if(s_hasError)
        goto end;

    convertSyntax(expr);
    _DISP_PTREE_IF(010, expr, "convertSyntax");
    collectTypeDesc(expr);
    _DISP_PTREE_IF(020, expr, "collectTypeDesc");

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

