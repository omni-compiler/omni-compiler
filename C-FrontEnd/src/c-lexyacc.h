/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#ifndef _C_LEXYACC_H_
#define _C_LEXYACC_H_

#include "ccol.h"

extern char             *yytext;
extern int              yylineno;
extern int              yydebug;

extern CLineNumInfo     s_lineNumInfo;
extern int              s_isParsing;
void                    setLineNumInfo();

extern void         yyerror(const char*);
extern int          yyparse(void);
extern int          yylex(void);
extern int          raw_yylex(void);
extern void         initLexer(FILE *fp);
extern void         freeLexer();
extern void         initParser();
extern void         freeParser();
extern void         copy_yytext(char *p);
extern const char   *getFileName(int fileId);
extern void         lexSyntaxError();
extern CExpr*       lexAllocExprCode(CExprCodeEnum exprCode, int token);
extern CExpr*       lexAllocSymbol();
extern CExpr*       execParse(FILE *fp);
extern void         lexStartSaveToken();
extern void         lexEndSaveTokenForGccAttr(CExpr *args);
extern void         lexEndSaveTokenForGccAsm(CExpr *asmExpr);

#endif /* _C_LEXYACC_H_ */

