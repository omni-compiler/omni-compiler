/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-cmn.h
 * common header file
 */

#ifndef _C_COMMON_H_
#define _C_COMMON_H_

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#include <sys/types.h>

#ifdef DEBUG
#define CEXPR_DEBUG
#endif

#ifdef CEXPR_DEBUG

#   define DBGPRINT(fmt)                {printf fmt;}

#   define assertYYLineno(c) {\
    if((c) == 0) {\
        DBGPRINT(("\n[Assertion Failed near rawLinNum %d]\n", s_lineNumInfo.ln_rawLineNum));\
        assert(c);\
    }\
}

#   define assertExpr(expr, c) {\
        if((c) == 0) {\
            DBGPRINT(("\n[Assertion Failed near rawLinNum %d]\n", s_lineNumInfo.ln_rawLineNum));\
            assert(expr != NULL);\
            dumpExpr(stderr, expr);\
            assert(0);\
        }\
    }

#   define assertExprCode(expr, c) {\
        if((expr != NULL && EXPR_CODE(expr) != (c))) {\
            DBGPRINT(("\n[Assertion Failed near rawLinNum %d]\n", s_lineNumInfo.ln_rawLineNum));\
            dumpExpr(stderr, expr);\
            assert(0);\
        }\
    }

#   define assertExprStruct(expr, c) {\
        if(expr != NULL && EXPR_STRUCT(expr) != (c)) {\
            DBGPRINT(("\n[Assertion Failed near rawLinNum %d]\n", s_lineNumInfo.ln_rawLineNum));\
            dumpExpr(stderr, expr);\
            assert(0);\
        }\
    }

/* CEXPR_TRACE_MODE 1:mono, 2:color */
#   if (CEXPR_TRACE_COLOR != 2)
#       define DBGPRINTC(color, fmt)    {printf fmt; fflush(stdout);}
#   else
#       define DBGPRINTC(color, fmt)    {fputs(color,stdout); printf fmt; fputs(ESQ_DEFAULT, stdout); fflush(stdout); }
#   endif

#   ifdef CEXPR_DEBUG_STAT
#       if (CEXPR_TRACE_COLOR != 2)
#           define STAT_TRACE(fmt)      {printf fmt; fflush(stdout);}
#       else
#           define STAT_TRACE(fmt)      DBGPRINTC(ESQ_GREEN, fmt);
#       endif
#   else
#       define STAT_TRACE(fmt)
#   endif

#   ifdef CEXPR_DEBUG_MEM
#       if (CEXPR_TRACE_COLOR != 2)
#           define ALLOC_TRACE(addr)    DBGPRINT(("@CExpr:" ADDR_PRINT_FMT "@\n", (uintptr_t)(addr)))
#       else
#           define ALLOC_TRACE(addr)    DBGPRINTC(ESQ_RED, ("@CExpr:" ADDR_PRINT_FMT "@\n", (uintptr_t)(addr)));
#       endif
#   else
#       define ALLOC_TRACE(addr)
#   endif

#   ifdef CEXPR_DEBUG_MEM
#       if (CEXPR_TRACE_COLOR != 2)
#           define XFREE(x)             { DBGPRINT(("@CExpr-free:" ADDR_PRINT_FMT "@\n", (uintptr_t)(x))); fflush(stdout); free(x); }
#       else
#           define XFREE(x)             { DBGPRINTC(ESQ_RED, ("@CExpr-free:" ADDR_PRINT_FMT "@\n", (uintptr_t)(x))); free(x); }
#       endif
#   else
#       define XFREE(x)                 free(x)
#   endif

#   define PRIVATE_STATIC
#   define ABORT()                      assert(0)

#   define DBGDUMPEXPR(e)               dumpExpr(stderr, (CExpr*)(e));
#   define DBGDUMPERROR()               dumpError(stderr)

#else

//! debug print
#   define DBGPRINT(fmt)
//! debug print
#   define DBGPRINTC(color, fmt)
//! assert or dump lineno at parsin
#   define assertYYLineno(c)
//! assert expr or dump expr
#   define assertExpr(expr, c)
//! assert expression code or dump expr
#   define assertExprCode(expr, c)
//! assert struct kind or dump expr
#   define assertExprStruct(expr, c)
//! call free
#   define XFREE(x)                     free(x)
//! normally static, empty for debug build
#   define PRIVATE_STATIC               static
//! call abort
#   define ABORT()                      abort()
//! dump e
#   define DBGDUMPEXPR(e)
//! dump errors
#   define DBGDUMPERROR()
#   define STAT_TRACE(fmt)
#   define ALLOC_TRACE(addr)

#endif


#if (defined CEXPR_DEBUG && defined MTRACE)

#   include <mcheck.h>
#   define XALLOC(x)            ({ x* p = (x*)memset(malloc(sizeof(x)), 0, sizeof(x)); ALLOC_TRACE(p); p; })
#   define XALLOCSZ(x, sz)      ({ x* p = (x*)memset(malloc(sz), 0, sz); p; })

#else

    extern  void* xalloc(size_t sz);
#   define XALLOC(x)            ((x*)xalloc(sizeof(x)))
#   define XALLOCSZ(x, sz)      ((x*)xalloc(sz))

#endif

#define ESQ_RED         "\x1b[31m"
#define ESQ_GREEN       "\x1b[32m"
#define ESQ_DEFAULT     "\x1b[39m"

#define MAX_NAME_SIZ 1024

/*
 * Common Functons, Macros
 */

#define C_FRONTEND_NAME "XcodeML/C-FrontEnd"
#define C_TARGET_LANG   "C"
#define C_FRONTEND_VER  PACKAGE_VERSION

#if __WORDSIZE == 64
#define ADDR_PRINT_FMT  "0x%016lx"
#else
#define ADDR_PRINT_FMT  "0x%08x"
#endif

#endif /* _C_COMMON_H_ */

