/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file c-option.h
 */

#ifndef _C_OPTION_H_
#define _C_OPTION_H_

#include "c-expr.h"

#define CEXIT_CODE_OK   0
#define CEXIT_CODE_ERR  -1

// data size
//! size of void*
extern unsigned int s_sizeAddr;
//! size of char
extern unsigned int s_sizeChar;
//! size of wchar_t
extern unsigned int s_sizeWChar;
//! size of short
extern unsigned int s_sizeShort;
//! size of int
extern unsigned int s_sizeInt;
//! size of long
extern unsigned int s_sizeLong;
//! size of long long
extern unsigned int s_sizeLongLong;
//! size of float
extern unsigned int s_sizeFloat;
//! size of double
extern unsigned int s_sizeDouble;
//! size of long double
extern unsigned int s_sizeLongDouble;
//! size of _Bool
extern unsigned int s_sizeBool;
//! type of sizeof operator
extern unsigned int s_basicTypeSizeOf;

//! alignment of void*
extern unsigned int s_alignAddr;
//! alignment of char
extern unsigned int s_alignChar;
//! alignment of wchar_t
extern unsigned int s_alignWChar;
//! alignment of short
extern unsigned int s_alignShort;
//! alignment of int
extern unsigned int s_alignInt;
//! alignment of long
extern unsigned int s_alignLong;
//! alignment of long long
extern unsigned int s_alignLongLong;
//! alignment of float
extern unsigned int s_alignFloat;
//! alignment of double
extern unsigned int s_alignDouble;
//! alignment of long double
extern unsigned int s_alignLongDouble;
//! alignment of _Bool
extern unsigned int s_alignBool;

//! verbose mode
extern unsigned int s_verbose;
//! output raw line number to XcodeML and error message
extern unsigned int s_rawlineNo;
//! suppress type IDs which are same types
extern unsigned int s_suppressSameTypes;
//! if output XcodeProgram attributes and line number
extern unsigned int s_xoutputInfo;
//! if output line number/file attribute
extern unsigned int s_outputLineNo;
//! if support gcc built-in type/functions
extern unsigned int s_supportGcc;
//! use built-in wchar_t
extern unsigned int s_useBuiltinWchar;
//! treat wide character as unsigned short
extern unsigned int s_useShortWchar;
//! use __builtin_va_arg
extern unsigned int s_useBuiltinVaArg;
//! transform function in initializer
extern unsigned int s_transFuncInInit;
//! transform xmp pragma
extern unsigned int s_useXmp;
//! use debug output for symbol table
extern unsigned int s_debugSymbol;

//! input file name
extern const char *s_inFile;
//! output file name
extern const char *s_outFile;
//! prefix which is named for anonymous struct/union
extern char s_anonymousCompositePrefix[];
//! prefix which is named for anonymous member
extern char s_anonymousMemberPrefix[];
//! prefix which is named for temporary variable
extern char s_tmpVarPrefix[];
//! prefix which is named for gcc local label
extern char s_gccLocalLabelPrefix[];
//! indent width of xml text
extern char s_xmlIndent[];
//! xml encoding attribute
extern char s_xmlEncoding[];
//! XcodeProgram source attribute
extern char s_sourceFileName[];
//! timestamp
extern char s_timeStamp[];

extern int          procOptions(int argc, char **argv);
extern unsigned int getBasicTypeSize(CBasicTypeEnum bt);
extern unsigned int getBasicTypeAlign(CBasicTypeEnum bt);
extern void         setTimestamp(void);
extern void         freeStaticOptionData(void);
extern int          isWarnableId(const char *msg);

#endif // _C_OPTION_H_

