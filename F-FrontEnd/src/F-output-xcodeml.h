/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
/**
 * \file F-output-xcodeml.h
 */

#ifndef _F_XCODEML_H_
#define _F_XCODEML_H_

#ifdef Min
#undef Min
#endif
#define Min(a, b) (((a) > (b)) ? (b) : (a))

extern void output_XcodeML_file();

#define CEXPR_OPTVAL_CHARLEN 128

#define C_SYM_NAME(X)  (SYM_NAME(X))

#define F_FRONTEND_NAME "XcodeML/Fortran-FrontEnd"
#define F_TARGET_LANG   "Fortran"
#define F_FRONTEND_VER  "1.0"

extern char s_timestamp[];
extern char s_xmlIndent[];

#endif /* _F_XCODEML_H_ */
