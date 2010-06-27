/* 
 * $TSUKUBA_Release: Omni OpenMP Compiler 3 $
 * $TSUKUBA_Copyright:
 *  PLEASE DESCRIBE LICENSE AGREEMENT HERE
 *  $
 */
#ifndef _EXC_PLATFORM_H
#define _EXC_PLATFORM_H

#include "config.h"

#ifndef DEFAULT_UNINITED_CHAR
#define DEFAULT_UNINITED_CHAR ' '
#endif /* !DEFAULT_UNINITED_CHAR */

#undef _ANSI_ARGS_
#undef CONST

#if ((defined(__STDC__) || defined(SABER)) && !defined(NO_PROTOTYPE)) || defined(__cplusplus) || defined(USE_PROTOTYPE) || defined(__USE_FIXED_PROTOTYPES__)
#define _USING_PROTOTYPES_ 1
#define _ANSI_ARGS_(x) x
#define CONST const
#else
#define _ANSI_ARGS_(x) ()
#define CONST /**/
#endif /* __STDC__ ||.... */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* !HAVE_STRING_H */

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* !HAVE_UNISTD_H */

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif /* !HAVE_LIMITS_H */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif /* !HAVE_STDLIB_H */

#include <ctype.h>

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef TIME_WITH_SYS_TIME
#include <sys/time.h>
#include <time.h>
#else
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#else
#include <time.h>
#endif /* HAVE_SYS_TIME_H */
#endif /* TIME_WITH_SYS_TIME */

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif /* !HAVE_NETDB_H */

#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif /* !HAVE_SYS_SOCKET_H */

#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif /* !HAVE_NETINET_IN_H */

#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif /* !HAVE_ARPA_INET_H */

#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif /* !HAVE_RESOURCE_H */

#ifdef GETTOD_NOT_DECLARED
extern int      gettimeofday _ANSI_ARGS_((struct timeval *tp, struct timezone *tzp));
#endif /* GETTOD_NOT_DECLARED */

#ifdef HAVE_BSDGETTIMEOFDAY
#define gettimeofday(X, Y) BSDgettimeofday((X), (Y))
#endif /* HAVE_BSDGETTIMEOFDAY */

#ifdef HAVE_LOCALE_H
#include <locale.h>
#endif /* HAVE_LOCALE_H */

#if defined(__STDC__) || defined(HAVE_STDARG_H)
#   include <stdarg.h>
#   define EXC_VARARGS(type, name) (type name, ...)
#   define EXC_VARARGS_DEF(type, name) (type name, ...)
#   define EXC_VARARGS_START(type, name, list) (va_start(list, name), name)
#else
#   include <varargs.h>
#   ifdef __cplusplus
#       define EXC_VARARGS(type, name) (type name, ...)
#       define EXC_VARARGS_DEF(type, name) (type va_alist, ...)
#   else
#       define EXC_VARARGS(type, name) ()
#       define EXC_VARARGS_DEF(type, name) (va_alist)
#   endif
#   define EXC_VARARGS_START(type, name, list) \
        type name = (va_start(list), va_arg(list, type))
#endif

#define FALSE 0
#define TRUE 1

#ifndef HAVE_STRDUP
extern char	*strdup _ANSI_ARGS_((char *s));
#endif /* HAVE_STRDUP */

#ifdef bzero
#undef bzero
#endif
#define bzero(p, s)	memset((p), 0, (s))

#ifdef bcopy
#undef bcopy
#endif
#define bcopy(s, d, l)	memcpy((d), (s), (l))

#ifdef HAS_INT16
typedef short _omInt16_t;
typedef unsigned short _omUint16_t;
typedef _omInt16_t _omShort;
typedef _omUint16_t _omUshort;
#endif /* HAS_INT16 */

#ifdef HAS_INT32
typedef int _omInt32_t;
typedef unsigned int _omUint32_t;
typedef _omInt32_t _omInt;
typedef _omUint32_t _omUint;
#endif /* HAS_INT32 */

#ifdef HAS_INT64
typedef long long int _omInt64_t;
typedef unsigned long long int _omUint64_t;
typedef _omInt64_t _omLonglongint;
typedef _omUint64_t _omUlonglongint;
#endif /* HAS_INT64 */

#if SIZEOF_VOID_P == 8
typedef unsigned long long _omAddrInt_t;
#elif SIZEOF_VOID_P == 4
typedef unsigned int _omAddrInt_t;
#else
/* error, sizeof_void_p is neither 4 nor 8.  */
#endif

#if defined(LIST_NEXT)
#undef LIST_NEXT
#endif /* LIST_NEXT */

#if defined(MIN)
#undef MIN
#endif /* MIN */

#if defined(MAX)
#undef MAX
#endif /* MAX */

#endif /* _EXC_PLATFORM_H */
