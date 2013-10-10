/* 
 * $Id: nata_macros.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_MACROS_H__
#define __NATA_MACROS_H__


#include <nata/nata_config.h>

#ifdef HAVE_PTHREAD_H

#include <pthread.h>

#define NATA_USE_PTHREAD

/*
 * Thread, Mutex, WaitCondition
 */

#if defined(NATA_API_POSIX)

#define INVALID_THREAD_ID	(pthread_t)-1

#elif defined(NATA_API_WIN32API)

extern pthread_t __iNvAlId_tHrEaD__;
#define INVALID_THREAD_ID	__iNvAlId_tHrEaD__

#endif /* NATA_API_POSIX, NATA_API_WIN32API */

#endif /* HAVE_PTHREAD_H */

#ifdef __GNUC__
#define __attr_format_printf__(x, y) \
    __attribute__ ((format(printf, x, y)))
#else
#define __attr_format_printf__(x, y) /**/
#endif /* __GNUC__ */

/*
 * BoundedBlockingQueue
 */
#define BOUNDEDBLOCKINGQUEUE_DEFAULT_MAX_LENGTH	128

/*
 * CircularBuffer
 */
#define CIRCULARBUFFER_DEFALUT_MAX_LENGTH	32

/*
 * Macro tricks
 */
#define macroStringify(x)	stringify(x)
#define stringify(x)	#x

/*
 * 64 bit value printing format
 */
#define printFormatLong		l
#define printFormatLongLong	ll

#if defined(NATA_API_POSIX)

/*
 * long long is kinda obsolete since we have int64_t.
 */
#if SIZEOF_INT64_T == SIZEOF_LONG_INT
#define printFormat64		printFormatLong
#elif SIZEOF_INT64_T == SIZEOF_LONG_LONG
#define printFormat64		printFormatLongLong
#endif /* SIZEOF_INT64_T == SIZEOF_LONG_INT ... */

#elif defined(NATA_API_WIN32API)

/*
 * MS runtime has no capability printing long int which size is 64 bit.
 */
#define printFormat64		I64

#endif /* NATA_API_POSIX, NATA_API_WIN32API */

#define PF64(c)	stringify(%) macroStringify(printFormat64) macroStringify(c)
#define PF64s(s, c) \
    stringify(%) macroStringify(s) macroStringify(printFormat64) \
    macroStringify(c)

#ifdef HAVE_PRINT_FORMAT_FOR_SIZE_T

#define printFormatSize_t	z

#else

#if SIZEOF_SIZE_T == SIZEOF_LONG_LONG || SIZEOF_SIZE_T == SIZEOF_INT64_T
#define printFormatSize_t	printFormat64
#elif SIZEOF_SIZE_T == SIZEOF_LONG_INT
#define printFormatSize_t	printFormatLong
#elif SIZEOF_SIZE_T == SIZEOF_INT
#define printFormatSize_t	d
#endif /* SIZEOF_SIZE_T == SIZEOF_LONG_LONG ... */

#endif /* HAVE_PRINT_FORMAT_FOR_SIZE_T */

#define PFSz(c)	stringify(%) macroStringify(printFormatSize_t) \
    macroStringify(c)
#define PFSzs(s, c) \
    stringify(%) macroStringify(s) macroStringify(printFormatSize_t) \
    macroStringify(c)

#define isValidString(x)	(((x) != NULL && *(x) != '\0') ? true : false)
#define freeIfNotNULL(ptr)	\
    if ((ptr) != NULL) { (void)free((void *)(ptr)); }
#define freeIfValidString(ptr)	\
    if (isValidString((ptr)) == true) { (void)free((void *)(ptr)); }
#if defined(__cplusplus)
#define deleteIfNotNULL(obj)	if ((obj) != NULL) { delete (obj); }
#endif /* __cplusplus */
#define booltostr(a)	((a) == true) ? "true" : "false"
#define isBitSet(A, B)	((((A) & (B)) == (B)) ? true : false)
#define isBitSet(A, B)	((((A) & (B)) == (B)) ? true : false)

#include <nata/nata_logger.h>

#ifdef __GNUC__
#define __PROC__	__PRETTY_FUNCTION__
#else
#define	__PROC__	__func__
#endif /* __GNUC__ */

#define nata_Msg(...) \
    nata_Log(log_Unknown, -1, __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#define nata_MsgDebug(level, ...) \
    nata_Log(log_Debug, (level), __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#define nata_MsgInfo(...) \
    nata_Log(log_Info, -1, __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#define nata_MsgWarning(...) \
    nata_Log(log_Warning, -1, __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#define nata_MsgError(...) \
    nata_Log(log_Error, -1, __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#define nata_MsgFatal(...) \
    nata_Log(log_Fatal, -1, __FILE__, __LINE__, __PROC__, __VA_ARGS__)

#ifdef errorExit
#undef errorExit
#endif /* errorExit */
#define errorExit(eCode, ...) {                 \
        nata_MsgError(__VA_ARGS__);             \
        exit(eCode);                            \
    }

#ifdef fatal
#undef fatal
#endif /* fatal */
#define fatal(...) {                            \
        nata_MsgFatal(__VA_ARGS__);             \
        abort();                                \
    }

#define dbgMsg(...) \
    nata_MsgDebug(1, __VA_ARGS__)


#include <nata/nata_perror.h>


#endif /* ! __NATA_MACROS_H__ */
