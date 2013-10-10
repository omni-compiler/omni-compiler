/*
 * $Id: nata_win32api.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_WIN32API_H__
#define __NATA_WIN32API_H__

#ifdef NATA_API_WIN32API

#if !defined(_WIN32_WINNT)
#define _WIN32_WINNT	0x0501	/* Windows XP */
#endif /* ! _WIN32_WINNT */

#if !defined(__MSVCRT_VERSION__)
#define __MSVCRT_VERSION__	0x0601	/* msvcrt.dll 6.10 or later */
#endif /* ! __MSVCRT_VERSION__ */

#if defined(__MINGW32__) || defined(__MINGW64__)
#if defined(_NO_OLDNAMES)
#undef _NO_OLDNAMES
#endif /* _NO_OLDNAMES */
#endif /* __MINGW64__ || __MINGW32__ */

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif /* HAVE_INTTYPES_H */

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif /* HAVE_STDINT_H */

#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif /* HAVE_MALLOC_H */

#ifdef NATA_OS_WINDOWS64
#ifndef s6_addr

typedef struct in6_addr {
    union {
        uint8_t Byte[16];
        uint16_t Word[8];
        uint32_t Long[4];
    } u;
} IN6_ADDR, *PIN6_ADDR, *LPIN6_ADDR;

#define in_addr6        in6_addr

#define _S6_un          u
#define _S6_u8          Byte
#define s6_addr         _S6_un._S6_u8

#define s6_bytes        u.Byte
#define s6_words        u.Word
#define s6_longs	u.Long

#define s6_addr16	s6_words
#define s6_addr32	s6_longs

#else

#error you need to modify mingw header.

#endif /* ! s6_addr */
#endif /* NATA_OS_WINDOWS64 */

typedef uint32_t in_addr_t;

#if defined(NATA_OS_WINDOWS64)
#include <wspiapi.h>
#elif defined(NATA_OS_WINDOWS32)
#include <ws2tcpip.h>
#else
#error Not yet supported.
#endif /* NATA_OS_WINDOWS64, NATA_OS_WINDOWS32 */
#include <ws2spi.h>
#include <winsock2.h>
#include <windows.h>

#ifdef NATA_OS_WINDOWS32
#include <sys/timeb.h>
#endif /* NATA_OS_WINDOWS32 */

#if 0
INT WSAAPI	inet_pton(INT  Family, PCTSTR pszAddrString, PVOID pAddrBuf);
PCTSTR WSAAPI	inet_ntop(INT  Family, PVOID pAddr,
                          PTSTR pStringBuf, size_t StringBufSize);
#endif

#endif /* NATA_API_WIN32API */

#endif /* ! __NATA_WIN32API_H__ */
