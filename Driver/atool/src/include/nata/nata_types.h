/*
 * $Id: nata_types.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_TYPES_H__
#define __NATA_TYPES_H__

#if defined(NATA_API_POSIX)

typedef void * sockopt_val_t;
typedef int socket_handle_t;
#define INVALID_SOCKET	-1
typedef size_t iolen_t;
typedef ssize_t siolen_t;
typedef time_t timeval_sec_t;
typedef suseconds_t timeval_usec_t;

#define isValidSocket(ret) (((ret) >= 0) ? true : false)
#ifndef O_BINARY
#define O_BINARY	0
#endif /* O_BINARY */
#define STDIO_WMODE	"w"
#define STDIO_RMODE	"r"
#define STDIO_RWMODE	"r+"
#define mO_RDONLY	O_RDONLY
#define mO_WRONLY	O_WRONLY
#define mO_RDWR		O_RDWR
#define __hndl2int(handle, mode)	(handle)
#define closesocket(handle)	close((handle))
#define closeFd(fd)		close((fd))

#elif defined(NATA_API_WIN32API)

typedef const char * sockopt_val_t;
typedef SOCKET socket_handle_t;
typedef unsigned int iolen_t;
typedef int siolen_t;
typedef long timeval_sec_t;
typedef long timeval_usec_t;

#define isValidSocket(ret) (((ret) != INVALID_SOCKET) ? true : false)
#define STDIO_WMODE	"wb"
#define STDIO_RMODE	"rb"
#define STDIO_RWMODE	"rb+"
#define mO_RDONLY	(O_RDONLY | O_BINARY)
#define mO_WRONLY	(O_WRONLY | O_BINARY)
#define mO_RDWR		(O_RDWR | O_BINARY)
#define __hndl2int(handle, mode)	_open_osfhandle((handle), (mode))
#define closeFd(fd)		_close((fd))
#define socket(pf, type, proto)	WSASocket((pf), (type), (proto), NULL, 0, 0)

#else

#error Unknown/Non-supported API.

#endif /* NATA_API_POSIX, NATA_API_WIN32API */

#endif /* __NATA_TYPES_H__ */
