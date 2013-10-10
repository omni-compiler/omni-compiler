/* 
 * $Id: nata_util.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_UTIL_H__
#define __NATA_UTIL_H__


#include <nata/nata_types.h>


#if defined(__cplusplus)
#define _BOOL_  bool
extern "C" {
#else
typedef enum {
    false = 0,
    true
} _BOOL_;
#define NATA_BOOL_DEFINED
#endif /* __cplusplus */



/*
 * Networking and IPC
 */


extern _BOOL_		nata_ConnectionSetup(in_addr_t addr, int port,
                                             FILE **inFdPtr,
                                             FILE **outFdPtr);
extern _BOOL_		nata_ConnectionSetup6(struct in6_addr *v6addrPtr,
                                              int port, FILE **inFdPtr,
                                              FILE **outFdPtr);
extern char *		nata_DumpIPv6Addr6(struct in6_addr *addrPtr);

extern char *		nata_GetHostOfIPAddress(in_addr_t addr);
extern char *		nata_GetHostOfIPAddress6(struct in6_addr *v6addrPtr);
extern in_addr_t	nata_GetLocalIPAddress(void);
extern _BOOL_		nata_GetLocalIPAddress6(struct in6_addr *v6addrPtr);
extern in_addr_t        nata_GetIPAddressOfHost(const char *host);
extern _BOOL_		nata_GetIPAddressOfHost6(const char *hostPtr,
                                                 struct in6_addr *v6addrPtr);
extern in_addr_t	nata_GetPeernameOfSocket(socket_handle_t sock,
                                                 uint16_t *portPtr);
extern _BOOL_		nata_GetPeernameOfSocket6(socket_handle_t sock,
                                                  struct in6_addr *v6addrPtr,
                                                  uint16_t *portPtr);
extern in_addr_t	nata_GetNameOfSocket(socket_handle_t sock,
                                             uint16_t *portPtr);
extern _BOOL_		nata_GetNameOfSocket6(socket_handle_t sock,
                                              struct in6_addr *addrPtr,
                                              uint16_t *portPtr);
extern int		nata_GetPortOfService(const char *name);

extern socket_handle_t	nata_ConnectPort(in_addr_t addr, int port);
extern socket_handle_t	nata_ConnectPort6(struct in6_addr *v6addrPtr,
                                          int port);
extern socket_handle_t	nata_BindPort(in_addr_t, int port);
extern socket_handle_t	nata_BindPort6(struct in6_addr *v6addrPtr,
                                       int port);
extern socket_handle_t	nata_AcceptPeer(socket_handle_t fd, in_addr_t *pAPtr,
                                        char **hnPtr, uint16_t *pPPtr);
extern socket_handle_t	nata_AcceptPeer6(socket_handle_t sock,
                                         struct in6_addr *v6addrPtr,
                                         char **hnPtr,
                                         uint16_t *portPtr);



/*
 * I/O and Network XDR
 */


#define nata_HandleToFileDescriptor(hndl, mode)	_hndl2int(handle, mode)


extern _BOOL_		nata_GetFILEPair(socket_handle_t fd,
                                         FILE **inFdPtr,
                                         FILE **outFdPtr);

extern _BOOL_		nata_IsNonBlock(socket_handle_t fd);

extern siolen_t		nata_RelayData(int src, int dst);
#define NATA_RELAY_CLOSED     0
#define NATA_RELAY_READ_FAIL  -1
#define NATA_RELAY_WRITE_FAIL -2

extern siolen_t		nata_ReadInt8(int fd, int8_t *buf, iolen_t len);
extern siolen_t		nata_WriteInt8(int fd, const int8_t *buf,
                                       iolen_t len);

extern siolen_t		nata_ReadInt16(int fd, int16_t *sPtr, iolen_t sLen);
extern siolen_t		nata_WriteInt16(int fd, const int16_t *sPtr,
                                        iolen_t sLen);

extern siolen_t		nata_ReadInt32(int fd, int32_t *iPtr, iolen_t iLen);
extern siolen_t		nata_WriteInt32(int fd, const int32_t *iPtr,
                                        iolen_t iLen);
extern siolen_t		nata_ReadInt64(int fd, int64_t *llPtr, iolen_t llLen);
extern siolen_t		nata_WriteInt64(int fd, const int64_t *llPtr,
                                        iolen_t llLen);

extern _BOOL_		nata_waitReadable(int fd, int64_t uSec);
extern int		nata_getMaxFileNo(void);



/*
 * Network XDR utils
 */


extern uint64_t		nata_ntohll(uint64_t n);
extern uint64_t		nata_htonll(uint64_t n);
extern void		nata_ntoh_in6_addr(struct in6_addr *srcPtr,
                                           struct in6_addr *dstPtr);
extern void		nata_hton_in6_addr(struct in6_addr *srcPtr,
                                           struct in6_addr *dstPtr);


/*
 * String/Text parsing
 */


extern int		nata_GetToken(char *buf, char **tokens, int max,
                                      const char *delm);
extern char *		nata_TrimRight(const char *org,
                                       const char *trimChars);

extern _BOOL_		nata_ParseInt32ByBase(const char *str, int32_t *val,
                                              int base);
extern _BOOL_		nata_ParseInt32(const char *str, int32_t *val);

extern _BOOL_		nata_ParseInt64ByBase(const char *str,
                                              int64_t *val,
                                              int base);
extern _BOOL_		nata_ParseInt64(const char *str,
                                        int64_t *val);

extern _BOOL_		nata_ParseFloat(const char *str, float *val);
extern _BOOL_		nata_ParseDouble(const char *str, double *val);
extern _BOOL_		nata_ParseLongDouble(const char *str,
                                             long double *val);

extern in_addr_t	nata_ParseAddressAndPort(const char *host_port,
                                                 int32_t *pPtr);
extern int		nata_ParseAddressAndPort6(const char *hostPortPtr,
                                                  struct in6_addr *v6addrPtr,
                                                  int32_t *portPtr);

extern in_addr_t	nata_ParseAddressAndMask(const char *host_mask,
                                                 uint32_t *mPtr);



/*
 * Misc. system utils
 */


extern int		nata_Mkdir(const char *path, mode_t mode,
                                   _BOOL_ doParent);
extern _BOOL_		nata_Daemonize(void);
extern void		nata_Yield(int usec);



#ifdef NATA_API_POSIX


/*
 * Tty manipulation
 */


extern _BOOL_		nata_TTYSetAttribute(int fd, struct termios *tPtr);
extern _BOOL_		nata_TTYGetAttribute(int fd, struct termios *tPtr);
extern _BOOL_		nata_TTYSetCanonicalMode(int ttyfd);
extern _BOOL_		nata_TTYSetRawMode(int ttyfd);
extern _BOOL_		nata_TTYSetNoEchoMode(int ttyfd);
extern _BOOL_		nata_TTYSetNoSignalMode(int ttyfd);
extern _BOOL_		nata_TTYSetBaudRate(int ttyfd, int rate);
extern _BOOL_		nata_TTYSetHardFlowMode(int ttyfd);

extern _BOOL_		nata_TTYGetPassword(char *pwBuf, size_t pwBufLen);

extern int		nata_PTYOpenMaster(char *slaveNameBuf,
                                           size_t slaveNameBufLen);

#endif /* NATA_API_POSIX */



#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* ! __NATA_UTIL_H__ */
