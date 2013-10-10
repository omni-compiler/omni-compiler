/* 
 * $Id: nata_safe_syscall.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_SAFE_SYSCALL_H__
#define __NATA_SAFE_SYSCALL_H__


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


extern int			nata_safe_Select(int maxFd,
						 fd_set *readFds,
						 fd_set *writeFds,
						 fd_set *exceptFds,
						 struct timeval *toPtr);

extern ssize_t			nata_safe_Read(int fd,
					       void *buf, size_t nBytes);
extern ssize_t			nata_safe_Write(int fd,
						void *buf, size_t nBytes);

extern int			nata_safe_Connect(int fd,
						  const struct sockaddr *saPtr,
						  socklen_t saLen);
extern int			nata_safe_Accept(int fd,
						 struct sockaddr *saPtr,
						 socklen_t *saLenPtr);

extern pid_t			nata_safe_Setsid(void);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#ifndef IN_NATA_SAFE_SYSCALL

#define select	nata_safe_Select
#define read	nata_safe_Read
#define write	nata_safe_Write
#define connect	nata_safe_Connect
#define accept	nata_safe_Accept

#endif /* ! IN_NATA_SAFE_SYSCALL */

#endif /* ! __NATA_SAFE_SYSCALL_H__ */
