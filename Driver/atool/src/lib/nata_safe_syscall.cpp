#include <nata/nata_rcsid.h>
__rcsId("$Id: nata_safe_syscall.cpp 86 2012-07-30 05:33:07Z m-hirano $")





#define IN_NATA_SAFE_SYSCALL

#include <nata/nata_includes.h>
#include <nata/nata_macros.h>
#include <nata/nata_safe_syscall.h>


#ifdef NATA_API_POSIX


#if defined(__cplusplus)
#define _BOOL_  bool
#else
#ifndef NATA_BOOL_DEFINED
typedef enum {
    false = 0,
    true
} _BOOL_;
#endif /* ! NATA_BOOL_DEFINED */
#endif /* __cplusplus */



static _BOOL_
waitReadable(int fd) {
    fd_set rFds;
    int sel;
    _BOOL_ ret = false;

    FD_ZERO(&rFds);
    FD_SET(fd, &rFds);

    sel = nata_safe_Select(fd + 1, &rFds, NULL, NULL, NULL);
    if (sel > 0 && FD_ISSET(fd, &rFds)) {
	ret = true;
    }

    return ret;
}
	

static _BOOL_
waitWritable(int fd) {
    fd_set wFds;
    int sel;
    _BOOL_ ret = false;

    FD_ZERO(&wFds);
    FD_SET(fd, &wFds);

    sel = nata_safe_Select(fd + 1, NULL, &wFds, NULL, NULL);
    if (sel > 0 && FD_ISSET(fd, &wFds)) {
	ret = true;
    }

    return ret;
}


static _BOOL_
isFdNonBlocking(int fd) {
    int stat = fcntl(fd, F_GETFL, 0);
    if (stat < 0) {
	perror("fcntl");
	return false;
    } else {
	if (stat & O_NONBLOCK) {
	    return true;
	} else {
	    return false;
	}
    }
}


#ifndef NATA_OS_LINUX
static int64_t
tv2ll(struct timeval *toPtr) {
    int64_t ret;

    ret = (int64_t)(toPtr->tv_sec) * 1000000LL;
    ret += (int64_t)(toPtr->tv_usec);
    return ret;
}


static void
ll2tv(struct timeval *toPtr, int64_t uSec) {
    toPtr->tv_sec = (unsigned int)(uSec / 1000000LL);
    toPtr->tv_usec = (unsigned int)(uSec % 1000000LL);
}
#endif /* ! NATA_OS_LINUX */


int
nata_safe_Select(int maxFd,
                 fd_set *readFds, fd_set *writeFds, fd_set *exceptFds,
                 struct timeval *toPtr) {
    if (toPtr == NULL) {
	int ret;

	doSelSimple:

	errno = 0;
	ret = select(maxFd, readFds, writeFds, exceptFds, NULL);
	if (ret < 0 && errno == EINTR) {
	    goto doSelSimple;
	}

	return ret;
    } else {
#ifndef NATA_OS_LINUX
	struct timeval start, intr, rto;
	int64_t lStart, lIntr, lRemains;
	int s;

	rto = *toPtr;
	lRemains = tv2ll(&rto);

	doSel:
	errno = 0;
	gettimeofday(&start, NULL);
	lStart = tv2ll(&start);

	s = select(maxFd, readFds, writeFds, exceptFds, &rto);
	if (s < 0 && errno == EINTR) {
	    gettimeofday(&intr, NULL);
	    lIntr = tv2ll(&intr);
	    lRemains -= (lIntr - lStart);
	    if (lRemains > 0) {
		ll2tv(&rto, lRemains);
		goto doSel;
	    }
	}
#else
	int s;

	doSel:
	errno = 0;

	s = select(maxFd, readFds, writeFds, exceptFds, toPtr);
	if (s < 0 && errno == EINTR) {
	    goto doSel;
	}
#endif /* ! NATA_OS_LINUX */

	return s;
    }
}


ssize_t
nata_safe_Read(int fd, void *buf, size_t nBytes) {
    ssize_t ret = -INT_MAX;
    int sErrno;

    ReRead:
    errno = 0;
    sErrno = 0;
    ret = read(fd, buf, nBytes);
    sErrno = errno;
    if (ret < 0) {
	if (errno == EINTR) {
	    goto ReRead;
	} else if (errno == EAGAIN) {
	    if (isFdNonBlocking(fd) == true) {
		if (waitReadable(fd) == true) {
		    goto ReRead;
		} else {
		    errno = sErrno;
		    return ret;
		}
	    }
	}
    }
    return ret;
}


ssize_t
nata_safe_Write(int fd, void *buf, size_t nBytes) {
    ssize_t ret = -INT_MAX;
    int sErrno;

    ReWrite:
    errno = 0;
    sErrno = 0;
    ret = write(fd, buf, nBytes);
    sErrno = errno;
    if (ret < 0) {
	if (errno == EINTR) {
	    goto ReWrite;
	} else if (errno == EAGAIN) {
	    if (isFdNonBlocking(fd) == true) {
		if (waitWritable(fd) == true) {
		    goto ReWrite;
		} else {
		    errno = sErrno;
		    return ret;
		}
	    }
	}
    }
    return ret;
}


int
nata_safe_Connect(int fd, const struct sockaddr *saPtr, socklen_t saLen) {
    int ret = -INT_MAX;
    int sErrno;

    ReConnect:
    errno = 0;
    sErrno = 0;
    ret = connect(fd, saPtr, saLen);
    sErrno = errno;
    if (ret != 0) {
	if (errno == EINTR) {
	    goto ReConnect;
	} else if (errno == EINPROGRESS) {
	    if (isFdNonBlocking(fd) == true) {
		if (waitWritable(fd) == true) {
		    /*
		     * Check failure by getsockopt(2).
		     */
		    int err = 0;
		    int sLen = sizeof(int);
		    int st = getsockopt(fd, SOL_SOCKET, SO_ERROR, (void *)&err,
					(socklen_t *)&sLen);
		    if (st == 0) {
			errno = err;
			if (err == 0) {
			    return 0;
			} else {
			    (void)close(fd);
			    return ret;
			}
		    } else {
			errno = sErrno;
			(void)close(fd);
			return ret;
		    }
		} else {
		    errno = sErrno;
		    (void)close(fd);
		    return ret;
		}
	    } else {
		errno = sErrno;
		(void)close(fd);
		return ret;
	    }
	}
	return ret;
    }
    return ret;
}


int
nata_safe_Accept(int fd, struct sockaddr *saPtr, socklen_t *saLenPtr) {
    int ret;
    int sErrno;

    ReAccept:
    errno = 0;
    sErrno = 0;
    ret = accept(fd, saPtr, saLenPtr);
    sErrno = errno;
    if (ret != 0) {
	if (errno == EINTR) {
	    goto ReAccept;
	} else if (errno == EAGAIN) {
	    if (isFdNonBlocking(fd) == true) {
		if (waitReadable(fd) == true) {
		    goto ReAccept;
		} else {
		    errno = sErrno;
		    return ret;
		}
	    } else {
		errno = sErrno;
		return ret;
	    }
	}
	return ret;
    }
    return ret;
}


pid_t
nata_safe_Setsid(void) {
#if defined(NATA_OS_LINUX) || defined(NATA_OS_CYGWIN)
#define SETPGRP(x, y)	setpgid(x, y)
#else
#define SETPGRP(x, y)	setpgrp(x, y)
#endif /* NATA_OS_LINUX || NATA_OS_CYGWIN */
    pid_t ret = (pid_t)-1;
    int fd = open("/dev/tty", O_RDONLY);

    if (fd >= 0) {
	(void)close(fd);
	/*
	 * Has a control terminal.
	 */
	if ((ret = setsid()) < 0) {
	    (void)SETPGRP(0, getppid());
	    if ((ret = setsid()) < 0) {
#ifdef TIOCNOTTY
		int one = 1;
		fd = open("/dev/tty", O_RDONLY);
		if (fd >= 0) {
		    (void)ioctl(fd, TIOCNOTTY, &one);
		}
		(void)close(fd);
		(void)SETPGRP(0, getppid());
		ret = getpid();
#endif /* TIOCNOTTY */
	    }
	}
    } else {
	/*
	 * Has no control terminal.
	 */
	if ((ret = setsid()) < 0) {
	    (void)SETPGRP(0, getpid());
	    ret = getpid();
	}
    }
    return ret;
}




#endif /* NATA_API_POSIX */
