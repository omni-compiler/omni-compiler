#include <nata/nata_rcsid.h>
__rcsId("$Id: nata_util.cpp 86 2012-07-30 05:33:07Z m-hirano $")

#include <nata/libnata.h>


#define MAX_BACKLOG 10


#if defined(NATA_USE_PTHREAD) && defined(NATA_API_WIN32API)
pthread_t __iNvAlId_tHrEaD__ = { NULL, UINT_MAX };
#endif /* NATA_USE_PTHREAD && NATA_API_WIN32API */


typedef enum {
    ReadOnly = 0,
    WriteOnly,
    ReadWrite
} ioDirectionT;


typedef struct {
    ioDirectionT ioDir;
    int mode;
    const char *modeStr;
} ioModeT;



static bool
setTcpNoDelay(socket_handle_t fd, bool onoff) {
    socklen_t val = (onoff == true) ? (socklen_t)1 : (socklen_t)0;
    if (setsockopt(fd,
                   IPPROTO_TCP, TCP_NODELAY,
                   (sockopt_val_t)&val, sizeof(val)) < 0) {
        perror("setsockopt");
        return false;
    }
    return true;
}


static FILE *
doFDopen(int fd, const char *mode) {
    FILE *ret = fdopen(fd, mode);
    if (ret != NULL) {
        (void)setvbuf(ret, NULL, _IONBF, 0);
    }
    return ret;
}


bool
nata_GetFILEPair(socket_handle_t fd, FILE **inFdPtr, FILE **outFdPtr) {
    int oFd = __hndl2int(fd, O_RDWR | O_BINARY);
    if (oFd < 0) {
        return false;
    }

    if (inFdPtr != NULL) {
    *inFdPtr = fdopen(oFd, STDIO_RMODE);
        if (*inFdPtr == NULL) {
            if (outFdPtr != NULL) {
                *outFdPtr = NULL;
            }
            return false;
        }
    }
    if (outFdPtr != NULL) {
        int wFd = dup(oFd);
        if (wFd < 0) {
            goto Fail;
        }
        *outFdPtr = doFDopen(wFd, STDIO_WMODE);
        if (*outFdPtr == NULL) {
            Fail:
            if (*inFdPtr != NULL) {
                (void)fclose(*inFdPtr);
                    *inFdPtr = NULL;
            }
            if (wFd >= 0) {
                (void)closeFd(wFd);
            }
            return false;
        }
    }
    return true;
}


bool
nata_ConnectionSetup(in_addr_t addr, int port,
                     FILE **inFdPtr, FILE **outFdPtr) {
    bool ret = false;
    socket_handle_t fd = nata_ConnectPort(addr, port);
    FILE *inFd = NULL;
    FILE *outFd = NULL;

    if (isValidSocket(fd) != true) {
        return false;
    }
    if (nata_GetFILEPair(fd, &inFd, &outFd) != true) {
        goto Done;
    }
    if (inFd == NULL || outFd == NULL) {
        goto Done;
    }

    if (inFdPtr != NULL) {
        *inFdPtr = inFd;
    }
    if (outFdPtr != NULL) {
        *outFdPtr = outFd;
    }
    ret = true;

    Done:

    if (ret == true) {
        return true;
    }

    if (inFd != NULL) {
        (void)fclose(inFd);
    }
    if (outFd != NULL) {
        (void)fclose(outFd);
    }
    if (isValidSocket(fd) == true) {
        (void)closesocket(fd);
    }

    return ret;
}


bool
nata_ConnectionSetup6(struct in6_addr *v6addrPtr, int port,
                      FILE **inFdPtr, FILE **outFdPtr) {
    bool ret = false;
    socket_handle_t fd = nata_ConnectPort6(v6addrPtr, port);
    FILE *inFd = NULL;
    FILE *outFd = NULL;

    if (isValidSocket(fd) != true) {
        return false;
    }
    if (nata_GetFILEPair(fd, &inFd, &outFd) != true) {
        goto Done;
    }
    if (inFd == NULL || outFd == NULL) {
        goto Done;
    }

    if (inFdPtr != NULL) {
        *inFdPtr = inFd;
    }
    if (outFdPtr != NULL) {
        *outFdPtr = outFd;
    }
    ret = true;

    Done:

    if (ret == true) {
        return true;
    }

    if (inFd != NULL) {
        (void)fclose(inFd);
    }
    if (outFd != NULL) {
        (void)fclose(outFd);
    }
    if (isValidSocket(fd) == true) {
        (void)closesocket(fd);
    }

    return ret;
}


char *
nata_DumpIPv6Addr6(struct in6_addr *v6addrPtr) {
    size_t idx;
    size_t sz = sizeof(struct in6_addr) / sizeof(uint16_t);
    char *resPtr = NULL;
    char hBuf[INET6_ADDRSTRLEN];
    struct in6_addr v6addr;

    if (v6addrPtr == NULL) {
        resPtr = NULL;
        goto END;
    }

    for (idx = 0; idx < sz; idx += sizeof(uint16_t)) {
        v6addr.s6_addr16[idx] = v6addrPtr->s6_addr16[idx + 1];
        v6addr.s6_addr16[idx + 1] = v6addrPtr->s6_addr16[idx];
    }
    snprintf(hBuf, INET6_ADDRSTRLEN, "%x:%x:%x:%x:%x:%x:%x:%x",
             v6addr.s6_addr16[0], v6addr.s6_addr16[1],
             v6addr.s6_addr16[2], v6addr.s6_addr16[3],
             v6addr.s6_addr16[4], v6addr.s6_addr16[5],
             v6addr.s6_addr16[6], v6addr.s6_addr16[7]);
    resPtr = strdup(hBuf);


    END:
    return resPtr;
}




char *
nata_GetHostOfIPAddress(in_addr_t addr) {
    /*
     * addr must be Host Byte Order.
     */
    char hostBuf[4096];
    in_addr_t caddr = htonl(addr);
#if !defined(NATA_OS_CYGWIN) && !defined(NATA_OS_NETBSD) && !defined(NATA_API_WIN32API)
    struct hostent *h = gethostbyaddr((void *)&caddr,
                                      sizeof(in_addr_t), AF_INET);
#else
    struct hostent *h = gethostbyaddr((const char *)&caddr,
                                      sizeof(in_addr_t), AF_INET);
#endif /* ! NATA_OS_CYGWIN && ! NATA_OS_NETBSD && ! NATA_API_WIN32API */
    if (h != NULL) {
        return strdup((char *)(h->h_name));
    } else {
        sprintf(hostBuf, "%d.%d.%d.%d",
                (int)((addr & 0xff000000) >> 24),
                (int)((addr & 0x00ff0000) >> 16),
                (int)((addr & 0x0000ff00) >> 8),
                (int)(addr & 0x000000ff));
        return strdup(hostBuf);
    }
}


char *
nata_GetHostOfIPAddress6(struct in6_addr *v6addrPtr) {
    char  *resPtr = NULL;

    if (v6addrPtr == NULL) {
        resPtr = NULL;
        goto END;
    } else {
#ifdef NATA_API_POSIX
        char  hostBuf[INET6_ADDRSTRLEN];
        struct in6_addr addr;

        nata_hton_in6_addr(v6addrPtr, &addr);

        if (inet_ntop(AF_INET6, &addr, hostBuf, (uint32_t)INET6_ADDRSTRLEN)) {
            resPtr = strdup(hostBuf);
            if (resPtr == NULL) {
                perror("strdup");
            }
        } else {
            resPtr = nata_DumpIPv6Addr6(v6addrPtr);
        }
#else
        resPtr = nata_DumpIPv6Addr6(v6addrPtr);
#endif /* NATA_API_POSIX */
    }

    END:
    return resPtr;
}


in_addr_t
nata_GetLocalIPAddress(void) {
    static in_addr_t localAddr = (in_addr_t)(0xffffffff);
    static char hostBuf[4096];
    if (0xffffffff == localAddr) {
        if (gethostname(hostBuf, 4096) == 0) {
            localAddr = nata_GetIPAddressOfHost(hostBuf);
        }
    }
    return localAddr;
}


bool
nata_GetLocalIPAddress6(struct in6_addr *v6addrPtr) {
    bool  resBool;

    if (v6addrPtr == NULL) {
        resBool = false;
    } else {
        struct in6_addr v6addrNBO = in6addr_loopback;
        nata_ntoh_in6_addr(&v6addrNBO, v6addrPtr);
        resBool  = true;
    }
    return resBool;
}


in_addr_t
nata_GetIPAddressOfHost(const char *host) {
    struct hostent *h = gethostbyname(host);
    if (h != NULL) {
        return (in_addr_t)ntohl(*((uint32_t *)(h->h_addr)));
    } else {
        /*
         * maybe host is XXX.XXX.XXX.XXX format.
         */
        return (in_addr_t)ntohl(inet_addr(host));
    }
}


bool
nata_GetIPAddressOfHost6(const char *hostPtr, struct in6_addr *v6addrPtr) {
    int res;
    bool resBool;
    struct addrinfo hints, *addrInfoPtr = NULL;
    struct in6_addr v6addr;

    if (hostPtr == NULL || strchr(hostPtr, ':') == NULL) {
        resBool = false;
        goto END;
    }

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET6;
    hints.ai_socktype = SOCK_STREAM;

    res = getaddrinfo(hostPtr, NULL, &hints, &addrInfoPtr);
    if (res != 0) {
        /*
         * maybe host is XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX:XXXX
         * format. Invoke getaddrinfo() again with a slight hint.
         */
        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_family = AF_INET6;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_flags = AI_NUMERICHOST;
        res = getaddrinfo(hostPtr, NULL, &hints, &addrInfoPtr);
    }        
    if (res != 0) {
#ifdef NATA_API_POSIX
        struct in6_addr tmpAddr;

        if (inet_pton(AF_INET6, hostPtr, &tmpAddr) != 1) {
            perror("inet_pton");
            resBool = false;
            goto END;
        }
        nata_ntoh_in6_addr(&tmpAddr, &v6addr);
#else
        goto END;
#endif /* NATA_API_POSIX */
    } else {
        struct sockaddr_in6 sin6;

        if (addrInfoPtr->ai_addrlen != sizeof(struct sockaddr_in6)) {
            resBool = false;
            goto END;
        }
        sin6 = *((struct sockaddr_in6*)addrInfoPtr->ai_addr);
        nata_ntoh_in6_addr(&(sin6.sin6_addr), &v6addr);
    }
    *v6addrPtr = v6addr;
    resBool = true;

    END:
    if (addrInfoPtr != NULL) {
        freeaddrinfo(addrInfoPtr);
    }
    return resBool;
}


in_addr_t
nata_GetPeernameOfSocket(socket_handle_t sock, uint16_t *portPtr) {
    struct sockaddr_in sin;
    socklen_t slen = sizeof(sin);

    if (getpeername(sock, (struct sockaddr *)&sin, &slen) != 0) {
        perror("getpeername");
        if (portPtr != NULL) {
            *portPtr = 0;
        }
        return 0;
    }
    if (portPtr != NULL) {
        *portPtr = (uint16_t)ntohs((uint16_t)sin.sin_port);
    }
    return ntohl(sin.sin_addr.s_addr);
}


bool
nata_GetPeernameOfSocket6(socket_handle_t sock,
                          struct in6_addr *v6addrPtr,
                          uint16_t *portPtr) {

    bool  resBool;
    struct sockaddr_in6  sin6;
    socklen_t  sa6len = sizeof(struct sockaddr_in6);

    if (v6addrPtr == NULL && portPtr == NULL) {
        resBool = false;
        goto END;
    }

    if (getpeername(sock, (struct sockaddr *)&sin6, &sa6len) != 0) {
        perror("getpeername");
        resBool = false;
        goto END;
    }
    if (v6addrPtr != NULL) {
        struct in6_addr addr;

        nata_ntoh_in6_addr(&sin6.sin6_addr, &addr);
        *v6addrPtr = addr;
    }
    if (portPtr != NULL) {
        *portPtr = (uint16_t)ntohs((uint16_t)sin6.sin6_port);
    }
    resBool = true;


    END:
    return resBool;
}


in_addr_t
nata_GetNameOfSocket(socket_handle_t sock, uint16_t *portPtr) {
    struct sockaddr_in sin;
    socklen_t slen = sizeof(sin);

    if (getsockname(sock, (struct sockaddr *)&sin, &slen) != 0) {
        perror("getsockname");
        if (portPtr != NULL) {
            *portPtr = 0;
        }
        return 0;
    }
    if (portPtr != NULL) {
        *portPtr = (uint16_t)ntohs((uint16_t)sin.sin_port);
    }
    return ntohl(sin.sin_addr.s_addr);
}


bool
nata_GetNameOfSocket6(socket_handle_t sock,
                      struct in6_addr *v6addrPtr,
                      uint16_t *portPtr) {

    bool resBool;
    struct sockaddr_in6 sin6;
    socklen_t sa6len = sizeof(struct sockaddr_in6);

    if (v6addrPtr == NULL && portPtr == NULL) {
        resBool = false;
        goto END;
    }

    if (getsockname(sock, (struct sockaddr *)&sin6, &sa6len) != 0) {
        perror("getsockname");
        resBool = false;
        goto END;
    }
    if (sa6len != sizeof(struct sockaddr_in6)) {
        resBool = false;
        goto END;
    }
    if (v6addrPtr != NULL) {
        struct in6_addr addr;

        nata_ntoh_in6_addr(&sin6.sin6_addr, &addr);
        *v6addrPtr = addr;
    }
    if (portPtr != NULL) {
        *portPtr = (uint16_t)ntohs((uint16_t)sin6.sin6_port);
    }
    resBool = true;

    END:
    return resBool;
}


int
nata_GetPortOfService(const char *name) {
    struct servent *serv = getservbyname(name, "tcp");
    if (serv == NULL) {
        char *ePtr;
        long int iPort = strtol(name, &ePtr, 10);
        if (name == ePtr) {
            return 0;
        } else {
            return (int)iPort;
        }
    } else {
        return (int)(ntohs((uint16_t)(serv->s_port)));
    }
}


bool
nata_IsNonBlock(socket_handle_t fd) {
#if defined(NATA_API_POSIX)

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

#elif defined(NATA_API_WIN32API)
    (void)fd;
    return false;

#else

#error must not here.    

#endif /* NATA_API_POSIX, NATA_API_WIN32API */
}


void
nata_Yield(int usec) {
    struct timeval to;

    /*
     * Value is not a big deal. I just want to switch the process
     * context. 100 usec is 1/1000 of the typical HZ, no unix 
     * guarantee this time quantam B)
     */
    to.tv_sec = usec / 1000000;
    to.tv_usec = usec % 1000000;
    (void)select(0, NULL, NULL, NULL, &to);
}


socket_handle_t
nata_ConnectPort(in_addr_t addr, int port) {
    struct sockaddr_in sin;
    socket_handle_t sock;

    memset((void *)&sin, 0, sizeof(struct sockaddr_in));

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if (isValidSocket(sock) != true) {
        perror("socket");
        return INVALID_SOCKET;
    }

    sin.sin_family = AF_INET;
    sin.sin_port = htons((uint16_t)port);
    sin.sin_addr.s_addr = htonl(addr);

    if (connect(sock, (struct sockaddr *)&sin, sizeof(sin)) != 0) {
        int sErrno = errno;
        perror("connect");
        (void)closesocket(sock);
        errno = sErrno;
        return INVALID_SOCKET;
    }

    setTcpNoDelay(sock, true);

    return sock;
}


socket_handle_t
nata_ConnectPort6(struct in6_addr *v6addrPtr, int port) {
    socket_handle_t sock;
    struct in6_addr v6addr;
    struct sockaddr_in6 sin6;

    if (v6addrPtr == NULL) {
        return INVALID_SOCKET;        
    }

    memset((void *)&sin6, 0, sizeof(struct sockaddr_in6));

    sock = socket(AF_INET6, SOCK_STREAM, 0);
    if (isValidSocket(sock) != true) {
        perror("socket");
        return INVALID_SOCKET;
    }

    sin6.sin6_family = AF_INET6;
    sin6.sin6_port = htons((uint16_t)port);
    nata_hton_in6_addr(v6addrPtr, &v6addr);
    sin6.sin6_addr = v6addr;

    if (connect(sock, (struct sockaddr *)&sin6,
                sizeof(struct sockaddr_in6)) != 0) {
        int sErrno = errno;
        perror("connect");
        (void)closesocket(sock);
        errno = sErrno;
        return INVALID_SOCKET;
    }

    setTcpNoDelay(sock, true);

    return sock;
}


socket_handle_t
nata_BindPort(in_addr_t addr, int port) {
    struct sockaddr_in sin;
    socket_handle_t sock;
    int one = 1;

    memset((void *)&sin, 0, sizeof(struct sockaddr_in));

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if (isValidSocket(sock) != true) {
        perror("socket");
        return INVALID_SOCKET;
    }

    sin.sin_family = AF_INET;
    sin.sin_port = htons((uint16_t)port);
    sin.sin_addr.s_addr = htonl(addr);

    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (sockopt_val_t)&one,
                   sizeof(int)) != 0) {
        perror("setsockopt");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

    if (bind(sock, (struct sockaddr *)&sin, sizeof(sin)) != 0) {
        perror("bind");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

    if (listen(sock, MAX_BACKLOG) != 0) {
        perror("listen");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

    return sock;
}


socket_handle_t
nata_BindPort6(struct in6_addr *v6addrPtr, int port) {
    socket_handle_t sock;
    int one = 1;
    struct sockaddr_in6 sin6;
    struct in6_addr v6addr;

    if (v6addrPtr == NULL) {
        return INVALID_SOCKET;
    }

    memset((void *)&sin6, 0, sizeof(struct sockaddr_in6));

    sock = socket(PF_INET6, SOCK_STREAM, 0);
    if (isValidSocket(sock) != true) {
        perror("socket");
        return INVALID_SOCKET;
    }

    sin6.sin6_family = AF_INET6;
    sin6.sin6_port = htons((uint16_t)port);
    nata_hton_in6_addr(v6addrPtr, &v6addr);
    sin6.sin6_addr = v6addr;

    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   (sockopt_val_t)&one, sizeof(int)) != 0) {
        perror("setsockopt");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

#ifdef NATA_API_POSIX
    if (setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY,
                   (sockopt_val_t)&one, sizeof(int)) != 0) {
        perror("setsockopt");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }
#endif /* NATA_API_POSIX */

    if (bind(sock, (struct sockaddr *)&sin6,
             sizeof(struct sockaddr_in6)) != 0) {
        perror("bind");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

    if (listen(sock, MAX_BACKLOG) != 0) {
        perror("listen");
        (void)closesocket(sock);
        return INVALID_SOCKET;
    }

    return sock;
}


socket_handle_t
nata_AcceptPeer(socket_handle_t fd, in_addr_t *pAPtr,
                char **hnPtr, uint16_t *pPPtr) {
    struct sockaddr_in peer;
    socklen_t pLen;
    socket_handle_t ret;

    pLen = sizeof(peer);
    memset((void *)&peer, 0, sizeof(peer));
    ret = accept(fd, (struct sockaddr *)&peer, &pLen);
    if (isValidSocket(ret) != true) {
        perror("accept");
    } else {
        setTcpNoDelay(ret, true);

        if (pAPtr != NULL) {
            *pAPtr = ntohl(peer.sin_addr.s_addr);
        }
        if (pPPtr != NULL) {
            *pPPtr = (uint16_t)ntohs((uint16_t)peer.sin_port);
        }
        if (hnPtr != NULL) {
            *hnPtr = nata_GetHostOfIPAddress(ntohl(peer.sin_addr.s_addr));
        }
    }

    return ret;
}


socket_handle_t
nata_AcceptPeer6(socket_handle_t sock,
                 struct in6_addr *v6addrPtr,
                 char **hnPtr,
                 uint16_t *portPtr) {
    socket_handle_t ret;
    struct in6_addr tV6addr;
    struct sockaddr_in6 peer;
    socklen_t pLen = sizeof(struct sockaddr_in6);

    if (v6addrPtr == NULL) {
        return INVALID_SOCKET;
    }

    memset((void *)&peer, 0, sizeof(struct sockaddr_in6));

    ret = accept(sock, (struct sockaddr *)&peer, &pLen);
    if (isValidSocket(ret) != true) {
        perror("accept");
    } else {
        setTcpNoDelay(ret, true);

        if (v6addrPtr != NULL) {
            nata_ntoh_in6_addr(&peer.sin6_addr, &tV6addr);
            *v6addrPtr = tV6addr;
        }
        if (hnPtr != NULL) {
            nata_ntoh_in6_addr(&peer.sin6_addr, &tV6addr);
            *hnPtr = nata_GetHostOfIPAddress6(&tV6addr);
        }
        if (portPtr != NULL) {
            *portPtr = (uint16_t)ntohs((uint16_t)peer.sin6_port);
        }
    }

    return ret;
}


siolen_t
nata_RelayData(int src, int dst) {
    char copyBuf[65536];
    siolen_t rn;

    rn = read(src, copyBuf, 65536);
    if (rn < 0) {
        perror("read");
        return NATA_RELAY_READ_FAIL;
    } else if (rn == 0) {
        return NATA_RELAY_CLOSED;
    }

    if (dst >= 0) {
        siolen_t aWn = 0;
        siolen_t wn;

        while (rn > 0) {
            wn = write(dst, copyBuf + aWn, (iolen_t)(rn - aWn));
            if (wn < 0) {
                perror("write");
                return NATA_RELAY_WRITE_FAIL;
           }
            aWn += wn;
            rn -= wn;
        }
        return aWn;
    } else {
        return rn;
    }
}



int
nata_GetToken(char *buf, char **tokens, int max, const char *delm) {
    int n = 0;
    int nonDelm = 0;

    while (*buf != '\0' && n < max) {
        while (strchr(delm, (int)*buf) != NULL && *buf != '\0') {
            buf++;
        }
        if (*buf == '\0') {
            break;
        }
        tokens[n] = buf;

        nonDelm = 0;
        while (strchr(delm, (int)*buf) == NULL && *buf != '\0') {
            nonDelm++;
            buf++;
        }
        if (*buf == '\0') {
            if (nonDelm > 0) {
                n++;
            }
            break;
        }
        *buf = '\0';
        n++;
        if (*(buf + 1) == '\0') {
            break;
        } else {
            buf++;
        }
    }

    return n;
}


char *
nata_TrimRight(const char *org, const char *trimChars) {
    char *buf = NULL;
    size_t len;
    char *st, *ed;

    if (isValidString(org) == false) {
        return NULL;
    }

    buf = strdup(org);
    len = strlen(buf);
    st = buf;
    ed = buf + len - 1;

    while (ed >= st) {
        if (strchr(trimChars, (int)*ed) != NULL) {
            *ed = '\0';
            ed--;
        } else {
            break;
        }
    }

    return buf;
}


siolen_t
nata_ReadInt8(int fd, int8_t *buf, iolen_t len) {
    siolen_t ret = 0;
    siolen_t n;

    while ((iolen_t)ret < len) {
        n = read(fd, (void *)(buf + ret), len - ret);
        if (n <= 0) {
            break;
        }
        ret += n;
    }

    return ret;
}


siolen_t
nata_WriteInt8(int fd, const int8_t *buf, iolen_t len) {
    siolen_t ret = 0;
    siolen_t n;

    while ((iolen_t)ret < len) {
        n = write(fd, (void *)(buf + ret), len - ret);
        if (n <= 0) {
            break;
        }
        ret += n;
    }

    return ret;
}


siolen_t
nata_ReadInt16(int fd, int16_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int16_t);
    int16_t val;

    while ((iolen_t)ret < sLen) {
        n = nata_ReadInt8(fd, (int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        sPtr[ret++] = (int16_t)ntohs(val);
    }

    return ret;
}


siolen_t
nata_WriteInt16(int fd, const int16_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int16_t);
    int16_t val;

    while ((iolen_t)ret < sLen) {
        val = htons(sPtr[ret]);
        n = nata_WriteInt8(fd, (const int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        ret++;
    }

    return ret;
}


siolen_t
nata_ReadInt32(int fd, int32_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int32_t);
    int32_t val;

    while ((iolen_t)ret < sLen) {
        n = nata_ReadInt8(fd, (int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        sPtr[ret++] = ntohl(val);
    }

    return ret;
}


siolen_t
nata_WriteInt32(int fd, const int32_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int32_t);
    int32_t val;

    while ((iolen_t)ret < sLen) {
        val = htonl(sPtr[ret]);
        n = nata_WriteInt8(fd, (const int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        ret++;
    }

    return ret;
}


siolen_t
nata_ReadInt64(int fd, int64_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int64_t);
    int64_t val;

    while ((iolen_t)ret < sLen) {
        n = nata_ReadInt8(fd, (int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        sPtr[ret++] = nata_ntohll(val);
    }

    return ret;
}


siolen_t
nata_WriteInt64(int fd, const int64_t *sPtr, iolen_t sLen) {
    siolen_t ret = 0;
    siolen_t n;
    iolen_t s = sizeof(int64_t);
    int64_t val;

    while ((iolen_t)ret < sLen) {
        val = nata_htonll(sPtr[ret]);
        n = nata_WriteInt8(fd, (const int8_t *)&val, s);
        if ((iolen_t)n != s) {
            break;
        }
        ret++;
    }

    return ret;
}


_BOOL_
nata_waitReadable(int fd, int64_t uSec) {
    _BOOL_ ret = false;
    int nSels;
    struct timeval to;
    struct timeval *tPtr = NULL;
    fd_set rFds;

    if (fd < 0) {
        errno = EBADF;
        perror("select");
        goto Done;
    }

    FD_ZERO(&rFds);

    if (uSec >= 0) {
        to.tv_sec = (long)(uSec / 1000000LL);
        to.tv_usec = (long)(uSec % 1000000LL);
        tPtr = &to;
    }

    FD_SET(fd, &rFds);

    nSels = select(fd + 1, &rFds, NULL, NULL, tPtr);
    if (nSels > 0) {
        ret = true;
    } else if (nSels < 0) {
        perror("select");
    }

    Done:
    return ret;
}


int
nata_getMaxFileNo(void) {
    struct rlimit rl;

    if (getrlimit(RLIMIT_NOFILE, &rl) < 0) {
        perror("getrlimit");
        return -INT_MAX;
    }
    return (int)(rl.rlim_cur);
}


_BOOL_
nata_Daemonize(void) {
#if defined(NATA_API_POSIX)

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        return false;
    } else if (pid == 0) {
        int i;

        (void)nata_safe_Setsid();

        for (i = 0; i < 64; i++) {
            (void)close(i);
        }

        i = open("/dev/zero", O_RDONLY);
        if (i >= 0 && i != 0) {
            (void)dup2(0, i);
            (void)close(i);
        }
        i = open("/dev/null", O_WRONLY);
        if (i >= 1 && i != 1) {
            (void)dup2(1, i);
            (void)close(i);
        }
        i = open("/dev/null", O_WRONLY);
        if (i >= 2 && i != 2) {
            (void)dup2(2, i);
            (void)close(i);
        }

        return true;
    } else {
        exit(0);
    }

#elif defined(NATA_API_WIN32API)

    return false;

#else

#error Unknown/Non-supported API.

#endif /* NATA_API_POSIX, NATA_API_WIN32API */

}


#define skipSpaces(s)                                   \
    while (*(s) != '\0' && isspace((int)*(s)) != 0) {   \
        (s)++;                                          \
    }

#define trimSpaces(b, s)                                \
    while ((s) >= (b) && isspace((int)*(s)) != 0) {     \
        *(s)-- = '\0';                                  \
    }


static bool
parseInt64ByBase(const char *str, int64_t *val, int base) {
    /*
     * str := 
     *	[[:space:]]*[\-\+][[:space:]]*[0-9]+[[:space:]]*[kKmMgGtTpP]
     */
    char *ePtr = NULL;
    bool ret = false;
    int64_t t = 1;
    char *buf = NULL;
    int64_t tmpVal;
    size_t len;
    char *endP = NULL;
    int64_t neg = 1;

    skipSpaces(str);
    switch ((int)str[0]) {
        case '-': {
            neg = -1;
            str++;
            break;
        }
        case '+': {
            str++;
            break;
        }
    }
    skipSpaces(str);

    buf = strdup(str);
    if (buf == NULL) {
        return false;
    }
    len = strlen(buf);
    if (len == 0) {
        return false;
    }
    endP = &(buf[len - 1]);
    trimSpaces(buf, endP);
    len = strlen(buf);

    if (base == 10) {
        bool doTrim = false;
        size_t lC = len - 1;

        switch ((int)(buf[lC])) {
            case 'k': case 'K': {
                t = 1024;
                doTrim = true;
                break;
            }
            case 'm': case 'M': {
                t = 1024 * 1024;
                doTrim = true;
                break;
            }
            case 'g': case 'G': {
                t = 1024 * 1024 * 1024;
                doTrim = true;
                break;
            }
            case 't': case 'T': {
                t = 1099511627776LL;	/* == 2 ^ 40 */
                doTrim = true;
                break;
            }
            case 'p': case 'P': {
                t = 1125899906842624LL;	/* == 2 ^ 50 */
                doTrim = true;
                break;
            }
            default: {
                if (isspace((int)buf[lC]) != 0) {
                    doTrim = true;
                }
                break;
            }
        }

        if (doTrim == true) {
            buf[lC] = '\0';
            endP = &(buf[lC - 1]);
            trimSpaces(buf, endP);
            len = strlen(buf);
        }
    }

    tmpVal = (int64_t)strtoll(buf, &ePtr, base);
    if (ePtr == (buf + len)) {
        ret = true;
        *val = tmpVal * t * neg;
    }

    (void)free(buf);
    return ret;
}


bool
nata_ParseInt32ByBase(const char *str, int32_t *val, int base) {
    int64_t val64;
    bool ret = false;

    if ((ret = parseInt64ByBase(str, &val64, base)) == true) {
        if (val64 > (int64_t)(INT_MAX) ||
            val64 < -(((int64_t)(INT_MAX) + 1LL))) {
            ret = false;
        } else {
            *val = (int32_t)val64;
        }
    }

    return ret;
}


bool
nata_ParseInt32(const char *str, int32_t *val) {
    bool ret = false;
    int32_t base = 10;
    int32_t neg = 1;

    skipSpaces(str);
    switch ((int)str[0]) {
        case '-': {
            neg = -1;
            str++;
            break;
        }
        case '+': {
            str++;
            break;
        }
    }
    skipSpaces(str);

    if (strncasecmp(str, "0x", 2) == 0 ||
        strncasecmp(str, "\\x", 2) == 0) {
        base = 16;
        str += 2;
    } else if (strncasecmp(str, "\\0", 2) == 0) {
        base = 8;
        str += 2;
    } else if (str[0] == 'H' || str[0] == 'h') {
        base = 16;
        str += 1;
    } else if (str[0] == 'B' || str[0] == 'b') {
        base = 2;
        str += 1;
    }

    ret = nata_ParseInt32ByBase(str, val, base);
    if (ret == true) {
        *val = *val * neg;
    }
    return ret;
}


bool
nata_ParseInt64ByBase(const char *str, int64_t *val, int base) {
    return parseInt64ByBase(str, val, base);
}


bool
nata_ParseInt64(const char *str, int64_t *val) {
    bool ret = false;
    int base = 10;
    int64_t neg = 1;

    skipSpaces(str);
    switch ((int)str[0]) {
        case '-': {
            neg = -1;
            str++;
            break;
        }
        case '+': {
            str++;
            break;
        }
    }
    skipSpaces(str);

    if (strncasecmp(str, "0x", 2) == 0 ||
        strncasecmp(str, "\\x", 2) == 0) {
        base = 16;
        str += 2;
    } else if (strncasecmp(str, "\\0", 2) == 0) {
        base = 8;
        str += 2;
    } else if (str[0] == 'H' || str[0] == 'h') {
        base = 16;
        str += 1;
    } else if (str[0] == 'B' || str[0] == 'b') {
        base = 2;
        str += 1;
    }

    ret = nata_ParseInt64ByBase(str, val, base);
    if (ret == true) {
        *val = *val * neg;
    }
    return ret;
}


bool
nata_ParseFloat(const char *str, float *val) {
    char *ePtr = NULL;
    bool ret = false;
    float tmp;
    size_t len;

    len = strlen(str);
    if (len == 0) {
        return false;
    }

    tmp = strtof(str, &ePtr);
    if (ePtr == (str + len)) {
        ret = true;
        *val = tmp;
    }

    return ret;
}


bool
nata_ParseDouble(const char *str, double *val) {
    char *ePtr = NULL;
    bool ret = false;
    double tmp;
    size_t len;

    len = strlen(str);
    if (len == 0) {
        return false;
    }

    tmp = strtod(str, &ePtr);
    if (ePtr == (str + len)) {
        ret = true;
        *val = tmp;
    }

    return ret;
}


bool
nata_ParseLongDouble(const char *str, long double *val) {
    char *ePtr = NULL;
    bool ret = false;
    long double tmp;
    size_t len;

    len = strlen(str);
    if (len == 0) {
        return false;
    }

#ifdef HAVE_STRTOLD
    tmp = strtold(str, &ePtr);
#else
    tmp = (long double)strtod(str, &ePtr);
#endif /* HAVE_STRTOLD */
    if (ePtr == (str + len)) {
        ret = true;
        *val = tmp;
    }

    return ret;
}


in_addr_t
nata_ParseAddressAndPort(const char *host_port, int32_t *pPtr) {
    char *tmp = strdup(host_port);
    char *clnPos = strchr(tmp, ':');
    int port = -1;
    in_addr_t ret = INADDR_ANY;

    if (clnPos == NULL) {
        ret = nata_GetIPAddressOfHost(tmp);
        port = -1;
    } else {
        *clnPos = '\0';
        if (strlen(tmp) == 0) {
            ret = INADDR_ANY;
        } else {
            ret = nata_GetIPAddressOfHost(tmp);
        }
        clnPos++;
        if (*clnPos == '\0') {
            port = -1;
        } else {
            int val = 0;
            if (nata_ParseInt32(clnPos, &val) == true) {
                port = val;
            } else {
                port = -1;
            }
        }
    }

    if (pPtr != NULL) {
        *pPtr = port;
    }

    (void)free(tmp);
    return ret;
}


int
nata_ParseAddressAndPort6(const char *hostPortPtr,
                          struct in6_addr *v6addrPtr,
                          int32_t *portPtr) {

    int  res = 0;
    char  *tmpHostPortPtr = strdup(hostPortPtr);
    char  *clnPosPtr = strchr(tmpHostPortPtr, ':');
    struct in6_addr  v6addr = in6addr_any;
    unsigned short  port = 0;

    if (clnPosPtr == NULL) {
        /*
         * Parse only an address.
         */
        res = nata_GetIPAddressOfHost6(hostPortPtr, &v6addr);
        if (res < 0) {
            res = -1;
            goto END;
        }
        res = 1;

    } else {
        /*
         * Parse an address.
         */
        *clnPosPtr = '\0';
        if (strlen(tmpHostPortPtr) != 0) {
            res = nata_GetIPAddressOfHost6(tmpHostPortPtr, &v6addr);
            if (res < 0) {
                res = -1;
                goto END;
            }
        }

        /*
         * Parse a port then.
         */
        clnPosPtr++;
        if (*clnPosPtr == '\0') {
            res = 1;
        } else {
            long  lVal = 0;
            char  *ePtr = NULL;

            errno = 0;
            lVal = strtol(clnPosPtr, &ePtr, 10);
            if (clnPosPtr == ePtr ||
                errno == ERANGE ||
                (0 > lVal || lVal > USHRT_MAX)) {

                res = 1;
            } else {
                port = (unsigned short) lVal;
                res = 0;
            }
        }
    }
    if (v6addrPtr != NULL) {
        *v6addrPtr = v6addr;
    }
    if (portPtr != NULL && res == 0) {
        *portPtr = port;
    }


    END:
    if (tmpHostPortPtr != NULL) {
        (void)free(tmpHostPortPtr);
    }
    return res;
}


in_addr_t
nata_ParseAddressAndMask(const char *host_mask, uint32_t *mPtr) {
    char *tmp = strdup(host_mask);
    char *slsPos = strchr(tmp, '/');
    uint32_t mask = 0;
    in_addr_t ret = INADDR_ANY;

    if (slsPos == NULL) {
        ret = nata_GetIPAddressOfHost(tmp);
    } else {
        *slsPos = '\0';
        if (strlen(tmp) == 0) {
            ret = INADDR_ANY;
        } else {
            ret = nata_GetIPAddressOfHost(tmp);
        }
        slsPos++;
        if (*slsPos != '\0') {
            int32_t val;
            if (nata_ParseInt32(slsPos, &val) == true && val >= 0) {
                if (val >= 32) {
                    mask = 0xffffffff;
                } else {
                    mask = ~((1 << (32 - val)) - 1);
                }
            }
        }
    }

    if (mPtr != NULL) {
        *mPtr = mask;
    }

    (void)free(tmp);
    return ret;
}


uint64_t
nata_ntohll(uint64_t n) {
#ifdef NATA_BIG_ENDIAN
    return n;
#else
    return 
        (((uint64_t)(ntohl((uint32_t)(n & 0xffffffff)))) << 32) +
        (uint64_t)(ntohl((uint32_t)((n >> 32) & 0xffffffff)));
#endif /* NATA_BIG_ENDIAN */
}


uint64_t
nata_htonll(uint64_t n) {
#ifdef NATA_BIG_ENDIAN
    return n;
#else
    return 
        (((uint64_t)(htonl((uint32_t)(n & 0xffffffff)))) << 32) +
        (uint64_t)(htonl((uint32_t)((n >> 32) & 0xffffffff)));
#endif /* NATA_BIG_ENDIAN */
}


void
nata_ntoh_in6_addr(struct in6_addr *srcPtr, struct in6_addr *dstPtr) {
#ifdef NATA_BIG_ENDIAN
    return;
#else
    dstPtr->s6_addr32[0] = ntohl(srcPtr->s6_addr32[0]);
    dstPtr->s6_addr32[1] = ntohl(srcPtr->s6_addr32[1]);
    dstPtr->s6_addr32[2] = ntohl(srcPtr->s6_addr32[2]);
    dstPtr->s6_addr32[3] = ntohl(srcPtr->s6_addr32[3]);
    return;
#endif /* NATA_BIG_ENDIAN */
}


void
nata_hton_in6_addr(struct in6_addr *srcPtr, struct in6_addr *dstPtr) {
#ifdef NATA_BIG_ENDIAN
    return;
#else
    dstPtr->s6_addr32[0] = htonl(srcPtr->s6_addr32[0]);
    dstPtr->s6_addr32[1] = htonl(srcPtr->s6_addr32[1]);
    dstPtr->s6_addr32[2] = htonl(srcPtr->s6_addr32[2]);
    dstPtr->s6_addr32[3] = htonl(srcPtr->s6_addr32[3]);
    return;
#endif /* NATA_BIG_ENDIAN */
}


int
nata_Mkdir(const char *path, mode_t mode, bool doParent) {
    struct stat stBuf;

#ifdef NATA_API_WIN32API
    (void)mode;
#endif /* NATA_API_WIN32API */

    if (isValidString(path) != true) {
        errno = ENOENT;
        return -1;
    }

    if (stat(path, &stBuf) == 0) {
        errno = EEXIST;
        return -1;
    }

    if (doParent == false) {
#if defined(NATA_API_POSIX) 
        return mkdir(path, mode);
#elif defined(NATA_API_WIN32API)
        return mkdir(path);
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
    } else {
        int ret = -1;
        int nTokens;
        char *tokens[4096];
        char *pBuf = strdup(path);
        int i;
        size_t tknLen;
        size_t consLen = sizeof(char) * (strlen(path) + 1);
        size_t curConsLen = 0;
        char *consPath = (char *)alloca(consLen);

        (void)memset((void *)consPath, 0, consLen);

        nTokens = nata_GetToken(pBuf, tokens, 4096, "/");

        mode |= 0700;

        for (i = 0; i < nTokens; i++) {
            if (i > 0) {
                consPath[curConsLen] = '/';
                curConsLen++;
            }
            tknLen = strlen(tokens[i]);
            (void)memcpy((void *)(consPath + curConsLen),
                         tokens[i],
                         tknLen);
            curConsLen += tknLen;
            consPath[curConsLen] = '\0';

            errno = 0;
            if (stat((const char *)consPath, &stBuf) == 0) {
                /*
                 * The path exists.
                 */
                if (S_ISDIR(stBuf.st_mode)) {
                    /*
                     * It's a directory. Continue.
                     */
                    continue;
                }
            } else {
                if (errno == ENOENT) {
                    if (
#if defined(NATA_API_POSIX) 
                        mkdir((const char *)consPath, mode)
#elif defined(NATA_API_WIN32API)
                        mkdir((const char *)consPath)
#else
#error Unknown/Non-supported API.
#endif /* NATA_API_POSIX, NATA_API_WIN32API */
                        < 0) {
                        goto Done;
                    }
                } else {
                    goto Done;
                }
            }
        }
        ret = 0;

        Done:
        freeIfNotNULL(pBuf);
        return ret;
    }
}



#ifdef NATA_API_POSIX


bool
nata_TTYSetAttribute(int fd, struct termios *tPtr) {
    return (tcsetattr(fd, TCSADRAIN, tPtr) == 0) ? true : false;
}


bool
nata_TTYGetAttribute(int fd, struct termios *tPtr) {
    return (tcgetattr(fd, tPtr) == 0) ? true : false;
}


bool
nata_TTYSetCanonicalMode(int ttyfd) {
    int i;
    struct termios new_ld;

    (void)memset((void *)&new_ld, 0, sizeof(new_ld));

    new_ld.c_ispeed = B38400;
    new_ld.c_ospeed = B38400;

#ifndef PENDIN
#define PENDIN 0
#endif /* !PENDIN */
    new_ld.c_lflag = (ICANON|ISIG|ECHO|ECHOE|ECHOK|IEXTEN|ECHOKE|ECHOCTL|
                      PENDIN);
    new_ld.c_iflag = (ICRNL|IXON|IXANY|IMAXBEL|BRKINT);
    new_ld.c_oflag = (OPOST|ONLCR|ONOCR|ONLRET);
    new_ld.c_cflag = (CREAD|CS8|HUPCL);

    for (i = 0; i < NCCS; i++) {
        new_ld.c_cc[i] = _POSIX_VDISABLE;
    }
    new_ld.c_cc[VINTR] = CTRL('c');
    new_ld.c_cc[VQUIT] = CTRL('\\');
    new_ld.c_cc[VERASE] = CTRL('h');
    new_ld.c_cc[VKILL] = CTRL('u');
    new_ld.c_cc[VEOF] = CTRL('d');
    new_ld.c_cc[VEOL] = CTRL('@');
    new_ld.c_cc[VEOL2] = CTRL('@');
    new_ld.c_cc[VSTART] = CTRL('q');
    new_ld.c_cc[VSTOP] = CTRL('s');
    new_ld.c_cc[VSUSP] = CTRL('z');
#if defined(VDSUSP)
    new_ld.c_cc[VDSUSP] = CTRL('y');
#endif /* VDSUSP */
    new_ld.c_cc[VREPRINT] = CTRL('r');
    new_ld.c_cc[VDISCARD] = CTRL('o');
    new_ld.c_cc[VWERASE] = CTRL('w');
    new_ld.c_cc[VLNEXT] = CTRL('v');

    new_ld.c_cc[VMIN] = 1;
    new_ld.c_cc[VTIME] = 0;

    return nata_TTYSetAttribute(ttyfd, &new_ld);
}


bool
nata_TTYSetRawMode(int ttyfd) {
    struct termios new_ld;

    if (nata_TTYGetAttribute(ttyfd, &new_ld) != true) {
        return false;
    }

#if 0
    new_ld.c_lflag = (ECHO|ECHOE|ECHOK|ECHOKE|ECHOCTL|PENDIN);
    new_ld.c_iflag = IXANY;
    new_ld.c_oflag = (OPOST|ONLCR|ONOCR|ONLRET);
#else
    new_ld.c_lflag = 0;
    new_ld.c_iflag = 0;
    new_ld.c_oflag = 0;
#endif
    new_ld.c_cflag = (CREAD|CS8|HUPCL);

    new_ld.c_cc[VMIN] = 1;
    new_ld.c_cc[VTIME] = 0;

    return nata_TTYSetAttribute(ttyfd, &new_ld);
}


bool
nata_TTYSetNoEchoMode(int ttyfd) {
    struct termios t;

    if (nata_TTYGetAttribute(ttyfd, &t) != true) {
        return false;
    }

    t.c_lflag &= ~ECHO;

    return nata_TTYSetAttribute(ttyfd, &t);
}


bool
nata_TTYSetNoSignalMode(int ttyfd) {
    struct termios t;
    int i;

    cc_t oVMIN = 0;
    cc_t oVTIME = 0;

    if (nata_TTYGetAttribute(ttyfd, &t) != true) {
        return false;
    }

    oVMIN = t.c_cc[VMIN];
    oVTIME = t.c_cc[VTIME];

    t.c_lflag &= ~ISIG;

    for (i = 0; i < NCCS; i++) {
        t.c_cc[i] = _POSIX_VDISABLE;
    }

    t.c_cc[VMIN] = oVMIN;
    t.c_cc[VTIME] = oVTIME;

    return nata_TTYSetAttribute(ttyfd, &t);
}


bool
nata_TTYSetBaudRate(int ttyfd, int rate) {
    struct termios t;

    if (nata_TTYGetAttribute(ttyfd, &t) != true) {
        return false;
    }

    t.c_ispeed = t.c_ospeed = rate;
    return nata_TTYSetAttribute(ttyfd, &t);
}


bool
nata_TTYSetHardFlowMode(int ttyfd) {
    struct termios t;

    if (nata_TTYGetAttribute(ttyfd, &t) != true) {
        return false;
    }

    t.c_cflag |= CRTSCTS;
    return nata_TTYSetAttribute(ttyfd, &t);
}


bool
nata_TTYGetPassword(char *pwBuf, size_t pwBufLen) {
    bool ret = false;
    char *tmp = (char *)alloca(pwBufLen);
    struct termios saveT;

    if (nata_TTYGetAttribute(0, &saveT) != true) {
        return false;
    }

    if (nata_TTYSetNoEchoMode(0) != true) {
        goto Done;
    }

    if (fgets(tmp, (int)pwBufLen, stdin) != NULL) {
        char *tmp2 = nata_TrimRight(tmp, "\n\r");
        size_t len;

        if (tmp2 == NULL) {
            fprintf(stderr, "Can't allocate a temporaly buffer.\n");
            goto Done;
        }

        len = strlen(tmp2);
        if (len >= pwBufLen) {
            fprintf(stderr, "Password/Passphrase too long.\n");
            (void)free(tmp2);
            goto Done;
        }

        (void)memcpy((void *)pwBuf, (void *)tmp2, len);
        pwBuf[len] = '\0';

        ret = true;
    }

    Done:
    (void)nata_TTYSetAttribute(0, &saveT);

    return ret;
}


int
nata_PTYOpenMaster(char *slaveNameBuf, size_t slaveNameBufLen) {
    int ret = open("/dev/ptmx", O_RDWR);

    if (ret < 0) {
        perror("open");
        return -1;
    }
    if (grantpt(ret) < 0) {
        perror("grantpt");
        (void)close(ret);
        return -1;
    }
    if (unlockpt(ret) < 0) {
        perror("unlockpt");
        (void)close(ret);
        return -1;
    }

#ifdef linux
    ptsname_r(ret, slaveNameBuf, slaveNameBufLen);
#else
    snprintf(slaveNameBuf, slaveNameBufLen, "%s", ptsname(ret));
#endif /* linux */
    return ret;
}


#endif /* NATA_API_POSIX */
