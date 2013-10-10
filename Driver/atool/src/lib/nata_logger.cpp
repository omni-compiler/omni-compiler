#include <nata/nata_rcsid.h>
__rcsId("$Id: nata_logger.cpp 86 2012-07-30 05:33:07Z m-hirano $")

#include <nata/libnata.h>

#include <nata/Mutex.h>
#include <nata/ScopedLock.h>
#include <nata/Thread.h>





static Mutex logLock;
static FILE *logFd = NULL;
static logDestinationT logDst = emit_Unknown;
static char *logArg = NULL;
static bool doMultiProcess = false;
static bool doDate = false;
static int dbgLevel = 0;


static const char * const logLevelStr[] = {
    "",
    "[DEBUG]",
    "[INFO ]",
    "[WARN ]",
    "[ERROR]",
    "[FATAL]",
    NULL
};


#ifdef NATA_API_POSIX
#include <syslog.h>

static int const syslogPriorities[] = {
    LOG_INFO,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARNING,
    LOG_ERR,
    LOG_CRIT
};
#endif /* NATA_API_POSIX */





static inline void
lockFd(FILE *fd, bool cmd) {
#ifdef NATA_API_POSIX
    if (fd != NULL) {
        struct flock fl;

        fl.l_type = (cmd == true) ? F_WRLCK : F_UNLCK;
        fl.l_start = 0;
        fl.l_whence = SEEK_SET;
        fl.l_len = 0;	/* Entire lock. */
        fl.l_pid = 0;

        (void)fcntl(fileno(fd), F_SETLKW, &fl);
    }
#else
    (void)fd;
    (void)cmd;
#endif /* NATA_API_POSIX */
}


static inline void
lock(void) {
    if (doMultiProcess == false) {
        logLock.lock();
    } else {
        if (logDst == emit_File) {
            lockFd(logFd, true);
        }
    }
}


static inline void
unlock(void) {
    if (doMultiProcess == false) {
        logLock.unlock();
    } else {
        if (logDst == emit_File) {
            lockFd(logFd, false);
        }
    }
}


static inline void
logFinal(void) {
    if (logDst == emit_File) {
        if (logFd != NULL) {
            lockFd(logFd, false);
            (void)fclose(logFd);
            logFd = NULL;
        }
    }
#ifdef NATA_API_POSIX
    else if (logDst == emit_Syslog) {
        closelog();
    }
#endif /* NATA_API_POSIX */
    freeIfNotNULL(logArg);
    logArg = NULL;
}


static inline bool
logInit(logDestinationT dst, const char *arg,
        bool multiProcess, bool date,
        int debugLevel) {
    bool ret = false;

    if (dst == emit_File || 
        dst == emit_Unknown) {
        int oFd = -INT_MAX;

        logFinal();

        if (isValidString(arg) == true && dst == emit_File) {
            oFd = open(arg, O_RDWR | O_CREAT | O_APPEND, 0600);
            if (oFd >= 0) {
                logFd = fdopen(oFd, "a+");
                if (logFd != NULL) {
                    ret = true;
                }
            }
        } else {
            logFd = NULL;	/* use stderr. */
            ret = true;
        }
    }
#ifdef NATA_API_POSIX
    else if (dst == emit_Syslog) {
        if (isValidString(arg) == true) {

            logFinal();

            openlog(arg, 0, LOG_USER);

            ret = true;
        }
    }
#endif /* NATA_API_POSIX */

    if (ret == true) {
        logDst = dst;
        freeIfNotNULL(logArg);
        logArg = (isValidString(arg) == true) ? strdup(arg) : NULL;
        doMultiProcess = multiProcess;
        doDate = date;
        dbgLevel = debugLevel;
    }

    return ret;
}


static inline bool
logReinit(void) {
    char *arg = (isValidString(logArg) == true) ? 
        strdup(logArg) : NULL;
    bool ret = logInit(logDst, arg, doMultiProcess, doDate, dbgLevel);

    freeIfNotNULL(arg);

    return ret;
}


static inline const char *
getLevelStr(logLevelT l) {
    if ((int)l >= (int)log_Unknown && (int)l <= (int)log_Fatal) {
        return logLevelStr[(int)l];
    }
    return NULL;
}


#ifdef NATA_API_POSIX
static inline int
getSyslogPriority(logLevelT l) {
    if ((int)l >= (int)log_Unknown && (int)l <= (int)log_Fatal) {
        return syslogPriorities[(int)l];
    }
    return LOG_INFO;
}
#endif /* NATA_API_POSIX */


static inline size_t
getDateStr(char *buf, size_t bufLen) {
    const char *fmt = "[%a %h %d %T %Z %Y]";
    time_t x = time(NULL);
    return strftime(buf, bufLen, fmt, localtime(&x));
}


static inline void
doLog(logLevelT l, const char *msg) {
    FILE *fd;
    int sErrno = errno;
    int oCanState;

#ifndef  NATA_API_POSIX
    (void)l;
#endif /* ! NATA_API_POSIX */

    if (doMultiProcess == false) {
        (void)pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &oCanState);
    }

    lock();

    if (logDst == emit_File ||
        logDst == emit_Unknown) {
        fd = (logFd != NULL) ? logFd : stderr;
        (void)fprintf(fd, "%s", msg);
        (void)fflush(fd);
    }
#ifdef NATA_API_POSIX
    else if (logDst == emit_Syslog) {
        int prio = getSyslogPriority(l);
        syslog(prio, "%s", msg);
    }
#endif /* NATA_API_POSIX */

    unlock();

    if (doMultiProcess == false) {
        (void)pthread_setcancelstate(oCanState, NULL);
    }

    errno = sErrno;
}





void
nata_Log(logLevelT lv,
         int debugLevel,
         const char *file,
         int line,
         const char *func,
         const char *fmt, ...) {
    if (lv != log_Debug ||
        (lv == log_Debug &&
         debugLevel <= dbgLevel)) {

        va_list args;
        char dateBuf[32];
        char msg[4096];
        size_t hdrLen;
        size_t leftLen;
        char thdInfoBuf[1024];
        uint32_t tId;
        const char *thdName = NULL;

        if (doDate == true) {
            getDateStr(dateBuf, sizeof(dateBuf));
        } else {
            dateBuf[0] = '\0';
        }

#if defined(NATA_API_POSIX)
        tId = (unsigned int)pthread_self();
#elif defined(NATA_API_WIN32API)
        tId = (unsigned int)GetCurrentThreadId();
#else
#error Unknown/Non-supported API.
#endif // NATA_API_POSIX, NATA_API_WIN32API

        thdName = Thread::findThreadName(tId);

#ifdef NATA_API_POSIX
#ifdef NATA_OS_LINUX
        tId = Thread::getLinuxThreadId();
#endif // NATA_OS_LINUX
#endif // NATA_API_POSIX

        if (isValidString(thdName) == true) {
            snprintf(thdInfoBuf, sizeof(thdInfoBuf), "[%u:%u:%s]",
                     (unsigned int)getpid(), 
                     (unsigned int)tId,
                     thdName);
        } else {
            snprintf(thdInfoBuf, sizeof(thdInfoBuf), "[%u:%u]",
                     (unsigned int)getpid(), 
                     (unsigned int)tId);
        }

        va_start(args, fmt);
        va_end(args);

        hdrLen = snprintf(msg, sizeof(msg),
                          "%s%s%s:%s:%d:%s: ",
                          dateBuf,
                          getLevelStr(lv),
                          thdInfoBuf,
                          file, line, func);

        /*
         * hdrLen indicates the buffer length WITHOUT '\0'.
         */
        leftLen = sizeof(msg) - hdrLen;
        if (leftLen > 1) {
            (void)vsnprintf(msg + hdrLen, leftLen -1, fmt, args);
        }

        doLog(lv, msg);
    }
}


bool
nata_InitializeLogger(logDestinationT dst,
                      const char *arg,
                      bool multiProcess,
                      bool date,
                      int debugLevel) {
    ScopedLock l(&logLock);
    return logInit(dst, arg, multiProcess, date, debugLevel);
}


bool
nata_ReinitializeLogger(void) {
    ScopedLock l(&logLock);
    return logReinit();
}


void
nata_FinalizeLogger(void) {
    ScopedLock l(&logLock);
    logFinal();
}


#ifdef NATA_API_POSIX
namespace LoggerStatics {


    static bool sIsInitialized = false;


    static void
    sChildAfterFork(void) {
        logLock.reinitialize();
    }


    class LoggerInitializer {
    public:
        LoggerInitializer(void) {
            if (sIsInitialized == false) {
                (void)pthread_atfork(NULL,
                                     NULL,
                                     sChildAfterFork);
                sIsInitialized = true;
            }
        }


    private:
        LoggerInitializer(const LoggerInitializer &obj);
        LoggerInitializer operator = (const LoggerInitializer &obj);
    };


    static LoggerInitializer sLi;
}
#endif // NATA_API_POSIX
