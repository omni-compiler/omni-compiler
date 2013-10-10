/* 
 * $Id: nata_logger.h 86 2012-07-30 05:33:07Z m-hirano $
 */
#ifndef __NATA_LOGGER_H__
#define __NATA_LOGGER_H__


typedef enum {
    log_Unknown = 0,
    log_Debug,
    log_Info,
    log_Warning,
    log_Error,
    log_Fatal
} logLevelT;


typedef enum {
    emit_Unknown = 0,
    emit_File
#ifdef NATA_API_POSIX
    ,
    emit_Syslog
#else
#define emit_Syslog	emit_Unknown
#endif /* NATA_API_POSIX */
} logDestinationT;


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


extern bool	nata_InitializeLogger(logDestinationT dst,
                                      const char *arg,
                                      bool multiProcess,
                                      bool date,
                                      int debugLevel);

extern bool	nata_ReinitializeLogger(void);

extern void	nata_FinalizeLogger(void);


extern void	nata_Log(logLevelT logLevel,
                         int debugLevel,
                         const char *file,
                         int line,
                         const char *func,
                         const char *fmt, ...)
    __attr_format_printf__(6, 7);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* ! __NATA_LOGGER_H__ */
