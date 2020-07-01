#include <string.h>
#include <stdarg.h>
#include "xmpco_internal.h"

/*****************************************\
  parameter variables
\*****************************************/

/*  Default values are defined in xmpco_params.h
 */
static unsigned _poolThreshold    = _XMPCO_default_poolThreshold;
static size_t   _localBufSize     = _XMPCO_default_localBufSize;
static BOOL     _isMsgMode        = _XMPCO_default_isMsgMode;
static BOOL     _isSafeBufferMode = _XMPCO_default_isSafeBufferMode;
static BOOL     _isSyncPutMode    = _XMPCO_default_isSyncPutMode;
static BOOL     _isEagerCommMode  = _XMPCO_default_isEagerCommMode;

static BOOL _isMsgMode_last;


/*****************************************\
  set functions
\*****************************************/
void _XMPCO_set_isSafeBufferMode(BOOL sw)    { _isSafeBufferMode = sw; }
void _XMPCO_set_isSyncPutMode(BOOL sw)       { _isSyncPutMode = sw; }
void _XMPCO_set_isEagerCommMode(BOOL sw)     { _isEagerCommMode = sw; }

void _XMPCO_set_poolThreshold(unsigned size)
{
  _poolThreshold = size;

  _XMPCO_debugPrint("set _poolThreshold = %u\n",
                          _poolThreshold);
}


void _XMPCO_set_localBufSize(unsigned size)
{
  unsigned sizeRU = ROUND_UP_MALLOC(size);

  _localBufSize = sizeRU;

  _XMPCO_debugPrint("set _localBufSize = %u\n",
                          _localBufSize);
}


void _XMPCO_set_isMsgMode(BOOL sw)
{
  _isMsgMode_last = _isMsgMode;

  switch (sw) {
  case FALSE:
  default:
    if (_XMPCO_get_isMsgMode())
      _XMPCO_debugPrint("Switch off _isMsgMode. Bye!\n");
    _isMsgMode = FALSE;
    return;

  case TRUE:
    if (_XMPCO_get_isMsgMode() == 0)
      _XMPCO_debugPrint("Switch on _isMsgMode.\n");
    _isMsgMode = TRUE;
    break;
  }
}

void _XMPCO_set_isMsgMode_quietly(BOOL sw)
{
  _isMsgMode_last = _isMsgMode;
  _isMsgMode = sw;
}

void _XMPCO_reset_isMsgMode()
{
  _isMsgMode = _isMsgMode_last;
}


/*****************************************\
  get functions
\*****************************************/
unsigned _XMPCO_get_poolThreshold()  { return _poolThreshold; }
size_t   _XMPCO_get_localBufSize()   { return _localBufSize; }

BOOL   _XMPCO_get_isMsgMode()        { return _isMsgMode; }
BOOL   _XMPCO_get_isSafeBufferMode() { return _isSafeBufferMode; }
BOOL   _XMPCO_get_isSyncPutMode()    { return _isSyncPutMode; }
BOOL   _XMPCO_get_isEagerCommMode()  { return _isEagerCommMode; }

