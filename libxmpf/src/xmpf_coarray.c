#include "xmpf_internal.h"

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)


/*****************************************\
  performance parameters
\*****************************************/

/* Threshold of memory size to share in the pool
 */
#define POOL_THRESHOLD (40*1024*1024)          // 40MB

/* Size of the communication buffer prepared for short communications
 * to avoid allocation and registration every communication time
 */
#define LOCAL_BUF_SIZE  (400000)               // ~400kB


/*****************************************\
  static vars and functions
\*****************************************/

static int _XMPF_coarrayMsg = 0;          // default: message off
static int _XMPF_coarrayMsg_last;         // for _XMPF_set/reset_coarrayMsg()

//static int _XMPF_coarrayErr = 0;          // default: aggressive error check off
static unsigned _XMPF_poolThreshold = POOL_THRESHOLD;
static size_t _XMPF_localBufSize = LOCAL_BUF_SIZE;
static BOOL _XMPF_isSafeBufferMode = FALSE;

static void _set_coarrayMsg(int sw)
{
  switch (sw) {
  case 0:
  default:
    if (_XMPF_coarrayMsg)
      _XMPF_coarrayDebugPrint("Switch _XMPF_coarrayMsg=0. Bye!\n");
    _XMPF_coarrayMsg = 0;
    return;

  case 1:
    if (_XMPF_coarrayMsg == 0)
      _XMPF_coarrayDebugPrint("Switch _XMPF_coarrayMsg=1.\n");
    _XMPF_coarrayMsg = 1;
    break;
  }
}


/*****************************************\
  static and extern functions
\*****************************************/

static void _set_poolThreshold(unsigned size);
static void _set_localBufSize(unsigned size);
static unsigned _envStringToBytes(char *str, char *envVarName);
static void _set_isSafeBufferMode(BOOL sw);

int _XMPF_get_coarrayMsg(void)
{
  return _XMPF_coarrayMsg;
}


void _XMPF_set_coarrayMsg(int sw)
{
  _XMPF_coarrayMsg_last = _XMPF_coarrayMsg;
  _XMPF_coarrayMsg = sw;
}

void _XMPF_reset_coarrayMsg(void)
{
  _XMPF_coarrayMsg = _XMPF_coarrayMsg_last;
}


static void _set_poolThreshold(unsigned size)
{
  _XMPF_poolThreshold = size;

  _XMPF_coarrayDebugPrint("set _XMPF_poolThreshold = %u\n",
                          _XMPF_poolThreshold);
}


static void _set_localBufSize(unsigned size)
{
  unsigned sizeRU = ROUND_UP_BOUNDARY(ROUND_UP_UNIT(size));

  _XMPF_localBufSize = sizeRU;

  _XMPF_coarrayDebugPrint("set _XMPF_localBufSize = %u\n",
                          _XMPF_localBufSize);
}


unsigned XMPF_get_poolThreshold(void)
{
  return _XMPF_poolThreshold;
}

size_t XMPF_get_localBufSize(void)
{
  return _XMPF_localBufSize;
}


void _set_isSafeBufferMode(BOOL sw)
{
  _XMPF_isSafeBufferMode = sw;
}

BOOL XMPF_isSafeBufferMode(void)
{
  return _XMPF_isSafeBufferMode;
}


/*****************************************\
  hidden API
\*****************************************/

/*
 *  hidden subroutine interface,
 *   which can be used in the user program
 */
void xmpf_coarray_msg_(int *sw)
{
  _set_coarrayMsg(*sw);
}


/*****************************************\
  initialization called in xmpf_main
\*****************************************/

/*  1. set static variable _this_image and _num_nodes
 *  2. read environment variable XMPF_COARRAY_MSG and set _XMPF_coarrayMsg
 *     usage: <v1><d><v2><d>...<vn>
 *        <vk>  value for image index k
 *        <d>   delimiter ',' or ' '
 *  3. read environmrnt variable XMPF_COARRAY_POOL and set
 *     _XMPF_poolThreshold
 *     usage: [0-9]+[kKmMgG]?
 */
void _XMPF_coarray_init(void)
{
  /*
   *  set who-am-i
   */
  _XMPF_set_this_image_initial();
  _XMPF_set_num_images_initial();

  /*
   * read environment variables
   */
  char *tok, *work, *env1, *env2, *env3, *env4;
  int i;
  char delim[] = ", ";
  unsigned len;

  env1 = getenv("XMPF_COARRAY_MSG");
  if (env1 != NULL) {
    work = strdup(env1);
    tok = strtok(work, delim);
    for (i = 1; tok != NULL; i++, tok = strtok(NULL, delim)) {
      if (_XMPF_this_image_current() == i)
        _set_coarrayMsg(atoi(tok));
    }
  }

  env2 = getenv("XMPF_COARRAY_POOL");
  if (env2 != NULL) {
    len = _envStringToBytes(env2, "XMPF_COARRAY_POOL");
    if (len != 0)
      _set_poolThreshold(len);
  }
    
  env3 = getenv("XMPF_COARRAY_BUF");
  if (env3 != NULL) {
    len = _envStringToBytes(env3, "XMPF_COARRAY_BUF");
    if (len != 0)
      _set_localBufSize(len);
  }
    
  env4 = getenv("XMPF_COARRAY_SAFE");
  if (env4 != NULL) {
    _set_isSafeBufferMode(atoi(env4));
  }
    
  if (_XMPF_coarrayMsg || (env2&&*env2) || (env3&&*env3) || (env4&&*env4)) {
    _XMPF_set_coarrayMsg(TRUE);

    _XMPF_coarrayDebugPrint("Execution time environment\n"
                            "   communication layer  :  %s (%u-byte boundary)\n"
                            "   environment vars     :  XMPF_COARRAY_MSG=%s\n"
                            "                           XMPF_COARRAY_POOL=%s\n"
                            "                           XMPF_COARRAY_BUF=%s\n"
                            "                           XMPF_COARRAY_SAFE=%s\n",
                            ONESIDED_COMM_LAYER, ONESIDED_BOUNDARY,
                            env1 ? env1 : "",
                            env2 ? env2 : "",
                            env3 ? env3 : "",
                            env4 ? env4 : ""
                            );

    _XMPF_reset_coarrayMsg();
  }

  _XMPF_coarrayDebugPrint("Specified Parameters\n"
                          "   runtime message switch        :  %s\n"
                          "   pooling/allocation threshold  :  %u bytes\n"
                          "   static buffer (localBuf) size :  %u bytes\n"
                          "   safe buffer mode (PUT only)   :  %s\n"
                          "   GET-communication interface   :  type %d\n"
                          "   PUT-communication interface   :  type %d\n",
                          _XMPF_get_coarrayMsg() ? "on" : "off",
                          XMPF_get_poolThreshold(),
                          XMPF_get_localBufSize(),
                          XMPF_isSafeBufferMode() ? "on" : "off",
                          GET_INTERFACE_TYPE, PUT_INTERFACE_TYPE
                          );
}


static unsigned _envStringToBytes(char *str, char *envVarName)
{
  unsigned len;
  unsigned char c;
  int stat;

  stat = sscanf(str, "%u%c", &len, &c);

  switch (stat) {
  case EOF:
  case 0:
    // use default value of poolThread
    break;

  case 1:
    return len;

  case 2:
    switch (c) {
    case 'k':
    case 'K':
      return len * 1024;

    case 'm':
    case 'M':
      return len * 1024 * 1024;

    case 'g':
    case 'G':
      return len * 1024 * 1024 * 1024;

    default:
      _XMPF_coarrayFatal("Usage of environment variable %s: [0-9]+[kKmMgG]?", envVarName);
      break;
    }
    break;

  default:
    _XMPF_coarrayFatal("Illegal value of environ variable XMPF_COARRAY_POOL.\n"
                       "  Usage: [0-9]+[kKmMgG]?");
    break;
  }

  // error
  return 0;
}


/* NOT USED
 */
void _XMPF_coarray_finalize(void)
{
  xmpf_sync_all_auto_();
}


/*****************************************\
  restriction checker
\*****************************************/

int _XMPF_nowInTask()
{
  return xmp_num_nodes() < xmp_all_num_nodes();
}

void _XMPF_checkIfInTask(char *msgopt)
{
  if (_XMPF_nowInTask())
    _XMPF_coarrayFatal("current restriction: "
                       "cannot use %s in any task construct\n",
                       msgopt);
}

void xmpf_coarray_fatal_with_len_(char *msg, int *msglen)
{
  _XMPF_coarrayFatal("FATAL ERROR: %*s\n", *msglen, msg);
}

void _XMPF_coarrayFatal(char *format, ...)
{
  int rank;
  char work[300];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  fprintf(stderr, "CAF[%d] %s\n", rank, work);

  //xmpf_finalize_each__();   This causes deadlock sometimes.

  _XMP_fatal("...fatal error in XMP/F Coarray runtime");
}

void _XMPF_coarrayDebugPrint(char *format, ...)
{
  int current, initial;

  if (!_XMPF_coarrayMsg)
    return;

  char work[300];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);

  current = _XMPF_this_image_current();
  initial = _XMPF_this_image_initial();
  if (current == initial)
    fprintf(stderr, "CAF[%d] %s", initial, work);
  else
    fprintf(stderr, "CAF[%d(now %d)] %s", initial, current, work);
}

