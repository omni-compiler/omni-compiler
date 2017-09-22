#include "xmpf_internal_coarray.h"

#define DIV_CEILING(m,n)  (((m)-1)/(n)+1)


static unsigned _envStringToBytes(char *str, char *envVarName);


/*****************************************\
  initialization called in xmpf_main
\*****************************************/

/*  1. set static variable _this_image and _num_nodes
 *  2. read environment variable XMPF_COARRAY_MSG
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
   *  clean static data
   */
  _XMPF_set_this_image_initial();
  _XMPF_set_num_images_initial();
  _XMPF_coarray_clean_image_nodes();

  /*
   * read environment variables
   */
  char *tok, *work, *env1, *env2, *env3, *env4, *env5, *env6;
  int i;
  char delim[] = ", ";
  unsigned len;

  _XMPF_coarrayDebugPrint("Execution time environment\n"
                          "   communication layer  :  %s\n"
                          "    - memory allocation boundary : %u byte(s)\n"
                          "    - communication boundary     : %u byte(s)\n",
                          ONESIDED_COMM_LAYER,
                          MALLOC_UNIT,
                          COMM_UNIT
                          );


  env1 = getenv("XMPF_COARRAY_MSG");
  if (env1 != NULL) {
    work = strdup(env1);
    tok = strtok(work, delim);
    for (i = 1; tok != NULL; i++, tok = strtok(NULL, delim)) {
      if (_XMPF_this_image_current() == i) {
        _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_MSG=%s as %s\n", env1, tok);
        _XMPCO_set_isMsgMode(atoi(tok));
      }
    }
  }

  _XMPCO_set_isMsgMode_quietly(TRUE);

  env2 = getenv("XMPF_COARRAY_POOL");
  if (env2 != NULL) {
    len = _envStringToBytes(env2, "XMPF_COARRAY_POOL");
    if (len != 0) {
      _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_POOL=%s\n", env2);
      _XMPCO_set_poolThreshold(len);
    }
  }
    
  env3 = getenv("XMPF_COARRAY_BUF");
  if (env3 != NULL) {
    len = _envStringToBytes(env3, "XMPF_COARRAY_BUF");
    if (len != 0) {
      _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_BUF=%s\n", env3);
      _XMPCO_set_localBufSize(len);
    }
  }
    
  env4 = getenv("XMPF_COARRAY_SAFE");
  if (env4 != NULL) {
    _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_SAFE=%s\n", env4);
    _XMPCO_set_isSafeBufferMode(atoi(env4));
  }
    
  env5 = getenv("XMPF_COARRAY_SYNCPUT");
  if (env5 != NULL) {
    _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_SYNCPUT=%s\n", env5);
    _XMPCO_set_isSyncPutMode(atoi(env5));
  }

  env6 = getenv("XMPF_COARRAY_EAGER");
  if (env6 != NULL) {
    _XMPF_coarrayDebugPrint("Accepted XMPF_COARRAY_EAGER=%s\n", env6);
    _XMPCO_set_isEagerCommMode(atoi(env6));
  }

  _XMPCO_reset_isMsgMode();


  _XMPF_coarrayDebugPrint("Specified Parameters\n"
                          "   runtime message switch        :  %s\n"
                          "   pooling/allocation threshold  :  %u bytes\n"
                          "   static buffer (localBuf) size :  %u bytes\n"
                          "   safe buffer mode (PUT only)   :  %s\n"
                          "   sync put mode (PUT only)      :  %s\n"
                          "   eager communication mode      :  %s\n"
                          "   GET-communication interface   :  type %d\n"
                          "   PUT-communication interface   :  type %d\n",
                          _XMPCO_get_isMsgMode() ? "on" : "off",
                          _XMPCO_get_poolThreshold(),
                          _XMPCO_get_localBufSize(),
                          _XMPCO_get_isSafeBufferMode() ? "on" : "off",
                          _XMPCO_get_isSyncPutMode() ? "on" : "off",
                          _XMPCO_get_isEagerCommMode() ? "on" : "off",
                          GET_INTERFACE_TYPE,
                          PUT_INTERFACE_TYPE
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


void _XMPF_coarray_finalize(void)
{
  xmpf_sync_all_auto_();
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
  fprintf(stderr, "XMPF [rank=%d] %s\n", rank, work);

  _XMP_fatal("...fatal error in XMP/F Coarray runtime");
}

void __XMPF_coarrayDebugPrint(char *format, ...)
{
  int current, initial;

  char work[800];
  va_list list;
  va_start(list, format);
  vsprintf(work, format, list);
  va_end(list);

  current = _XMPF_this_image_current();
  initial = _XMPF_this_image_initial();
  if (current == initial)
    fprintf(stderr, "XMPF [%d] %s", initial, work);
  else
    fprintf(stderr, "XMPF [%d(current=%d)] %s", initial, current, work);
}


/*****************************************\
  hidden API,
   which can be used in the program
\*****************************************/

/*
 *  verbose message from CAF runtime
 *    sw == 1(on) or 0(off) for each node
 */
void xmpf_coarray_msg_(int *sw)
{
  _XMPCO_set_isMsgMode(*sw);
}


