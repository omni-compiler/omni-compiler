#include "xmpco_internal.h"

static unsigned int _count_syncall = 0;
static unsigned int _count_syncall_auto = 0;


/*****************************************\
  sync all
\*****************************************/

void XMPCO_sync_all()
{
  int stat = 0;

  ////////////////////////////////////
  _XMPCO_debugPrint("GACHACHA\n");
  ////////////////////////////////////
  if (_XMPCO_is_subset_exec()) {
  ////////////////////////////////////
  _XMPCO_debugPrint("GACHA consume\n");
  ////////////////////////////////////
    XMPCO_sync_all_withComm(_XMPCO_consume_comm_current());
    _XMPCO_debugPrint("SYNCALL on subset(%d images) done\n",
                      _XMPCO_get_currentNumImages());
    return;
  }

  _count_syncall += 1;
  xmp_sync_all(&stat);
  if (stat != 0)
    _XMPCO_fatal("SYNCALL failed with stat=%d", stat);

  _XMPCO_debugPrint("SYNCALL done (_count_syncall=%d, stat=%d)\n",
                    _count_syncall, stat);
}


/* entry for automatic syncall at the end of procedures
 */
void XMPCO_sync_all_auto()
{
  int stat = 0;

  if (_XMPCO_is_subset_exec()) {
    XMPCO_sync_all_withComm(_XMPCO_consume_comm_current());
    _XMPCO_debugPrint("SYNCALL AUTO on subset(%d images) done\n",
                      _XMPCO_get_currentNumImages());
    return;
  } 

  _count_syncall_auto += 1;
  xmp_sync_all(&stat);
  if (stat != 0)
    _XMPCO_fatal("SYNCALL_AUTO failed with stat=%d", stat);

  _XMPCO_debugPrint("SYNCALL_AUTO done (_count_syncall_auto=%d, stat=%d)\n",
                    _count_syncall_auto, stat);
}

void XMPCO_sync_all_withComm(MPI_Comm comm)
{
  int stat = 0;
  
  xmp_sync_memory(&stat);
  if (stat != 0)
    _XMPCO_fatal("SYNC MEMORY inside SYNC ALL failed with stat=%d", stat);
  stat = MPI_Barrier(comm);
  if (stat != 0)
    _XMPCO_fatal("MPI_Barrier inside SYNC ALL failed with stat=%d", stat);
  return;
}



