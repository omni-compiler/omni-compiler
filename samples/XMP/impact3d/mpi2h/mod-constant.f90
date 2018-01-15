module constant
      use parameter
      real*8 :: sgam, smue, sram, seps, szero, szero2, somga
      real*8 :: ssend(8*lx*(llz+1)*2), srecv(8*lx*(llz+1)*2)
      integer :: lrank, llby, luby, llbz, lubz, &
                 lrky, lrkyp, lrkym, lrkz, lrkzp, lrkzm
      include "mpif.h"
end module constant
