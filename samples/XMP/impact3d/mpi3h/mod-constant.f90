module constant
      use parameter
      real*8 :: sgam, smue, sram, seps, szero, szero2, somga
      real*8 :: ssenx(8*(lly+1)*(llz+1)*2), &
                srecx(8*(lly+1)*(llz+1)*2), &
                sseny(8*(llx+1)*(llz+1)*2), &
                srecy(8*(llx+1)*(llz+1)*2)
      integer :: lrank, llbx, lubx, llby, luby, llbz, lubz, & 
                 lrkx, lrkxp, lrkxm, lrky, lrkyp, lrkym, &
                 lrkz, lrkzp, lrkzm
      include "mpif.h"
end module constant
