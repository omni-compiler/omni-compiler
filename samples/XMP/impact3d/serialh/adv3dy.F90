!----------------------------------------------------------------------
! advance y with 3-d tvd scheme
!                           /
      subroutine adv3dy ( fstep )
!
!   <arguments>
!     fstep : advance step
!
!   <remarks>
!     none
!
!    coded by sakagami,h. ( isr ) 88/06/29
! modified by Sakagami,H. ( NIFS ) 13/10/18 : xmp benchmark version
!----------------------------------------------------------------------
#include "phys.macro"
      use parameter
      use constant
      use phys
      include "implicit.h"
      save
!....
      call measure ( 2, ' ', 3 )
!....
!$OMP PARALLEL 
!$OMP DO SCHEDULE(STATIC) &
!$OMP    PRIVATE(iy,ix,wu0,wu1,wv0,wv1,ww0,ww1,wh0,wh1,wuu,wvv,www, &
!$OMP            whh,wcc,wdr,wdm,wdn,wdl,wde,wc1,wc2)
      do iz = 1, lz
      do iy = 1, ly-1
      do ix = 1, lx
!..              * calc. u^(j,k+1/2,l), v^(j,k+1/2,l), c^(j,k+1/2,l) *
         wu0 = sm(ix,iy  ,iz) / sr(ix,iy  ,iz)
         wu1 = sm(ix,iy+1,iz) / sr(ix,iy+1,iz)
         wv0 = sn(ix,iy  ,iz) / sr(ix,iy  ,iz)
         wv1 = sn(ix,iy+1,iz) / sr(ix,iy+1,iz)
         ww0 = sl(ix,iy  ,iz) / sr(ix,iy  ,iz)
         ww1 = sl(ix,iy+1,iz) / sr(ix,iy+1,iz)
         wh0 = ( se(ix,iy  ,iz) + sp(ix,iy  ,iz) ) / sr(ix,iy  ,iz)
         wh1 = ( se(ix,iy+1,iz) + sp(ix,iy+1,iz) ) / sr(ix,iy+1,iz)
!.
         wuu = ( sqrt( sr(ix,iy,iz) ) * wu0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wu1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         wvv = ( sqrt( sr(ix,iy,iz) ) * wv0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wv1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         www = ( sqrt( sr(ix,iy,iz) ) * ww0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * ww1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         whh = ( sqrt( sr(ix,iy,iz) ) * wh0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wh1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         wcc = sqrt( ( sgam - 1.0d0 ) * &
                     ( whh - 0.5d0 * ( wuu**2 + wvv**2 + www**2 ) ) )
!..                                        * calc. alfa(i)(j,k+1/2,l) *
         wdr = sr(ix,iy+1,iz) - sr(ix,iy,iz)
         wdm = sm(ix,iy+1,iz) - sm(ix,iy,iz)
         wdn = sn(ix,iy+1,iz) - sn(ix,iy,iz)
         wdl = sl(ix,iy+1,iz) - sl(ix,iy,iz)
         wde = se(ix,iy+1,iz) - se(ix,iy,iz)
!..
         if( wcc .gt. szero ) then
            wc1 = ( sgam - 1.0d0 ) * &
              ( wde + 0.5d0 * wdr * ( wuu**2 + wvv**2 + www**2 ) &
                   - ( wdm * wuu + wdn * wvv + wdl * www ) ) / wcc**2
            wc2 = ( wdn - wdr * wvv ) / wcc
!..
            walfa1(ix,iy,iz) = wdr - wc1
            walfa2(ix,iy,iz) = wdl - wdr * www
            walfa3(ix,iy,iz) = wdm - wdr * wuu
            walfa4(ix,iy,iz) = 0.5d0 * ( wc1 + wc2 )
            walfa5(ix,iy,iz) = 0.5d0 * ( wc1 - wc2 )
         else
            walfa1(ix,iy,iz) = wdr
            walfa2(ix,iy,iz) = wdl - wdr * www
            walfa3(ix,iy,iz) = wdm - wdr * wuu
            walfa4(ix,iy,iz) = 0.0d0
            walfa5(ix,iy,iz) = 0.0d0
         end if
!..                                         * calc. nue(i)(j,k+1/2,l) *
         wnue1(ix,iy,iz) = wvv
         wnue2(ix,iy,iz) = wvv
         wnue3(ix,iy,iz) = wvv
         wnue4(ix,iy,iz) = wvv + wcc
         wnue5(ix,iy,iz) = wvv - wcc
      end do
      end do
      end do
!...
      wra2 = sram * fstep
!..
!$OMP DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
      do iy = 1, ly-1
      do ix = 1, lx
         wnue1(ix,iy,iz) = wnue1(ix,iy,iz) * wra2
         wnue2(ix,iy,iz) = wnue2(ix,iy,iz) * wra2
         wnue3(ix,iy,iz) = wnue3(ix,iy,iz) * wra2
         wnue4(ix,iy,iz) = wnue4(ix,iy,iz) * wra2
         wnue5(ix,iy,iz) = wnue5(ix,iy,iz) * wra2
      end do
      end do
      end do
!....                                             * calc. g(i)(j,k,l) *
!$OMP DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
      do iy = 2, ly-1
      do ix = 1, lx
!..
         if( walfa1(ix,iy-1,iz) * walfa1(ix,iy,iz) .gt. szero2 ) then
            if( walfa1(ix,iy,iz) .gt. 0.0d0 ) then
               wg1(ix,iy,iz) = &
                        min( walfa1(ix,iy-1,iz), walfa1(ix,iy,iz) )
            else
               wg1(ix,iy,iz) = &
                        max( walfa1(ix,iy-1,iz), walfa1(ix,iy,iz) )
            end if
            wtmp1(ix,iy,iz) = somga * &
           abs(      walfa1(ix,iy,iz)   -      walfa1(ix,iy-1,iz)   ) &
            / ( abs( walfa1(ix,iy,iz) ) + abs( walfa1(ix,iy-1,iz) ) )
         else
            wg1(ix,iy,iz) = 0.0d0
            wtmp1(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa2(ix,iy-1,iz) * walfa2(ix,iy,iz) .gt. szero2 ) then
            if( walfa2(ix,iy,iz) .gt. 0.0d0 ) then
               wg2(ix,iy,iz) = &
                        min( walfa2(ix,iy-1,iz), walfa2(ix,iy,iz) )
            else
               wg2(ix,iy,iz) = &
                        max( walfa2(ix,iy-1,iz), walfa2(ix,iy,iz) )
            end if
            wtmp2(ix,iy,iz) = somga * &
           abs(      walfa2(ix,iy,iz)   -      walfa2(ix,iy-1,iz)   ) &
            / ( abs( walfa2(ix,iy,iz) ) + abs( walfa2(ix,iy-1,iz) ) )
         else
            wg2(ix,iy,iz) = 0.0d0
            wtmp2(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa3(ix,iy-1,iz) * walfa3(ix,iy,iz) .gt. szero2 ) then
            if( walfa3(ix,iy,iz) .gt. 0.0d0 ) then
               wg3(ix,iy,iz) = &
                        min( walfa3(ix,iy-1,iz), walfa3(ix,iy,iz) )
            else
               wg3(ix,iy,iz) = &
                        max( walfa3(ix,iy-1,iz), walfa3(ix,iy,iz) )
            end if
            wtmp3(ix,iy,iz) = somga * &
           abs(      walfa3(ix,iy,iz)   -      walfa3(ix,iy-1,iz)   ) &
            / ( abs( walfa3(ix,iy,iz) ) + abs( walfa3(ix,iy-1,iz) ) )
         else
            wg3(ix,iy,iz) = 0.0d0
            wtmp3(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa4(ix,iy-1,iz) * walfa4(ix,iy,iz) .gt. szero2 ) then
            if( walfa4(ix,iy,iz) .gt. 0.0d0 ) then
               wg4(ix,iy,iz) = &
                        min( walfa4(ix,iy-1,iz), walfa4(ix,iy,iz) )
            else
               wg4(ix,iy,iz) = &
                        max( walfa4(ix,iy-1,iz), walfa4(ix,iy,iz) )
            end if
         else
            wg4(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa5(ix,iy-1,iz) * walfa5(ix,iy,iz) .gt. szero2 ) then
            if( walfa5(ix,iy,iz) .gt. 0.0d0 ) then
               wg5(ix,iy,iz) = &
                        min( walfa5(ix,iy-1,iz), walfa5(ix,iy,iz) )
            else
               wg5(ix,iy,iz) = &
                        max( walfa5(ix,iy-1,iz), walfa5(ix,iy,iz) )
            end if
         else
            wg5(ix,iy,iz) = 0.0d0
         end if
      end do
      end do
      end do
!....
!$OMP DO SCHEDULE(STATIC) &
!$OMP    PRIVATE(iy,ix,wq,wgg,wbeta1,wbeta2,wbeta3,wbeta4,wbeta5, &
!$OMP            wu0,wu1,wv0,wv1,ww0,ww1,wh0,wh1,wuu,wvv,www,whh,wcc, &
!$OMP            wrf0,wrf1,wmf0,wmf1,wnf0,wnf1,wlf0,wlf1,wef0,wef1)
      do iz = 1, lz
      do iy = 2, ly-2
      do ix = 1, lx
!....             * calc. set nue(i)(j,k+1/2,l) + gamma(i)(j,k+1/2,l) *
!....                                        * and beta(i)(j,k+1/2,l) *
         wq = abs( wnue1(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue1(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp1(ix,iy+1,iz), wtmp1(ix,iy,iz) ) )
         if( abs( walfa1(ix,iy,iz) ) .ge. szero ) then
            wnue1(ix,iy,iz) = wnue1(ix,iy,iz) + wgg * &
               ( wg1(ix,iy+1,iz) - wg1(ix,iy,iz) ) / walfa1(ix,iy,iz)
         end if
         wq = abs( wnue1(ix,iy,iz) )
         wbeta1 = wgg * ( wg1(ix,iy+1,iz) + wg1(ix,iy,iz) ) &
                                              - wq * walfa1(ix,iy,iz)
!.
         wq = abs( wnue2(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue2(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp2(ix,iy+1,iz), wtmp2(ix,iy,iz) ) )
         if( abs( walfa2(ix,iy,iz) ) .ge. szero ) then
            wnue2(ix,iy,iz) = wnue2(ix,iy,iz) + wgg * &
               ( wg2(ix,iy+1,iz) - wg2(ix,iy,iz) ) / walfa2(ix,iy,iz)
         end if
         wq = abs( wnue2(ix,iy,iz) )
         wbeta2 = wgg * ( wg2(ix,iy+1,iz) + wg2(ix,iy,iz) ) &
                                             - wq * walfa2(ix,iy,iz)
!.
         wq = abs( wnue3(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue3(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp3(ix,iy+1,iz), wtmp3(ix,iy,iz) ) )
         if( abs( walfa3(ix,iy,iz) ) .ge. szero ) then
            wnue3(ix,iy,iz) = wnue3(ix,iy,iz) + wgg * &
               ( wg3(ix,iy+1,iz) - wg3(ix,iy,iz) ) / walfa3(ix,iy,iz)
         end if
         wq = abs( wnue3(ix,iy,iz) )
         wbeta3 = wgg * ( wg3(ix,iy+1,iz) + wg3(ix,iy,iz) ) &
                                              - wq * walfa3(ix,iy,iz)
!.
         wq = abs( wnue4(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0*seps )
         wgg = 0.5d0 * ( wq - wnue4(ix,iy,iz)**2 )
         if( abs( walfa4(ix,iy,iz) ) .ge. szero ) then
            wnue4(ix,iy,iz) = wnue4(ix,iy,iz) + wgg * &
               ( wg4(ix,iy+1,iz) - wg4(ix,iy,iz) ) / walfa4(ix,iy,iz)
         end if
         wq = abs( wnue4(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0 * seps )
         wbeta4 = wgg * ( wg4(ix,iy+1,iz) + wg4(ix,iy,iz) ) &
                                              - wq * walfa4(ix,iy,iz)
!.
         wq = abs( wnue5(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0*seps )
         wgg = 0.5d0 * ( wq - wnue5(ix,iy,iz)**2 )
         if( abs( walfa5(ix,iy,iz) ) .ge. szero ) then
            wnue5(ix,iy,iz) = wnue5(ix,iy,iz) + wgg * &
               ( wg5(ix,iy+1,iz) - wg5(ix,iy,iz) ) / walfa5(ix,iy,iz)
         end if
         wq = abs( wnue5(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0 * seps )
         wbeta5 = wgg * ( wg5(ix,iy+1,iz) + wg5(ix,iy,iz) ) &
                                              - wq * walfa5(ix,iy,iz)
!..                                * calc. modified flux f(j,k+1/2,l) *
         wu0 = sm(ix,iy  ,iz) / sr(ix,iy  ,iz)
         wu1 = sm(ix,iy+1,iz) / sr(ix,iy+1,iz)
         wv0 = sn(ix,iy  ,iz) / sr(ix,iy  ,iz)
         wv1 = sn(ix,iy+1,iz) / sr(ix,iy+1,iz)
         ww0 = sl(ix,iy  ,iz) / sr(ix,iy  ,iz)
         ww1 = sl(ix,iy+1,iz) / sr(ix,iy+1,iz)
         wh0 = ( se(ix,iy  ,iz) + sp(ix,iy  ,iz) ) / sr(ix,iy  ,iz)
         wh1 = ( se(ix,iy+1,iz) + sp(ix,iy+1,iz) ) / sr(ix,iy+1,iz)
!.
         wrf0 = sn(ix,iy  ,iz)
         wrf1 = sn(ix,iy+1,iz)
         wmf0 = sn(ix,iy  ,iz) * wu0
         wmf1 = sn(ix,iy+1,iz) * wu1
         wnf0 = sn(ix,iy  ,iz) * wv0 + sp(ix,iy  ,iz)
         wnf1 = sn(ix,iy+1,iz) * wv1 + sp(ix,iy+1,iz)
         wlf0 = sn(ix,iy  ,iz) * ww0
         wlf1 = sn(ix,iy+1,iz) * ww1
         wef0 = ( se(ix,iy  ,iz) + sp(ix,iy  ,iz) ) * wv0
         wef1 = ( se(ix,iy+1,iz) + sp(ix,iy+1,iz) ) * wv1
!.
         wuu = ( sqrt( sr(ix,iy,iz) ) * wu0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wu1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         wvv = ( sqrt( sr(ix,iy,iz) ) * wv0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wv1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         www = ( sqrt( sr(ix,iy,iz) ) * ww0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * ww1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         whh = ( sqrt( sr(ix,iy,iz) ) * wh0 + &
                                       sqrt( sr(ix,iy+1,iz) ) * wh1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy+1,iz) ) )
         wcc = sqrt( ( sgam - 1.0d0 ) * &
                     ( whh - 0.5d0 * ( wuu**2 + wvv**2 + www**2 ) ) )
!..
         wff1(ix,iy,iz) = 0.5d0 * ( ( wrf0 + wrf1 ) + &
            ( wbeta1 &
            + wbeta4 &
            + wbeta5 &
            ) / wra2 )
         wff2(ix,iy,iz) = 0.5d0 * ( ( wmf0 + wmf1 ) + &
            ( wbeta1 * wuu &
            + wbeta3 &
            + wbeta4 * wuu &
            + wbeta5 * wuu &
            ) / wra2 )
         wff3(ix,iy,iz) = 0.5d0 * ( ( wnf0 + wnf1 ) + &
            ( wbeta1 * wvv &
            + wbeta4 * ( wvv + wcc ) &
            + wbeta5 * ( wvv - wcc ) &
            ) / wra2 )
         wff4(ix,iy,iz) = 0.5d0 * ( ( wlf0 + wlf1 ) + &
            ( wbeta1 * www &
            + wbeta2 &
            + wbeta4 * www &
            + wbeta5 * www &
            ) / wra2 )
         wff5(ix,iy,iz) = 0.5d0 * ( ( wef0 + wef1 ) + &
            ( wbeta1 * 0.5d0 * ( wuu**2 + wvv**2 + www**2 ) &
            + wbeta2 * www &
            + wbeta3 * wuu &
            + wbeta4 * ( whh + wcc * wvv ) &
            + wbeta5 * ( whh - wcc * wvv ) &
            ) / wra2 )
      end do
      end do
      end do
!....                                                     * advance y *
!$OMP DO SCHEDULE(STATIC) PRIVATE(iy,ix)
      do iz = 1, lz
      do iy = 3, ly-2
      do ix = 1, lx
         sr(ix,iy,iz) = sr(ix,iy,iz) + &
                          wra2 * ( wff1(ix,iy-1,iz) - wff1(ix,iy,iz) )
         sm(ix,iy,iz) = sm(ix,iy,iz) + &
                          wra2 * ( wff2(ix,iy-1,iz) - wff2(ix,iy,iz) )
         sn(ix,iy,iz) = sn(ix,iy,iz) + &
                          wra2 * ( wff3(ix,iy-1,iz) - wff3(ix,iy,iz) )
         sl(ix,iy,iz) = sl(ix,iy,iz) + &
                          wra2 * ( wff4(ix,iy-1,iz) - wff4(ix,iy,iz) )
         se(ix,iy,iz) = se(ix,iy,iz) + &
                          wra2 * ( wff5(ix,iy-1,iz) - wff5(ix,iy,iz) )
      end do
      end do
      end do
!$OMP END PARALLEL
!....
      call measure ( 3, ' ', 3 )
!....
      return
      end
