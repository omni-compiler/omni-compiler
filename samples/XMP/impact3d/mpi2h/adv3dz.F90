!----------------------------------------------------------------------
! advance z with 3-d tvd scheme
!                           /
      subroutine adv3dz ( fstep )
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
      call measure ( 2, ' ', 4 )
!...
      wra2 = sram * fstep
!....
      call measure ( 2, ' ', 8 )
      call shiftz( physval1, 6, lx, lly+1, llz+1, -1 )
      call measure ( 3, ' ', 8 )
!....
!$OMP PARALLEL FIRSTPRIVATE(wra2) PRIVATE(izs,ize, &
!$OMP             wu0,wu1,wv0,wv1,ww0,ww1,wh0,wh1,wuu,wvv,www,whh,wcc, &
!$OMP             wdr,wdm,wdn,wdl,wde,wc1,wc2, &
!$OMP             wq,wgg,wbeta1,wbeta2,wbeta3,wbeta4,wbeta5, &
!$OMP             wrf0,wrf1,wmf0,wmf1,wnf0,wnf1,wlf0,wlf1,wef0,wef1)
      ize = min( lz-1, lubz )
      ize = ize - llbz + 1
!...
!$OMP DO SCHEDULE(STATIC)
      do iz = 1, ize
      do iy = 1, lly
      do ix = 1, lx
!..              * calc. u^(j,k,l+1/2), v^(j,k,l+1/2), c^(j,k,l+1/2) *
         wu0 = sm(ix,iy,iz  ) / sr(ix,iy,iz  )
         wu1 = sm(ix,iy,iz+1) / sr(ix,iy,iz+1)
         wv0 = sn(ix,iy,iz  ) / sr(ix,iy,iz  )
         wv1 = sn(ix,iy,iz+1) / sr(ix,iy,iz+1)
         ww0 = sl(ix,iy,iz  ) / sr(ix,iy,iz  )
         ww1 = sl(ix,iy,iz+1) / sr(ix,iy,iz+1)
         wh0 = ( se(ix,iy,iz  ) + sp(ix,iy,iz  ) ) / sr(ix,iy,iz  )
         wh1 = ( se(ix,iy,iz+1) + sp(ix,iy,iz+1) ) / sr(ix,iy,iz+1)
!.
         wuu = ( sqrt( sr(ix,iy,iz) ) * wu0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wu1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         wvv = ( sqrt( sr(ix,iy,iz) ) * wv0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wv1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         www = ( sqrt( sr(ix,iy,iz) ) * ww0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * ww1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         whh = ( sqrt( sr(ix,iy,iz) ) * wh0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wh1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         wcc = sqrt( ( sgam - 1.0d0 ) * &
                     ( whh - 0.5d0 * ( wuu**2 + wvv**2 + www**2 ) ) )
!..                                        * calc. alfa(i)(j,k,l+1/2) *
         wdr = sr(ix,iy,iz+1) - sr(ix,iy,iz)
         wdm = sm(ix,iy,iz+1) - sm(ix,iy,iz)
         wdn = sn(ix,iy,iz+1) - sn(ix,iy,iz)
         wdl = sl(ix,iy,iz+1) - sl(ix,iy,iz)
         wde = se(ix,iy,iz+1) - se(ix,iy,iz)
!..
         if( wcc .gt. szero ) then
            wc1 = ( sgam - 1.0d0 ) * &
              ( wde + 0.5d0 * wdr * ( wuu**2 + wvv**2 + www**2 ) &
                   - ( wdm * wuu + wdn * wvv + wdl * www ) ) / wcc**2
            wc2 = ( wdl - wdr * www ) / wcc
!..
            walfa1(ix,iy,iz) = wdr - wc1
            walfa2(ix,iy,iz) = wdm - wdr * wuu
            walfa3(ix,iy,iz) = wdn - wdr * wvv
            walfa4(ix,iy,iz) = 0.5d0 * ( wc1 + wc2 )
            walfa5(ix,iy,iz) = 0.5d0 * ( wc1 - wc2 )
         else
            walfa1(ix,iy,iz) = wdr
            walfa2(ix,iy,iz) = wdm - wdr * wuu
            walfa3(ix,iy,iz) = wdn - wdr * wvv
            walfa4(ix,iy,iz) = 0.0d0
            walfa5(ix,iy,iz) = 0.0d0
         end if
!..                                         * calc. nue(i)(j,k,l+1/2) *
         wnue1(ix,iy,iz) = www
         wnue2(ix,iy,iz) = www
         wnue3(ix,iy,iz) = www
         wnue4(ix,iy,iz) = www + wcc
         wnue5(ix,iy,iz) = www - wcc
      end do
      end do
      end do
!..
!$OMP DO SCHEDULE(STATIC)
      do iz = 1, ize
      do iy = 1, lly
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
      call measure ( 2, ' ', 8 )
!$OMP MASTER
      call shiftz( physval3, 5, lx, lly+1, llz+1, 1 )
!$OMP END MASTER
!$OMP BARRIER
      call measure ( 3, ' ', 8 )
!....
      izs = max( 2, llbz )
      izs = izs - llbz + 1
!...
!$OMP DO SCHEDULE(STATIC)
      do iz = izs, ize
      do iy = 1, lly
      do ix = 1, lx
!..
         if( walfa1(ix,iy,iz-1) * walfa1(ix,iy,iz) .gt. szero2 ) then
            if( walfa1(ix,iy,iz) .gt. 0.0d0 ) then
               wg1(ix,iy,iz) = &
                        min( walfa1(ix,iy,iz-1), walfa1(ix,iy,iz) )
            else
               wg1(ix,iy,iz) = &
                        max( walfa1(ix,iy,iz-1), walfa1(ix,iy,iz) )
            end if
            wtmp1(ix,iy,iz) = somga * &
           abs(      walfa1(ix,iy,iz)   -      walfa1(ix,iy,iz-1)   ) &
            / ( abs( walfa1(ix,iy,iz) ) + abs( walfa1(ix,iy,iz-1) ) )
         else
            wg1(ix,iy,iz) = 0.0d0
            wtmp1(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa2(ix,iy,iz-1) * walfa2(ix,iy,iz) .gt. szero2 ) then
            if( walfa2(ix,iy,iz) .gt. 0.0d0 ) then
               wg2(ix,iy,iz) = &
                        min( walfa2(ix,iy,iz-1), walfa2(ix,iy,iz) )
            else
               wg2(ix,iy,iz) = &
                        max( walfa2(ix,iy,iz-1), walfa2(ix,iy,iz) )
            end if
            wtmp2(ix,iy,iz) = somga * &
           abs(      walfa2(ix,iy,iz)   -      walfa2(ix,iy,iz-1)   ) &
            / ( abs( walfa2(ix,iy,iz) ) + abs( walfa2(ix,iy,iz-1) ) )
         else
            wg2(ix,iy,iz) = 0.0d0
            wtmp2(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa3(ix,iy,iz-1) * walfa3(ix,iy,iz) .gt. szero2 ) then
            if( walfa3(ix,iy,iz) .gt. 0.0d0 ) then
               wg3(ix,iy,iz) = &
                        min( walfa3(ix,iy,iz-1), walfa3(ix,iy,iz) )
            else
               wg3(ix,iy,iz) = &
                        max( walfa3(ix,iy,iz-1), walfa3(ix,iy,iz) )
            end if
            wtmp3(ix,iy,iz) = somga * &
           abs(      walfa3(ix,iy,iz)   -      walfa3(ix,iy,iz-1)   ) &
            / ( abs( walfa3(ix,iy,iz) ) + abs( walfa3(ix,iy,iz-1) ) )
         else
            wg3(ix,iy,iz) = 0.0d0
            wtmp3(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa4(ix,iy,iz-1) * walfa4(ix,iy,iz) .gt. szero2 ) then
            if( walfa4(ix,iy,iz) .gt. 0.0d0 ) then
               wg4(ix,iy,iz) = &
                        min( walfa4(ix,iy,iz-1), walfa4(ix,iy,iz) )
            else
               wg4(ix,iy,iz) = &
                        max( walfa4(ix,iy,iz-1), walfa4(ix,iy,iz) )
            end if
         else
            wg4(ix,iy,iz) = 0.0d0
         end if
!..
         if( walfa5(ix,iy,iz-1) * walfa5(ix,iy,iz) .gt. szero2 ) then
            if( walfa5(ix,iy,iz) .gt. 0.0d0 ) then
               wg5(ix,iy,iz) = &
                        min( walfa5(ix,iy,iz-1), walfa5(ix,iy,iz) )
            else
               wg5(ix,iy,iz) = &
                        max( walfa5(ix,iy,iz-1), walfa5(ix,iy,iz) )
            end if
         else
            wg5(ix,iy,iz) = 0.0d0
         end if
      end do
      end do
      end do
!....
      call measure ( 2, ' ', 8 )
!$OMP MASTER
      call shiftz( physval4, 8, lx, lly+1, llz+1, -1 )
!$OMP END MASTER
!$OMP BARRIER
      call measure ( 3, ' ', 8 )
!....
      ize = min( lz-2, lubz )
      ize = ize - llbz + 1
!...
!$OMP DO SCHEDULE(STATIC)
      do iz = izs, ize
      do iy = 1, lly
      do ix = 1, lx
!....             * calc. set nue(i)(j,k,l+1/2) + gamma(i)(j,k,l+1/2) *
!....                                        * and beta(i)(j,k,l+1/2) *
         wq = abs( wnue1(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue1(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp1(ix,iy,iz+1), wtmp1(ix,iy,iz) ) )
         if( abs( walfa1(ix,iy,iz) ) .ge. szero ) then
            wnue1(ix,iy,iz) = wnue1(ix,iy,iz) + wgg * &
               ( wg1(ix,iy,iz+1) - wg1(ix,iy,iz) ) / walfa1(ix,iy,iz)
         end if
         wq = abs( wnue1(ix,iy,iz) )
         wbeta1 = wgg * ( wg1(ix,iy,iz+1) + wg1(ix,iy,iz) ) &
                                              - wq * walfa1(ix,iy,iz)
!.
         wq = abs( wnue2(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue2(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp2(ix,iy,iz+1), wtmp2(ix,iy,iz) ) )
         if( abs( walfa2(ix,iy,iz) ) .ge. szero ) then
            wnue2(ix,iy,iz) = wnue2(ix,iy,iz) + wgg * &
               ( wg2(ix,iy,iz+1) - wg2(ix,iy,iz) ) / walfa2(ix,iy,iz)
         end if
         wq = abs( wnue2(ix,iy,iz) )
         wbeta2 = wgg * ( wg2(ix,iy,iz+1) + wg2(ix,iy,iz) ) &
                                             - wq * walfa2(ix,iy,iz)
!.
         wq = abs( wnue3(ix,iy,iz) )
         wgg = 0.5d0 * ( wq - wnue3(ix,iy,iz)**2 ) &
                * ( 1.0d0 + max( wtmp3(ix,iy,iz+1), wtmp3(ix,iy,iz) ) )
         if( abs( walfa3(ix,iy,iz) ) .ge. szero ) then
            wnue3(ix,iy,iz) = wnue3(ix,iy,iz) + wgg * &
               ( wg3(ix,iy,iz+1) - wg3(ix,iy,iz) ) / walfa3(ix,iy,iz)
         end if
         wq = abs( wnue3(ix,iy,iz) )
         wbeta3 = wgg * ( wg3(ix,iy,iz+1) + wg3(ix,iy,iz) ) &
                                              - wq * walfa3(ix,iy,iz)
!.
         wq = abs( wnue4(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0*seps )
         wgg = 0.5d0 * ( wq - wnue4(ix,iy,iz)**2 )
         if( abs( walfa4(ix,iy,iz) ) .ge. szero ) then
            wnue4(ix,iy,iz) = wnue4(ix,iy,iz) + wgg * &
               ( wg4(ix,iy,iz+1) - wg4(ix,iy,iz) ) / walfa4(ix,iy,iz)
         end if
         wq = abs( wnue4(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0 * seps )
         wbeta4 = wgg * ( wg4(ix,iy,iz+1) + wg4(ix,iy,iz) ) &
                                              - wq * walfa4(ix,iy,iz)
!.
         wq = abs( wnue5(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0*seps )
         wgg = 0.5d0 * ( wq - wnue5(ix,iy,iz)**2 )
         if( abs( walfa5(ix,iy,iz) ) .ge. szero ) then
            wnue5(ix,iy,iz) = wnue5(ix,iy,iz) + wgg * &
               ( wg5(ix,iy,iz+1) - wg5(ix,iy,iz) ) / walfa5(ix,iy,iz)
         end if
         wq = abs( wnue5(ix,iy,iz) )
         if( wq .lt. seps ) wq = ( wq*wq + seps*seps ) / ( 2.0 * seps )
         wbeta5 = wgg * ( wg5(ix,iy,iz+1) + wg5(ix,iy,iz) ) &
                                              - wq * walfa5(ix,iy,iz)
!..                                * calc. modified flux f(j,k,l+1/2) *
         wu0 = sm(ix,iy,iz  ) / sr(ix,iy,iz  )
         wu1 = sm(ix,iy,iz+1) / sr(ix,iy,iz+1)
         wv0 = sn(ix,iy,iz  ) / sr(ix,iy,iz  )
         wv1 = sn(ix,iy,iz+1) / sr(ix,iy,iz+1)
         ww0 = sl(ix,iy,iz  ) / sr(ix,iy,iz  )
         ww1 = sl(ix,iy,iz+1) / sr(ix,iy,iz+1)
         wh0 = ( se(ix,iy,iz  ) + sp(ix,iy,iz  ) ) / sr(ix,iy,iz  )
         wh1 = ( se(ix,iy,iz+1) + sp(ix,iy,iz+1) ) / sr(ix,iy,iz+1)
!.
         wrf0 = sl(ix,iy,iz  )
         wrf1 = sl(ix,iy,iz+1)
         wmf0 = sl(ix,iy,iz  ) * wu0
         wmf1 = sl(ix,iy,iz+1) * wu1
         wnf0 = sl(ix,iy,iz  ) * wv0
         wnf1 = sl(ix,iy,iz+1) * wv1
         wlf0 = sl(ix,iy,iz  ) * ww0 + sp(ix,iy,iz  )
         wlf1 = sl(ix,iy,iz+1) * ww1 + sp(ix,iy,iz+1)
         wef0 = ( se(ix,iy,iz  ) + sp(ix,iy,iz  ) ) * ww0
         wef1 = ( se(ix,iy,iz+1) + sp(ix,iy,iz+1) ) * ww1
!.
         wuu = ( sqrt( sr(ix,iy,iz) ) * wu0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wu1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         wvv = ( sqrt( sr(ix,iy,iz) ) * wv0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wv1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         www = ( sqrt( sr(ix,iy,iz) ) * ww0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * ww1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
         whh = ( sqrt( sr(ix,iy,iz) ) * wh0 + &
                                       sqrt( sr(ix,iy,iz+1) ) * wh1 ) &
            / ( sqrt( sr(ix,iy,iz) ) + sqrt( sr(ix,iy,iz+1) ) )
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
            + wbeta2 &
            + wbeta4 * wuu &
            + wbeta5 * wuu &
            ) / wra2 )
         wff3(ix,iy,iz) = 0.5d0 * ( ( wnf0 + wnf1 ) + &
            ( wbeta1 * wvv &
            + wbeta3 &
            + wbeta4 * wvv &
            + wbeta5 * wvv &
            ) / wra2 )
         wff4(ix,iy,iz) = 0.5d0 * ( ( wlf0 + wlf1 ) + &
            ( wbeta1 * www &
            + wbeta4 * ( www + wcc ) &
            + wbeta5 * ( www - wcc ) &
            ) / wra2 )
         wff5(ix,iy,iz) = 0.5d0 * ( ( wef0 + wef1 ) + &
            ( wbeta1 * 0.5d0 * ( wuu**2 + wvv**2 + www**2 ) &
            + wbeta2 * wuu &
            + wbeta3 * wvv &
            + wbeta4 * ( whh + wcc * www ) &
            + wbeta5 * ( whh - wcc * www ) &
            ) / wra2 )
      end do
      end do
      end do
!....
      call measure ( 2, ' ', 8 )
!$OMP MASTER
      call shiftz( physval5, 5, lx, lly+1, llz+1, 1 )
!$OMP END MASTER
!$OMP BARRIER
      call measure ( 3, ' ', 8 )
!....                                                     * advance z *
      izs = max( 3, llbz )
      izs = izs - llbz + 1
!...
!$OMP DO SCHEDULE(STATIC)
      do iz = izs, ize
      do iy = 1, lly
      do ix = 1, lx
         sr(ix,iy,iz) = sr(ix,iy,iz) + &
                          wra2 * ( wff1(ix,iy,iz-1) - wff1(ix,iy,iz) )
         sm(ix,iy,iz) = sm(ix,iy,iz) + &
                          wra2 * ( wff2(ix,iy,iz-1) - wff2(ix,iy,iz) )
         sn(ix,iy,iz) = sn(ix,iy,iz) + &
                          wra2 * ( wff3(ix,iy,iz-1) - wff3(ix,iy,iz) )
         sl(ix,iy,iz) = sl(ix,iy,iz) + &
                          wra2 * ( wff4(ix,iy,iz-1) - wff4(ix,iy,iz) )
         se(ix,iy,iz) = se(ix,iy,iz) + &
                          wra2 * ( wff5(ix,iy,iz-1) - wff5(ix,iy,iz) )
      end do
      end do
      end do
!$OMP END PARALLEL
!....
      call measure ( 3, ' ', 4 )
!....
      return
      end
