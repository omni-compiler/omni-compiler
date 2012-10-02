!=======================================================================
! PRAGMATEST
!
!=======================================================================
!
!
!

program PRAGMATEST
    INTEGER*4 :: a, b
!
!$OMP PARALLEL DEFAULT(NONE) &
!$OMP          PRIVATE(aa,bb,cc,i,j,k,l,rr,qq,xx,yy,zz,ggg, &
!$OMP                  ixx,iyy,izz,ddd,ggg,hhh,jj,  &
!$OMP                  ss,tt,uu,www,iww,jww  )

! COMMENT01
! COMMENT02
! COMMENT03
    i = i+1

!$OMP end PARALLEL

end program PRAGMATEST
