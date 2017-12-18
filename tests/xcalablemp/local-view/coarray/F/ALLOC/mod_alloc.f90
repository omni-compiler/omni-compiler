module modalloc
  real(8), allocatable :: ama(:,:)[:]

contains
  subroutine tete

  end subroutine tete
end module modalloc

program zan
  use modalloc
  real(8), allocatable :: lo(:,:)[:]
  real(8) :: expect

  allocate(lo(10,10)[*])
  allocate(ama(10,10)[*])

  me=this_image()
  ni=num_images()

  do j=1,10
     do i=1,10
        lo(i,j)=i*100+j*10+me
     enddo
  enddo

  sync all

  do j=1,10
     do i=1,10
        ama(j,i)[mod(me,ni)+1] = lo(i,j)
     enddo
  enddo

  sync all

!!  write(*,*) me,lo(1,2),ama(2,1)

  if (me==1) then
     me_dif=ni-1
  else
     me_dif=-1
  endif
  nerr=0

  do j=1,10
     do i=1,10
        expect=lo(j,i)+me_dif
        if (abs(ama(i,j)-expect) > 0.00000001) then
           nerr=nerr+1
           write(*,*) "NG: expect, ama(i,j)=",expect, ama(i,j)
        endif
     enddo
  enddo


  if (nerr==0) then
     write(*,100) me, "OK"
  else
     write(*,101) me, "NG", nerr
  endif

100 format("[",i0,"] ",a) 
101 format("[",i0,"] ",a," nerr=",i0) 

end program zan

subroutine tete
  use modalloc
end subroutine tete

