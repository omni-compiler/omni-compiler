  logical lock[*], value
  double precision z=0.9999

  me = this_image()
  lock = .true.

  if (me==1) then
     value=.true.
     write(*,*) me, "wait start"
     do while (value)
        call atomic_ref(value, lock)
        call zzz(z)
     enddo
     write(*,*) me, "wait completed"
  end if

  if (me==2) then
     write(*,*) me, "unlock start"
     call atomic_define(lock[1], .false.)
     write(*,*) me, "unlock completed"
  endif

  end

  subroutine zzz(a)
    double precision a
    do k=1,1000
       do j=1,1000
          a=a*a
       enddo
    enddo
  end subroutine zzz

