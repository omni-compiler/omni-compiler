      PROGRAM main
        REAL,TARGET  :: x(-1:1,-2:2)
        REAL,POINTER :: p(:,:)
        p(1:,1:) => x
        do i1=-1, 1
        do i2=-2, 2
          i3 = i3 + 1
          x(i1,i2) = i3
        end do
        end do
        if(p(3,2).eq.12.0) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        end if
      END PROGRAM main
