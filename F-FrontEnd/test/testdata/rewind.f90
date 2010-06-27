      program main
        REWIND 10
        REWIND (21, ERR=99, IOSTAT=ios)
99      WRITE(*,*) "error"
      end program main
