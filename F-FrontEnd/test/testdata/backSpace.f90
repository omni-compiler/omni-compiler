      program main
        BACKSPACE 10
        BACKSPACE (21, ERR=99, IOSTAT=ios)
99      WRITE(*,*) "error"
      end program main
