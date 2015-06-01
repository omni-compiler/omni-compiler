  subroutine poo
    interface tenten
       subroutine zappa(n)
         integer n
       end subroutine zappa
       subroutine dedeno(r)
         real r
       end subroutine dedeno
    end interface tenten

    call peppo

    contains

      subroutine peppo
        call tenten(350)
      end subroutine peppo

    end subroutine poo

    subroutine zappa(n)
      write(*,*) "happy value=",n
    end subroutine zappa

    program main
      call poo
    end program main

