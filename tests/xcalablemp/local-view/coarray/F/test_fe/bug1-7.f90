      interface zzz
         subroutine zzz1(arg)
           integer :: arg
         end subroutine
         subroutine zzz2(arg)
           character(len=1) :: arg
         end subroutine
      end interface
      integer n
      call zzz(n)
    end
