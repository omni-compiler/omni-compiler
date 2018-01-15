
      subroutine sub(x)
        real x(100)
        open(100, file='mydata.dat', asynchronous='yes')
        write(100, asynchronous='yes') x
        wait(unit=100)
      end subroutine sub
      !
      program main
        integer,asynchronous :: ioitem
        integer :: v
        block
          asynchronous :: v
        end block
      end program

