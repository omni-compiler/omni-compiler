      program main
        integer, dimension(10)   :: b
        integer, dimension(10)   :: c
        integer i

        DATA i / 1 /
        DATA (b(i), i=1,10) / 10*0 /

        print *, (b(i), i=1,10)
      end
