      program main
        integer, dimension(10, 5)   :: b
        integer i, j

        DATA i / 1 /
        DATA ((b(i, j), i=1,10), j=1,5) / 50*0 /

        print *, ((b(i, j), i=1,10), j=1,5)
      end
