      program test_aint
        real r
        double precision d
        r = 1.234E0_4
        d = 4.321_8
        print *, aint(r), dint(d)
        d = aint(r,8)
      end program test_aint
