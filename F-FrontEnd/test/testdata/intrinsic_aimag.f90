      program test_aimag
        complex sc
        complex(8) dc ! or double complex?
        sc = cmplx(1.e0_4, 0.e0_4)
        dc = cmplx(0.e0_8, 1.e0_8)
        print *, aimag(sc), dimag(dc)
      end program test_aimag
