      program main
        integer         :: i
        integer(KIND=2) :: i2
        integer(KIND=4) :: i4
        integer(KIND=8) :: i8
        character       :: s

        real(KIND=4)    :: r4
        double precision :: r8

        i = min(2,2,3,4,5)

        s = min("a", "b", "c")

        i4 = min0(i4,i4,i4)

        r4 = amin0(i4,i4)

        r4 = min1(r4,r4,r4)

        r4 = amin1(r4,r4)

        r8 = dmin1(r8, r8, r8)


      end program main
