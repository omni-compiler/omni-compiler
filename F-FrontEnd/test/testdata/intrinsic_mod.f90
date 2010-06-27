      program main
        integer         :: i
        integer(KIND=4) :: i4
        integer(KIND=8) :: i8

        real                :: r
        double precision    :: d

        i4 = mod(i4,i4)
        i8 = mod(i4,i8)
        i8 = mod(i8,i4)
        i8 = mod(i8,i8)

        r = mod(r,r)
        d = mod(d,r)
        d = mod(r,d)
        d = mod(d,d)

        r = amod(r,amod(r,r))
        d = dmod(d,dmod(d,d))

      end program main
