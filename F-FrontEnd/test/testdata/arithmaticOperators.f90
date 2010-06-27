      program main
        integer i
        integer j
        integer a
        logical b

        i = 1
        j = 2

        a = i + j
        a = i - j
        a = i * j
        a = i / j

        a = -i

        b = (i == j)
        b = (i .eq. j)
        b = (i /= j)
        b = (i .ne. j)
        b = (i <= j)
        b = (i .le. j)
        b = (i < j)
        b = (i .lt. j)
        b = (i >= j)
        b = (i .ge. j)
        b = (i > j)
        b = (i .gt. j)

        b = (b .and. b)
        b = (b .or. b)
        b = (b .eqv. b)
        b = (b .neqv. b)

        b = .not. b

      end program main
