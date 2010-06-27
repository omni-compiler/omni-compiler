          program large_integers
            integer,parameter :: k5 = selected_int_kind(5)
            integer,parameter :: k15 = selected_int_kind(15)
            integer(kind=k5) :: i5
            integer(kind=k15) :: i15

            print *, huge(i5), huge(i15)

            ! The following inequalities are always true
            print *, huge(i5) >= 10_k5**5-1
            print *, huge(i15) >= 10_k15**15-1
          end program large_integers
