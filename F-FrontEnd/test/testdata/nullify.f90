      program main
        type tt
                integer a
                integer, pointer :: b
        end type
        integer, target  :: i
        integer, pointer :: j
        integer, pointer :: k
        type(tt) :: x

        write (*,*) j
        write (*,*) k
        NULLIFY (j)
        NULLIFY (j, k, x%b)
      end
