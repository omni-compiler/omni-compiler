  module m
    TYPE :: t
      INTEGER :: v
     CONTAINS
      PROCEDURE,NOPASS :: p => f
    END TYPE
  contains
    FUNCTION f(a)
      INTEGER :: f, a
    END FUNCTION f
    subroutine sub2(v, g)
      CLASS(t), POINTER :: v
      PROCEDURE(f) :: g
      PROCEDURE(f), POINTER :: h
      h => v%p
    end subroutine sub2
  end module m
  program main; use m; end program main
