      PROGRAM main
        TYPE t
          INTEGER :: v
          PROCEDURE(REAL),NOPASS,POINTER :: p1 => null()
          PROCEDURE(cp),PASS(a),POINTER :: p2 => null()
        END TYPE t
        INTERFACE
          FUNCTION f(a)
            INTEGER :: f
            INTEGER :: a
          END FUNCTION f
          FUNCTION real(a)
            INTEGER :: real
            INTEGER :: a
          END FUNCTION real
          FUNCTION module(a)
            INTEGER :: module
            INTEGER :: a
          END FUNCTION module
          MODULE FUNCTION p(a)
            REAL :: p
            REAL :: a
          END FUNCTION p
        END INTERFACE

        PROCEDURE(REAL), POINTER :: g1 => null()

        PROCEDURE(f), POINTER :: g2 => f

        PROCEDURE(COMPLEX), POINTER :: g3

        PROCEDURE(REAL(KIND=4)), POINTER :: g4

        PROCEDURE(REAL*8), POINTER :: g5

        PROCEDURE(TYPE(t)), POINTER :: g6

        PROCEDURE(TYPE), POINTER :: g7

        PROCEDURE(module), POINTER :: g8

        PROCEDURE(), POINTER :: g9

        PROCEDURE(g2), POINTER :: g10

        INTEGER :: a
        COMPLEX :: c1, c2, c3

      CONTAINS
        FUNCTION TYPE(a, b)
          COMPLEX :: TYPE
          COMPLEX :: a
          COMPLEX :: b
        END FUNCTION TYPE

        SUBROUTINE sub(p1, p2)
          PROCEDURE (REAL) :: p1
          PROCEDURE (f) :: p2
          REAL :: r
          INTEGER :: i
          r = p1()
          i = p2(3)
        END SUBROUTINE sub

        FUNCTION cp(a)
          CLASS(t) :: a
          TYPE(t) :: cp
        END FUNCTION cp


      END PROGRAM main

      FUNCTION p(a)
        REAL :: p
        REAL :: a
      END FUNCTION p
