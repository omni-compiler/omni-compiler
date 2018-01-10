      MODULE m
        TYPE t
          INTEGER :: v
         CONTAINS
          PROCEDURE :: f, f2
        END TYPE t

        INTERFACE gen
          MODULE PROCEDURE :: f
          MODULE PROCEDURE g
          PROCEDURE :: h
          PROCEDURE i
        END INTERFACE gen

        INTERFACE
          MODULE FUNCTION mf()
            INTEGER :: mf
          END FUNCTION mf
        END INTERFACE
       CONTAINS
        FUNCTION f(a)
          INTEGER :: f
          CLASS(t) :: a
          f = a%v
        END FUNCTION f
        FUNCTION f2(a)
          INTEGER :: f2
          CLASS(t) :: a
          f2 = a%v * 3
        END FUNCTION f
        FUNCTION g(a)
          REAL :: g, a
          g = a
        END FUNCTION g
        FUNCTION h(a)
          COMPLEX :: h, a
          h = a
        END FUNCTION h
        FUNCTION i(a)
          TYPE(t) :: i, a
          i%v = a%v
        END FUNCTION i
        
        MODULE PROCEDURE mf
          mf = 1
        END PROCEDURE
      END MODULE m
      
