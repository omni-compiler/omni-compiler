      MODULE mod
        TYPE function
          INTEGER :: v
        END TYPE function
       CONTAINS
        FUNCTION f(a)
        END FUNCTION f

        ELEMENTAL FUNCTION g(a)
          INTENT(IN) :: a
        END FUNCTION g

        INTEGER FUNCTION h(a)
        END FUNCTION

        ELEMENTAL INTEGER FUNCTION i(a)
          INTENT(IN) :: a
        END FUNCTION

        INTEGER ELEMENTAL FUNCTION j(a)
          INTENT(IN) :: a
        END FUNCTION

        ELEMENTAL INTEGER PURE FUNCTION k(a)
          INTENT(IN) :: a
        END FUNCTION

        ELEMENTAL PURE INTEGER FUNCTION l(a)
          INTENT(IN) :: a
        END FUNCTION

        ELEMENTAL PURE INTEGER FUNCTION m(a)
          INTENT(IN) :: a
        END FUNCTION

        TYPE(function) FUNCTION n(a)
          INTEGER :: a
        END FUNCTION

        ELEMENTAL TYPE(function) FUNCTION o(a)
          INTENT(IN) :: a
        END FUNCTION

        TYPE(function) ELEMENTAL FUNCTION p(a)
          INTENT(IN) :: a
        END FUNCTION
      END MODULE mod

      PROGRAM main; USE mod; END PROGRAM main
