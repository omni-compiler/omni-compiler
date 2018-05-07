      INTERFACE g
        ELEMENTAL FUNCTION f(a)
          INTEGER :: f
          INTEGER, VALUE :: a
        END FUNCTION f
      END INTERFACE g
      PRINT *, g((/1,2,3/))
      END

