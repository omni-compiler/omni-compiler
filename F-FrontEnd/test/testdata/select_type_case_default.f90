      TYPE t
         INTEGER :: i
      END TYPE t

      CONTAINS
      SUBROUTINE s(obj)
        CLASS(t) ::  obj(3,3)
        SELECT TYPE(obj)
        CLASS DEFAULT
           PRINT *, obj(2,3)%i
        END SELECT
      END SUBROUTINE s
      END
