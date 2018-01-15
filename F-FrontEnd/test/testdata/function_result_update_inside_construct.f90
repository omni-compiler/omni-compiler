      FUNCTION f(a)
        INTEGER:: f
        CLASS(*) :: a
        INTEGER, DIMENSION(1:3) :: b = (/1,2,3/)
        SELECT TYPE(a)
        TYPE IS(INTEGER)
          f = 1
        END SELECT
        BLOCK
          f = 2
        END BLOCK
        ASSOCIATE(z => 1) 
          f = 3
        END ASSOCIATE
      END FUNCTION f

      END
