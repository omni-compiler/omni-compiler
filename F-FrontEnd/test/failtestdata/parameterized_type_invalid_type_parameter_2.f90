       PROGRAM main
         TYPE t (k)
           INTEGER, KIND :: k = kind(4)
           INTEGER, LEN :: k = 10
           INTEGER (KIND=k) :: v
         END TYPE t
       END PROGRAM main
