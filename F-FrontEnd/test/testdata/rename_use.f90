      MODULE rat_arith
         TYPE rat
           INTEGER  n, d
         END TYPE

         TYPE(rat), PRIVATE, PARAMETER :: zero = rat(0,1)
         TYPE(rat), PUBLIC, PARAMETER  :: one = rat(1,1)
         TYPE(rat)  r1, r2
         NAMELIST  /nml_rat/  r1, r2
      END MODULE

      PROGRAM Mine
         USE rat_arith, ONLY: rat, one_rat => one, r1, r2, nml_rat

         READ *, r2; r1 = one_rat
         WRITE( *, NML = nml_rat)
       END PROGRAM Mine
