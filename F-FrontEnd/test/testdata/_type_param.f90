      subroutine x()
         TYPE rat
           INTEGER  n, d
         END TYPE

         TYPE(rat), PARAMETER :: zero = rat(0,1)
         TYPE(rat), PARAMETER  :: one = rat(1,1)
         TYPE(rat)  r1, r2

       end subroutine x
