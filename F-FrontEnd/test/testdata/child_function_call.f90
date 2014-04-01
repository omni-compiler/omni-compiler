      program testch
          if (ch().eq."hello") continue
      contains
          function ch ()
              character(len=10) :: ch
              ch ="hello"
          end function ch
      end program testch
