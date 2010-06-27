      program main
      casedefault = 1
      implicitnone = 1
      open: if (a .eq. 2) then
        a = 3
      endif open
      close: if (a .eq. 2) then
        a = 3
      endif close
      read: if (a .eq. 2) then
        a = 3
      endif read
      case: if (a .eq. 2) then
        a = 3
      endif case
      casedefault: if (a .eq. 2) then
        a = 3
      endif casedefault
      implicit: if (a .eq. 2) then
        a = 3
      endif implicit
      implicitnone: if (a .eq. 2) then
        a = 3
      endif implicitnone
      end program
