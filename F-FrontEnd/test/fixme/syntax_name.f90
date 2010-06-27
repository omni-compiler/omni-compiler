      program main
      open: if (a .eq. 2) then
        a = 3
      endif open
      close: if (a .eq. 2) then
        a = 3
      endif close
      case: if (a .eq. 2) then
        a = 3
      endif case
      implicitnone: if (a .eq. 2) then
        a = 3
      endif implicitnone
      read: if (a .eq. 2) then
        a = 3
      endif read
      casedefault: if (a .eq. 2) then
        a = 3
      endif casedefault
      end program
