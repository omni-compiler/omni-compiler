module parameter
      integer, parameter :: lx = @lx@
      integer, parameter :: ly = @ly@
      integer, parameter :: lz = @lz@
      integer, parameter :: lstep = @lstep@
      integer, parameter :: lnpx = @lnpx@
      integer, parameter :: lnpy = @lnpy@
      integer, parameter :: lnpz = @lnpz@
      integer, parameter :: llx = lx / lnpx
      integer, parameter :: lly = ly / lnpy
      integer, parameter :: llz = lz / lnpz
end module parameter
