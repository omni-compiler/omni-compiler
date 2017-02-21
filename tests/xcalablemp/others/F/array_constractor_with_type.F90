      program main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        type t
          integer :: v
        end type t
        type(t), dimension(3)   :: derivedType3d
        derivedType3d = [ t :: t(1), t(2), t(3) ]
        if(derivedType3d(3)%v.eq.3) then
          print *, 'PASS'
        else
          print *, 'NG'
          call exit(1)
        end if
#else
PRINT *, 'SKIPPED'
#endif
      end
