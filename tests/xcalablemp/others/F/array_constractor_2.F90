      program main
#if defined(__GNUC__) && (6 < __GNUC__ || 6 == __GNUC__ && 1 < __GNUC_MINOR__) \
  || defined(__INTEL_COMPILER) && (1600 < __INTEL_COMPILER)
        real, dimension(3)   :: vector3d
        character(LEN=10), dimension(2) :: char2d
        vector3d = [ REAL :: 1.0, 2.0, 3.0 * 2 ]
        char2d = [ CHARACTER(LEN= 3) :: "abc" , "NG "  ]
        char2d = [ CHARACTER(LEN=10) :: "abce", "PASS" ]
        if(vector3d(3).eq.6.0) then
          print *, char2d(2)
        else
          print *, 'NG'
          call exit(1)
        end if
#else
PRINT *, 'SKIPPED'
#endif
      end

