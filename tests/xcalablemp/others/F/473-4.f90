  intrinsic command_argument_count
  n = command_argument_count()
  print *, "Number of arguments are:", n
  end

!! 2015.12.01
!! sh-4.1$ xmpf90 bug473-4.f90
!! "bug473-4.f90", line 2: compiler error: compile_intrinsic_call: not intrinsic symbol
!! /home/iwashita/Project/OMNI-test/libexec/omni_common_lib.sh: line 41: 29410 Aborted                 (core dumped) ${@+"$@"}


