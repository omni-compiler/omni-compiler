#include <stdio.h>
#include <stdlib.h>

// Check an existence of a "__omni_tmp__" directory when using xmpcc with --debug option.
// e.g.
//   xmpcc --debug 364.c
//   if [ -d "__omni_tmp__" ]; then
//     echo "PASS"
//   else
//     echo "ERROR"
//   fi

#pragma xmp nodes p(1)
int main()
{
  return WEXITSTATUS(system("./364.sh"));
}
