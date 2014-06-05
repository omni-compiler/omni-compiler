#include<stdio.h>
#include<string.h>

#define MACRO_CORRECT "happy guppy"

int main()
{
  char work[] = DEFINED_AT_COMPILE_TIME;
  
  if (strcmp(work, MACRO_CORRECT) == 0)
    return 0;

  /// error!!
  fprintf(stderr, "Error. work=\"%s\", MACRO_CORRECT=\"%s\"\n", work, MACRO_CORRECT);
  return 1;
}

