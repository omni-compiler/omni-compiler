#include <stdio.h>

int Func() {
  return printf("PASS\n");
}

int (*FP)(void)= Func;

int main() {
  return 5!=(*FP)();
}

