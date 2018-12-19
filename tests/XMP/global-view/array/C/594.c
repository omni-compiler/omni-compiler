#pragma xmp nodes p[4]
#pragma xmp template t[16]
#pragma xmp distribute t[block] onto p
int a[16];
#pragma xmp align a[i] with t[i]

int main(){

#pragma xmp array on t(:)
  a[:] = 0;

  return 0;
}
