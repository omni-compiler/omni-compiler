int main(){
  int a[4];
#pragma xmp nodes p[*]
  {
#pragma xmp template t[4]
#pragma xmp distribute t[block] onto p
#pragma xmp align a[i] with t[i]
  }
  return 0;
}
