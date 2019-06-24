#pragma xmp nodes p(2,2)

#pragma xmp template t(0:3, 0:3)
#pragma xmp distribute t(block, block) onto p

double v[4][4];
#pragma xmp align v[i][j] with t(j,i)

int main(){

  double tmp = 0;

  int i, j;
#pragma xmp loop (j, i) on t(j, i)
  for (i = 0; i < 4; i++){

#pragma omp parallel for
    for (j = 0; j < 4; j++)
      tmp += v[i][j];
  }

  return 0;
}
