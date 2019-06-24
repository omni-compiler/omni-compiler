#pragma xmp template t(0:3)
#pragma xmp nodes p(1)
#pragma xmp distribute t(block) onto p

typedef struct QCDSpinor {
  double v[100];
} QCDSpinor_t;

static void norm2_t(QCDSpinor_t w[4])
{
#pragma xmp align w[j] with t(j)
  int tmp;

#pragma xmp loop (i) on t(i)
  for(int i=0;i<4;i++){
    for(int j=0;j<100;j++)
      tmp = w[i].v[j];
  }
}

int main(){
  QCDSpinor_t w[4];
  norm2_t(w);
  return 0;
}
