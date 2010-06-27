# 1 "test.c"
int func(void)
{
int a = 1;
int b = 2;
int c = 3;
a = (b = c, - b, ~ c, c = (b--), b > c);
return 0;
}

