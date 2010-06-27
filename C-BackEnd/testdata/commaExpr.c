# 1 "commaExpr.c"
int func(void)
{
int n = 3;
int m = 5;
n = (1, 2, 3, 4, m);
return 0;
}

