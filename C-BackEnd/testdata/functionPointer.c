# 1 "functionPointer.c"
int callee(int n, int m)
{
return n + m;
}
void caller()
{
auto int (* func)(int, int) = callee;
func(3, 4);
}

