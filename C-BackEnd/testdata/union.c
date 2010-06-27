# 1 "union.c"
union UNION {
int i;
char c;
};
int main()
{
auto union UNION uni;
(*(&(&(uni))->i)) = (1);
return (&(uni))->i;
}

