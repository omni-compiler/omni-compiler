# 1 "memberArray.c"
struct array_struct {
int array[1];
};
struct array_struct as;
void function()
{
(*((*(&(&(as))->array)) + (0))) = (0);
(*(((&(as))->array) + (0))) = (1);
}

