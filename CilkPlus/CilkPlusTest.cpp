#include <cilk/cilk.h> 
#include <cilk/reducer_list.h> 
#include <list> 
#include <iostream>



cilk::reducer< cilk::op_add<int> > fib_sum(0);
using namespace std;

void fib_with_reducer_internal(int n)
{
    if (n < 2)
    {
        *fib_sum += n;
    }
    else
    {
        cilk_spawn fib_with_reducer_internal(n-1);
        fib_with_reducer_internal(n-2);
        cilk_sync;
    }
}



int main(int argc, char const *argv[])
{
	fib_with_reducer_internal(10);
	cout << fib_sum << endl;
	return 0;
}