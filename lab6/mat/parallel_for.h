#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads);

#endif
