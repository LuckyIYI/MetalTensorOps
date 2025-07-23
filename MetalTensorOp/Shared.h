#ifndef shared_h
#define shared_h

#include <simd/simd.h>

#ifndef __METAL__
#import <Metal/MTLTypes.h>
#endif


typedef struct {
    MTLResourceID weight[8];
    MTLResourceID bias[8];

} TensorArguments;



#endif /* shared_h */
