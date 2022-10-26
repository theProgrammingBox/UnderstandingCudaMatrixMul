#pragma once
#include <iostream>
#include <cuda_runtime.h>

#include "Randoms.h"

using std::cout;
using std::endl;

const uint32_t BLOCK_SIZE = 16;
const uint32_t VECTOR_SIZE = 4;

static Random randoms;