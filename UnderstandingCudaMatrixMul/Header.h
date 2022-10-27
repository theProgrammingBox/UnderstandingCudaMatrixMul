#pragma once
#include <iostream>
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
//#include <cuda_runtime.h>

#include "Randoms.h"

//using std::cout;
//using std::endl;
using namespace std;
using namespace olc;

const uint32_t BLOCK_SIZE = 16;
const uint32_t VECTOR_SIZE = 4;

static Random randoms;