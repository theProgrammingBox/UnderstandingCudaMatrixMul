#pragma once
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <iostream>

#include "Randoms.h"

using namespace std;
using namespace olc;

const uint32_t BLOCK_SIZE = 16;
const uint32_t VECTOR_SIZE = 4;

static Random randoms;

static void FillRandom(float* matrix, uint32_t matrixSize)
{
	for (uint32_t i = matrixSize; i--;) matrix[i] = randoms.normalRand();
}

static void FillZero(float* matrix, uint32_t matrixBytes)
{
	memset(matrix, 0, matrixBytes);
}

static void PrintMatrix(float* matrix, uint32_t matrixEntries, uint32_t matrixFeatures)
{
	for (uint32_t i = 0; i < matrixEntries; i++)
	{
		for (uint32_t j = 0; j < matrixFeatures; j++)
		{
			cout << matrix[i * matrixFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

static void MatrixMulCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputEntries, uint32_t inputFeatures, uint32_t outputFeatures)
{
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			for (uint32_t k = 0; k < inputFeatures; k++)
			{
				outputMatrix[i * outputFeatures + j] += inputMatrix[i * inputFeatures + k] * weightMatrix[k * outputFeatures + j];
			}
		}
	}
}