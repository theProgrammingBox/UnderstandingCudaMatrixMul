#include "header.h"

void FillRandom(float* matrix, uint32_t matrixSize)
{
	for (uint32_t i = matrixSize; i--;) matrix[i] = randoms.normalRand();
}

void FillZero(float* matrix, uint32_t matrixBytes)
{
	memset(matrix, 0, matrixBytes);
}

void PrintMatrix(float* matrix, uint32_t matrixEntries, uint32_t matrixFeatures)
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

void MatrixMulCPU(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputEntries, uint32_t inputFeatures, uint32_t outputFeatures)
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

__global__ void matrixMulGPU1(float* inputMatrix, float* weightMatrix, float* outputMatrix, uint32_t inputEntries, uint32_t inputFeatures, uint32_t outputFeatures)
{
	uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < inputEntries && col < outputFeatures)
	{
		for (uint32_t i = 0; i < inputFeatures; i++)
		{
			outputMatrix[row * outputFeatures + col] += inputMatrix[row * inputFeatures + i] * weightMatrix[i * outputFeatures + col];
		}
	}
}


int main()
{
	uint32_t inputEntries = 2;
	uint32_t inputFeatures = 4;
	uint32_t outputFeatures = 1;

	uint32_t inputMatrixSize = inputFeatures * inputEntries;
	uint32_t weightMatrixSize = inputFeatures * outputFeatures;
	uint32_t outputMatrixSize = outputFeatures * inputEntries;

	uint32_t inputMatrixBytes = sizeof(float) * inputMatrixSize;
	uint32_t weightMatrixBytes = sizeof(float) * weightMatrixSize;
	uint32_t outputMatrixBytes = sizeof(float) * outputMatrixSize;

	float* inputMatrix = (float*)malloc(inputMatrixBytes);
	float* weightMatrix = (float*)malloc(weightMatrixBytes);
	float* outputMatrix = (float*)malloc(outputMatrixBytes);

	FillRandom(inputMatrix, inputMatrixSize);
	FillRandom(weightMatrix, weightMatrixSize);

	float* gpuInputMatrix;
	float* gpuWeightMatrix;
	float* gpuOutputMatrix;

	cudaMalloc((void**)&gpuInputMatrix, inputMatrixBytes);
	cudaMalloc((void**)&gpuWeightMatrix, weightMatrixBytes);
	cudaMalloc((void**)&gpuOutputMatrix, outputMatrixBytes);

	dim3 threads, grid;



	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);

	FillZero(outputMatrix, outputMatrixBytes);
	MatrixMulCPU(inputMatrix, weightMatrix, outputMatrix, inputEntries, inputFeatures, outputFeatures);
	
	cudaEventCreate(&stop);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "CPU time: " << elapsedTime << " ms" << endl;
	PrintMatrix(outputMatrix, inputEntries, outputFeatures);

	

	cudaEventCreate(&start);
	cudaEventRecord(start, 0);
	
	FillZero(outputMatrix, outputMatrixBytes);
	cudaMemcpy(gpuInputMatrix, inputMatrix, inputMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuWeightMatrix, weightMatrix, weightMatrixBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuOutputMatrix, outputMatrix, outputMatrixBytes, cudaMemcpyHostToDevice);
	//

	return 0;
}