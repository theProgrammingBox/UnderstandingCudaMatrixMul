#pragma once
#include "header.h"

class TileGPU : public olc::PixelGameEngine
{
private:
	int step = 0;
	float keyD = 0.0f;

	uint32_t inputEntries = 18;
	uint32_t inputFeatures = 17;
	uint32_t outputFeatures = 19;

	uint32_t inputMatrixSize = inputFeatures * inputEntries;
	uint32_t weightMatrixSize = inputFeatures * outputFeatures;
	uint32_t outputMatrixSize = outputFeatures * inputEntries;

	uint32_t inputMatrixBytes = sizeof(float) * inputMatrixSize;
	uint32_t weightMatrixBytes = sizeof(float) * weightMatrixSize;
	uint32_t outputMatrixBytes = sizeof(float) * outputMatrixSize;

	float* inputMatrix = (float*)malloc(inputMatrixBytes);
	float* weightMatrix = (float*)malloc(weightMatrixBytes);
	float* outputMatrix = (float*)malloc(outputMatrixBytes);
	float* biasMatrix = (float*)malloc(outputMatrixBytes);

	const int scale = 10;
	const int hscale = scale / 2;
	const int hscalem = hscale - 1;

	uint32_t inputEntriesCeilBlocks = ceil((float)inputEntries / BLOCK_SIZE);
	uint32_t inputFeaturesCeilBlocks = ceil((float)inputFeatures / BLOCK_SIZE);
	uint32_t outputFeaturesCeilBlocks = ceil((float)outputFeatures / BLOCK_SIZE);
	uint32_t inputEntriesCeil = inputEntriesCeilBlocks * BLOCK_SIZE;
	uint32_t inputFeaturesCeil = inputFeaturesCeilBlocks * BLOCK_SIZE;
	uint32_t outputFeaturesCeil = outputFeaturesCeilBlocks * BLOCK_SIZE;

	vf2d inputMatrixStartPos = vf2d(0, 0);
	vf2d weightMatrixStartPos = inputMatrixStartPos + vf2d(scale * inputFeatures, scale * inputEntriesCeil);
	vf2d outputMatrixStartPos = inputMatrixStartPos + vf2d(scale * inputFeatures, 0);
	vf2d inputBlockMatrixStartPos = outputMatrixStartPos + vf2d(scale * outputFeaturesCeil, 0);
	vf2d weightBlockMatrixStartPos = inputBlockMatrixStartPos + vf2d(scale * BLOCK_SIZE, 0);
	vf2d blockSize = vf2d(BLOCK_SIZE * scale, BLOCK_SIZE * scale);
	vf2d prevBlockPos = vf2d(-1000, -1000);
	vf2d prevThreadPos = vf2d(-1000, -1000);
	vf2d prevInputPos = vf2d(-1000, -1000);
	vf2d prevWeightPos = vf2d(-1000, -1000);
	vf2d prevOutputPos = vf2d(-1000, -1000);
	vf2d prevInputBlockPos = vf2d(-1000, -1000);
	vf2d prevWeightBlockPos = vf2d(-1000, -1000);

public:
	TileGPU()
	{
		sAppName = "Visualize naive GPU";
	}

	bool OnUserCreate() override
	{
		FillRandom(inputMatrix, inputMatrixSize);
		FillRandom(weightMatrix, weightMatrixSize);
		FillRandom(biasMatrix, outputMatrixSize);

		FillZero(outputMatrix, outputMatrixBytes);
		MatrixMulCPU(inputMatrix, weightMatrix, outputMatrix, inputEntries, inputFeatures, outputFeatures);
		PrintMatrix(outputMatrix, inputEntries, outputFeatures);

		DrawRect(inputMatrixStartPos, vf2d(inputFeatures, inputEntries) * scale, GREEN);
		DrawRect(weightMatrixStartPos, vf2d(outputFeatures, inputFeatures) * scale, BLUE);
		DrawRect(outputMatrixStartPos, vf2d(outputFeatures, inputEntries) * scale, RED);
		DrawRect(inputBlockMatrixStartPos, blockSize, GREEN);
		DrawRect(weightBlockMatrixStartPos, blockSize, BLUE);

		for (uint32_t blocky = 0; blocky < inputEntriesCeilBlocks; blocky++)
		{
			for (uint32_t blockx = 0; blockx < outputFeaturesCeilBlocks; blockx++)
			{
				DrawRect(outputMatrixStartPos + vf2d(blockx, blocky) * blockSize, blockSize, DARK_GREY);
			}
		}

		for (int i = 0; i < inputEntries; i++)
		{
			for (int j = 0; j < inputFeatures; j++)
			{
				DrawCircle(inputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		for (int i = 0; i < inputFeatures; i++)
		{
			for (int j = 0; j < outputFeatures; j++)
			{
				DrawCircle(weightMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		for (int i = 0; i < inputEntriesCeil; i++)
		{
			for (int j = 0; j < outputFeaturesCeil; j++)
			{
				DrawCircle(outputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			for (int j = 0; j < BLOCK_SIZE; j++)
			{
				DrawCircle(inputBlockMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		for (int i = 0; i < BLOCK_SIZE; i++)
		{
			for (int j = 0; j < BLOCK_SIZE; j++)
			{
				DrawCircle(weightBlockMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetKey(olc::Key::SPACE).bHeld)
		{
			const float timef = 0.01f;
			if (GetKey(olc::Key::UP).bHeld)
			{
				keyD += fElapsedTime;
				if (keyD > timef)
				{
					step++;
					keyD = 0;
				}
			}
			if (GetKey(olc::Key::DOWN).bHeld)
			{
				keyD -= fElapsedTime;
				if (keyD < -timef)
				{
					step--;
					keyD = 0;
				}
			}
		}
		else
		{
			if (GetKey(olc::Key::UP).bPressed) step++;
			if (GetKey(olc::Key::DOWN).bPressed) step--;
		}
		if (step < 0) step = 0;


		vf2d blockDMPos;
		vf2d blockDPos;
		vf2d threadDPos;
		vf2d outputDPos;
		vf2d inputDPos;
		vf2d weightDPos;
		vf2d inputBlockDPos;
		vf2d weightBlockDPos;
		int step2 = 0;
		for (uint32_t blockx = 0; blockx < outputFeaturesCeilBlocks; blockx++)
		{
			for (uint32_t blocky = 0; blocky < inputEntriesCeilBlocks; blocky++)
			{
				blockDMPos = vf2d(blockx, blocky) * blockSize;
				blockDPos = outputMatrixStartPos + blockDMPos;

				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
				{
					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
					{
						if (blockx * BLOCK_SIZE + threadx < outputFeatures && blocky * BLOCK_SIZE + thready < inputEntries)
						{
							/*threadDPos = blockDPos + vf2d(threadx * scale + hscale, thready * scale + hscale);
							outputDPos = outputMatrixStartPos + blockDMPos + vf2d(threadx * scale + hscale, thready * scale + hscale);

							for (uint32_t k = 0; k < inputFeatures; k++)
							{
								inputDPos = inputMatrixStartPos + vf2d(k * scale + hscale, thready * scale + hscale + blockDMPos.y);

								weightDPos = weightMatrixStartPos + vf2d(threadx * scale + hscale + blockDMPos.x, k * scale + hscale);

								step2++;
								if (step2 > step) break;
							}*/
							inputDPos = inputMatrixStartPos + vf2d(threadx * scale + hscale, thready * scale + hscale + blockDMPos.y);
							inputBlockDPos = inputBlockMatrixStartPos + vf2d(threadx * scale + hscale, thready * scale + hscale);
							step2++;
						}
						if (step2 > step) break;
					}
					if (step2 > step) break;
				}
				if (step2 > step) break;
			}
			if (step2 > step) break;
		}
		if (step == step2) step--;
		DrawRect(prevBlockPos, blockSize, DARK_GREY);
		DrawRect(blockDPos, blockSize, YELLOW);
		DrawCircle(prevThreadPos, hscalem);
		DrawCircle(threadDPos, hscalem, DARK_GREY);
		DrawCircle(prevOutputPos, hscalem);
		DrawCircle(outputDPos, hscalem, DARK_GREY);
		DrawCircle(prevInputPos, hscalem);
		DrawCircle(inputDPos, hscalem, DARK_GREY);
		DrawCircle(prevWeightPos, hscalem);
		DrawCircle(weightDPos, hscalem, DARK_GREY);
		DrawCircle(prevInputBlockPos, hscalem);
		DrawCircle(inputBlockDPos, hscalem, DARK_GREY);
		DrawCircle(prevWeightBlockPos, hscalem);
		DrawCircle(weightBlockDPos, hscalem, DARK_GREY);
		prevBlockPos = blockDPos;
		prevThreadPos = threadDPos;
		prevOutputPos = outputDPos;
		prevInputPos = inputDPos;
		prevWeightPos = weightDPos;
		prevInputBlockPos = inputBlockDPos;
		prevWeightBlockPos = weightBlockDPos;

		return true;
	}
};

//__global__ void matrixMultiplyKernel(float* input, float* weight, float* output, int inputEntries, int inputFeatures, int outputFeatures)
//{
//	int outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
//	int inputIndex = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (outputIndex < outputFeatures && inputIndex < inputEntries)
//	{
//		float sum = 0;
//		for (int k = 0; k < inputFeatures; k++)
//		{
//			sum += input[inputIndex * inputFeatures + k] * weight[outputIndex * inputFeatures + k];
//		}
//		output[outputIndex * inputEntries + inputIndex] = sum;
//	}
//}
//
//int main()
//{
//	TileGPU demo;
//	float* gpuInput;
//	float* gpuWeight;
//	float* gpuOutput;
//	cudaMalloc(&gpuInput, sizeof(float) * demo.inputEntries * demo.inputFeatures);
//	cudaMalloc(&gpuWeight, sizeof(float) * demo.inputFeatures * demo.outputFeatures);
//	cudaMalloc(&gpuOutput, sizeof(float) * demo.inputEntries * demo.outputFeatures);
//	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
//	dim3 thread(demo.outputFeaturesCeilBlocks, demo.inputEntriesCeilBlocks);
//	cudaMemcpy(gpuInput, demo.input, sizeof(float) * demo.inputEntries * demo.inputFeatures, cudaMemcpyHostToDevice);
//	cudaMemcpy(gpuWeight, demo.weight, sizeof(float) * demo.inputFeatures * demo.outputFeatures, cudaMemcpyHostToDevice);
//	matrixMultiplyKernel <<<thread, block>>> (demo.input, demo.weight, demo.output, demo.inputEntries, demo.inputFeatures, demo.outputFeatures);
//	cudaMemcpy(demo.output, gpuOutput, sizeof(float) * demo.inputEntries * demo.outputFeatures, cudaMemcpyDeviceToHost);
//	cudaFree(gpuInput);
//	cudaFree(gpuWeight);
//	cudaFree(gpuOutput);
//	if (demo.Construct(800, 800, 1, 1))
//		demo.Start();
//
//	return 0;
//}