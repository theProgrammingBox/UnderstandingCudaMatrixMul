#pragma once
#include "header.h"

class NaiveGPU : public olc::PixelGameEngine
{
private:
	int step = 0;
	float keyD = 0.0f;

	uint32_t inputEntries = 2;
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

	const int scale = 14;
	const int hscale = scale / 2;
	const int hscalem = hscale - 1;

	uint32_t inputEntriesCeilBlocks = ceil((float)inputEntries / BLOCK_SIZE);
	uint32_t inputFeaturesCeilBlocks = ceil((float)inputFeatures / BLOCK_SIZE);
	uint32_t outputFeaturesCeilBlocks = ceil((float)outputFeatures / BLOCK_SIZE);
	uint32_t inputEntriesCeil = inputEntriesCeilBlocks * BLOCK_SIZE;
	uint32_t inputFeaturesCeil = inputFeaturesCeilBlocks * BLOCK_SIZE;
	uint32_t outputFeaturesCeil = outputFeaturesCeilBlocks * BLOCK_SIZE;

	/*vf2d inputMatrixStartPos = vf2d(0, 0);
	vf2d weightMatrixStartPos = vf2d(scale * inputFeaturesCeil, scale * inputEntriesCeil);
	vf2d outputMatrixStartPos = vf2d(scale * inputFeaturesCeil, 0);*/
	vf2d inputMatrixStartPos = vf2d(scale * outputFeaturesCeil, scale * BLOCK_SIZE);
	vf2d weightMatrixStartPos = vf2d(0, scale * inputEntriesCeil + scale * BLOCK_SIZE);
	vf2d outputMatrixStartPos = vf2d(0, scale * BLOCK_SIZE);
	vf2d blockSize = vf2d(BLOCK_SIZE * scale, BLOCK_SIZE * scale);
	vf2d prevBlockPos = vf2d(-1000, -1000);
	vf2d prevThreadPos = vf2d(-1000, -1000);
	vf2d previnputPos = vf2d(-1000, -1000);
	vf2d prevWeightPos = vf2d(-1000, -1000);

public:
	NaiveGPU()
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
		for (uint32_t blockx = 0; blockx < outputFeaturesCeilBlocks; blockx++)
		{
			for (uint32_t blocky = 0; blocky < inputEntriesCeilBlocks; blocky++)
			{
				DrawRect(outputMatrixStartPos + vf2d(blockx, blocky) * blockSize, blockSize, RED);
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

		for (int i = 0; i < inputEntries; i++)
		{
			for (int j = 0; j < outputFeatures; j++)
			{
				DrawCircle(outputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetKey(olc::Key::SPACE).bHeld)
		{
			const float timef = 0.00f;
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

		int step2 = 0;
		for (uint32_t blockx = 0; blockx < outputFeaturesCeilBlocks; blockx++)
		{
			for (uint32_t blocky = 0; blocky < inputEntriesCeilBlocks; blocky++)
			{
				vf2d blockDMPos = vf2d(blockx, blocky) * blockSize;
				vf2d blockDPos = outputMatrixStartPos + blockDMPos;
				DrawRect(prevBlockPos, blockSize, RED);
				DrawRect(blockDPos, blockSize, YELLOW);
				prevBlockPos = blockDPos;

				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
				{
					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
					{
						if (blockx * BLOCK_SIZE + threadx < outputFeatures && blocky * BLOCK_SIZE + thready < inputEntries)
						{
							vf2d threadDPos = blockDPos + vf2d(threadx * scale + hscale, thready * scale + hscale);
							DrawCircle(prevThreadPos, hscalem);
							DrawCircle(threadDPos, hscalem, DARK_GREY);
							prevThreadPos = threadDPos;

							for (uint32_t k = 0; k < inputFeatures; k++)
							{
								vf2d inputDPos = inputMatrixStartPos + vf2d(k * scale + hscale, thready * scale + hscale + blockDMPos.y);
								DrawCircle(previnputPos, hscalem);
								DrawCircle(inputDPos, hscalem, DARK_GREY);
								previnputPos = inputDPos;

								vf2d weightDPos = weightMatrixStartPos + vf2d(threadx * scale + hscale + blockDMPos.x, k * scale + hscale);
								DrawCircle(prevWeightPos, hscalem);
								DrawCircle(weightDPos, hscalem, DARK_GREY);
								prevWeightPos = weightDPos;

								step2++;
								if (step2 > step) break;
							}
						}
						if (step2 > step) break;
					}
					if (step2 > step) break;
				}
				if (step2 > step) break;
			}
			if (step2 > step) break;
		}
		if (step == step2) step = step2 - 1;

		return true;
	}
};