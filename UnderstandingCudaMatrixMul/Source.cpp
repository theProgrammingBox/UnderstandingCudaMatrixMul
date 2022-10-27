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

class Example : public olc::PixelGameEngine
{
private:
	int step;

	uint32_t inputEntries = 20;
	uint32_t inputFeatures = 6;
	uint32_t outputFeatures = 8;

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

public:
	Example()
	{
		sAppName = "Visualize";
	}

	bool OnUserCreate() override
	{
		step = 0;

		FillRandom(inputMatrix, inputMatrixSize);
		FillRandom(weightMatrix, weightMatrixSize);
		FillRandom(biasMatrix, outputMatrixSize);

		FillZero(outputMatrix, outputMatrixBytes);
		MatrixMulCPU(inputMatrix, weightMatrix, outputMatrix, inputEntries, inputFeatures, outputFeatures);
		PrintMatrix(outputMatrix, inputEntries, outputFeatures);
		return true;
	}

	bool OnUserUpdate(float fElapsedTime) override
	{
		if (GetKey(olc::Key::UP).bPressed) step++;
		if (GetKey(olc::Key::DOWN).bPressed) step--;

		const int scale = 20;
		const int hscale = scale / 2;
		const int hscalem = hscale - 1;
		int step2 = 0;

		uint32_t inputEntriesCeilBlocks = ceil((float)inputEntries / BLOCK_SIZE);
		uint32_t inputFeaturesCeilBlocks = ceil((float)inputFeatures / BLOCK_SIZE);
		uint32_t outputFeaturesCeilBlocks = ceil((float)outputFeatures / BLOCK_SIZE);
		uint32_t inputEntriesCeil = inputEntriesCeilBlocks * BLOCK_SIZE;
		uint32_t inputFeaturesCeil = inputFeaturesCeilBlocks * BLOCK_SIZE;
		uint32_t outputFeaturesCeil = outputFeaturesCeilBlocks * BLOCK_SIZE;
		Clear(BLACK);

		vf2d inputMatrixStartPos = vf2d(0, 0);
		vf2d weightMatrixStartPos = vf2d(scale * inputFeaturesCeil, scale * inputEntriesCeil);
		vf2d outputMatrixStartPos = vf2d(scale * inputFeaturesCeil, 0);

		vf2d inputMatrixSize = vf2d(scale * inputFeaturesCeil, scale * inputEntriesCeil);
		vf2d weightMatrixSize = vf2d(scale * outputFeaturesCeil, scale * inputFeaturesCeil);
		vf2d outputMatrixSize = vf2d(scale * outputFeaturesCeil, scale * inputEntriesCeil);

		FillRect(inputMatrixStartPos, inputMatrixSize, GREEN);
		FillRect(weightMatrixStartPos, weightMatrixSize, BLUE);
		FillRect(outputMatrixStartPos, outputMatrixSize, RED);

		for (int i = 0; i < inputEntries; i++)
		{
			for (int j = 0; j < inputFeatures; j++)
			{
				FillCircle(inputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
			for (int j = inputFeatures; j < inputFeaturesCeil; j++)
			{
				FillCircle(inputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem, BLACK);
			}
		}
		for (int i = inputEntries; i < inputEntriesCeil; i++)
		{
			for (int j = 0; j < inputFeaturesCeil; j++)
			{
				FillCircle(inputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem, BLACK);
			}
		}

		for (int i = 0; i < inputFeatures; i++)
		{
			for (int j = 0; j < outputFeatures; j++)
			{
				FillCircle(weightMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}

		for (int i = 0; i < inputEntries; i++)
		{
			for (int j = 0; j < outputFeatures; j++)
			{
				FillCircle(outputMatrixStartPos + vf2d(hscale + j * scale, hscale + i * scale), hscalem);
			}
		}
		/*uint32_t prevx1 = -100;
		uint32_t prevy1 = -100;
		uint32_t prevx2 = -100;
		uint32_t prevy2 = -100;
		uint32_t prevx3 = -100;
		uint32_t prevy3 = -100;

		for (uint32_t blockx = 0; blockx < ceil((float)inputEntries / BLOCK_SIZE); blockx++)
		{
			for (uint32_t blocky = 0; blocky < ceil((float)outputFeatures / BLOCK_SIZE / BLOCK_SIZE); blocky++)
			{
				int dblockx = scale * (outputFeatures + inputFeatures);
				FillRect(dblockx, 0, BLOCK_SIZE * scale, BLOCK_SIZE * scale, GREY);
				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
				{
					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
					{
						int x = blockx * BLOCK_SIZE + threadx;
						int y = blocky * BLOCK_SIZE + thready;
						int scaledx = dblockx + hscale + x * scale;
						int scaledy = hscale + y * scale;
						FillCircle(scaledx, scaledy, hscalem);
					}
				}


				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
				{
					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
					{
						if (++step2 > step) return true;
						int x = blockx * BLOCK_SIZE + threadx;
						int y = blocky * BLOCK_SIZE + thready;
						int scaledx = dblockx + hscale + x * scale;
						int scaledy = hscale + y * scale;
						FillCircle(scaledx, scaledy, hscalem, BLACK);
						FillCircle(prevx1, prevy1, hscalem);
						prevx1 = scaledx;
						prevy1 = scaledy;

						float sum = 0.0f;
						FillCircle(hscale + dblockx, hscale + BLOCK_SIZE * scale, hscalem);
						for (uint32_t k = 0; k < inputFeatures; k++)
						{
							if (++step2 > step) return true;
							sum += inputMatrix[y * inputFeatures + k] * weightMatrix[k * outputFeatures + x];
							int xx1 = hscale + k * scale;
							int yy1 = hscale + y * scale;
							FillCircle(xx1, yy1, hscalem, BLACK);
							FillCircle(prevx2, prevy2, hscalem);
							prevx2 = xx1;
							prevy2 = yy1;

							int xx2 = scale * inputFeatures + hscale + x * scale;
							int yy2 = scale * inputEntries + hscale + k * scale;
							FillCircle(xx2, yy2, hscalem, BLACK);
							FillCircle(prevx3, prevy3, hscalem);
							prevx3 = xx2;
							prevy3 = yy2;

							FillCircle(hscale + dblockx, hscale + BLOCK_SIZE * scale, hscalem, BLACK);
						}
					}
				}
			}
		}*/

		return true;
	}
};

//void visualizeWithCPU()
//{
//	for (uint32_t blockx = 0; blockx < ceil((float)inputEntries / BLOCK_SIZE; blockx++)
//	{
//		for (uint32_t blocky = 0; blocky < ceil((float)outputFeatures / BLOCK_SIZE) / BLOCK_SIZE; blocky++)
//		{
//			float inputBlock[BLOCK_SIZE][BLOCK_SIZE];
//				float weightBlock[BLOCK_SIZE][BLOCK_SIZE];
//				uint32_t inputBegin = blocky * (BLOCK_SIZE * inputFeatures);
//				uint32_t inputEnd = inputBegin + inputFeatures;
//				uint32_t weightBegin = blockx * BLOCK_SIZE;
//				uint32_t weightStep = BLOCK_SIZE * outputFeatures;
//
//				float sum = 0.0f;
//			for (uint32_t x = inputBegin, y = weightBegin; x < inputEnd; x += BLOCK_SIZE, y += weightStep)
//			{
//				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
//				{
//					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
//					{
//						inputBlock[threadx][thready] = inputMatrix[threadx * inputFeatures + x + thready] * (blocky * BLOCK_SIZE + thready < inputEntries&& x - inputBegin + threadx < inputFeatures);
//						weightBlock[threadx][thready] = weightMatrix[threadx * outputFeatures + y + thready] * (x - inputBegin + thready < inputFeatures&& blockx* BLOCK_SIZE + threadx < outputFeatures);
//					}
//				}
//
//
//				for (uint32_t threadx = 0; threadx < BLOCK_SIZE; threadx++)
//				{
//					for (uint32_t thready = 0; thready < BLOCK_SIZE; thready++)
//					{
//						for (uint32_t k = 0; k < BLOCK_SIZE; k++)
//						{
//							sum += inputBlock[threadCol][k] * weightBlock[k][threadRow];
//						}
//					}
//				}
//			}
//			uint32_t outputIndex = outputFeatures * BLOCK_SIZE * blocky + BLOCK_SIZE * blockx;
//			outputMatrix[outputIndex + outputFeatures * threadx + thready] = sum;
//		}
//	}
//}

int main()
{
	Example demo;
	if (demo.Construct(1000, 1000, 1, 1))
		demo.Start();

	return 0;
}