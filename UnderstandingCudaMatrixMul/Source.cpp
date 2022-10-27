#include "naiveGPUVis.h"

int main()
{
	NaiveGPU niaveGPU;
	if (niaveGPU.Construct(1000, 1000, 1, 1))
		niaveGPU.Start();

	return 0;
}