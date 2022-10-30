#include "naiveGPUVis.h"
#include "tileGPUVis.h"

int main()
{
	int option = 1;
	if (option == 0)
	{
		NaiveGPU naive;
		if (naive.Construct(1000, 1000, 1, 1))
			naive.Start();
	}
	else if (option == 1)
	{
		TileGPU tile;
		if (tile.Construct(1000, 1000, 1, 1))
			tile.Start();
	}

	return 0;
}