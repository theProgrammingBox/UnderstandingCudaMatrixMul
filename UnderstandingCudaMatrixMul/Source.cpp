#include "naiveGPUVis.h"
#include "tileGPUVis.h"

int main()
{
	int option = 0;
	if (option == 0)
	{
		NaiveGPU naive;
		if (naive.Construct(800, 600, 1, 1))
			naive.Start();
	}
	else
	{
		TileGPU tile;
		if (tile.Construct(800, 600, 1, 1))
			tile.Start();
	}

	return 0;
}