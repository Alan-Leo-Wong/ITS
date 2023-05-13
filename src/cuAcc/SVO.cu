#include "../SVO.h"
#include "../MortonLUT.h"
#include "../utils/IO.hpp"
#include "../utils/Geometry.hpp"
#include "../utils/cuda/CUDAMath.hpp"
#include "../utils/cuda/CUDAUtil.cuh"
#include "../utils/cuda/CUDACheck.cuh"
#include "../utils/String.hpp"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>
#include <iomanip>

namespace morton {
	__constant__ uint32_t d_morton256_x[256] =
	{
		0x00000000,
		0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049, 0x00000200,
		0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249, 0x00001000,
		0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049, 0x00001200,
		0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249, 0x00008000,
		0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049, 0x00008200,
		0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249, 0x00009000,
		0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049, 0x00009200,
		0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249, 0x00040000,
		0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049, 0x00040200,
		0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249, 0x00041000,
		0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049, 0x00041200,
		0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249, 0x00048000,
		0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049, 0x00048200,
		0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249, 0x00049000,
		0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049, 0x00049200,
		0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249, 0x00200000,
		0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049, 0x00200200,
		0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249, 0x00201000,
		0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049, 0x00201200,
		0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249, 0x00208000,
		0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049, 0x00208200,
		0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249, 0x00209000,
		0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049, 0x00209200,
		0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249, 0x00240000,
		0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049, 0x00240200,
		0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249, 0x00241000,
		0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049, 0x00241200,
		0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249, 0x00248000,
		0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049, 0x00248200,
		0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249, 0x00249000,
		0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049, 0x00249200,
		0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249
	};

	__constant__ uint32_t d_morton256_y[256] =
	{
		0x00000000,
		0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092, 0x00000400,
		0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492, 0x00002000,
		0x00002002, 0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092, 0x00002400,
		0x00002402, 0x00002410, 0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492, 0x00010000,
		0x00010002, 0x00010010, 0x00010012, 0x00010080, 0x00010082, 0x00010090, 0x00010092, 0x00010400,
		0x00010402, 0x00010410, 0x00010412, 0x00010480, 0x00010482, 0x00010490, 0x00010492, 0x00012000,
		0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082, 0x00012090, 0x00012092, 0x00012400,
		0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490, 0x00012492, 0x00080000,
		0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092, 0x00080400,
		0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492, 0x00082000,
		0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092, 0x00082400,
		0x00082402, 0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492, 0x00090000,
		0x00090002, 0x00090010, 0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092, 0x00090400,
		0x00090402, 0x00090410, 0x00090412, 0x00090480, 0x00090482, 0x00090490, 0x00090492, 0x00092000,
		0x00092002, 0x00092010, 0x00092012, 0x00092080, 0x00092082, 0x00092090, 0x00092092, 0x00092400,
		0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482, 0x00092490, 0x00092492, 0x00400000,
		0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090, 0x00400092, 0x00400400,
		0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492, 0x00402000,
		0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092, 0x00402400,
		0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492, 0x00410000,
		0x00410002, 0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092, 0x00410400,
		0x00410402, 0x00410410, 0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492, 0x00412000,
		0x00412002, 0x00412010, 0x00412012, 0x00412080, 0x00412082, 0x00412090, 0x00412092, 0x00412400,
		0x00412402, 0x00412410, 0x00412412, 0x00412480, 0x00412482, 0x00412490, 0x00412492, 0x00480000,
		0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082, 0x00480090, 0x00480092, 0x00480400,
		0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490, 0x00480492, 0x00482000,
		0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092, 0x00482400,
		0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492, 0x00490000,
		0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092, 0x00490400,
		0x00490402, 0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492, 0x00492000,
		0x00492002, 0x00492010, 0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092, 0x00492400,
		0x00492402, 0x00492410, 0x00492412, 0x00492480, 0x00492482, 0x00492490, 0x00492492
	};

	__constant__ uint32_t d_morton256_z[256] =
	{
		0x00000000,
		0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124, 0x00000800,
		0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924, 0x00004000,
		0x00004004, 0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124, 0x00004800,
		0x00004804, 0x00004820, 0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924, 0x00020000,
		0x00020004, 0x00020020, 0x00020024, 0x00020100, 0x00020104, 0x00020120, 0x00020124, 0x00020800,
		0x00020804, 0x00020820, 0x00020824, 0x00020900, 0x00020904, 0x00020920, 0x00020924, 0x00024000,
		0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104, 0x00024120, 0x00024124, 0x00024800,
		0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920, 0x00024924, 0x00100000,
		0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124, 0x00100800,
		0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924, 0x00104000,
		0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124, 0x00104800,
		0x00104804, 0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924, 0x00120000,
		0x00120004, 0x00120020, 0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124, 0x00120800,
		0x00120804, 0x00120820, 0x00120824, 0x00120900, 0x00120904, 0x00120920, 0x00120924, 0x00124000,
		0x00124004, 0x00124020, 0x00124024, 0x00124100, 0x00124104, 0x00124120, 0x00124124, 0x00124800,
		0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904, 0x00124920, 0x00124924, 0x00800000,
		0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120, 0x00800124, 0x00800800,
		0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924, 0x00804000,
		0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124, 0x00804800,
		0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924, 0x00820000,
		0x00820004, 0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124, 0x00820800,
		0x00820804, 0x00820820, 0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924, 0x00824000,
		0x00824004, 0x00824020, 0x00824024, 0x00824100, 0x00824104, 0x00824120, 0x00824124, 0x00824800,
		0x00824804, 0x00824820, 0x00824824, 0x00824900, 0x00824904, 0x00824920, 0x00824924, 0x00900000,
		0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104, 0x00900120, 0x00900124, 0x00900800,
		0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920, 0x00900924, 0x00904000,
		0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124, 0x00904800,
		0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924, 0x00920000,
		0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124, 0x00920800,
		0x00920804, 0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924, 0x00924000,
		0x00924004, 0x00924020, 0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124, 0x00924800,
		0x00924804, 0x00924820, 0x00924824, 0x00924900, 0x00924904, 0x00924920, 0x00924924
	};

	// LUT for Morton3D decode X
	__constant__ uint_fast8_t d_Morton3D_decode_x_512[512] =
	{
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7,
		4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7
	};

	// LUT for Morton3D decode Y
	__constant__ uint_fast8_t d_Morton3D_decode_y_512[512] =
	{
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
		2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7,
		4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5,
		6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7
	};

	// LUT for Morton3D decode Z
	__constant__ uint_fast8_t d_Morton3D_decode_z_512[512] =
	{
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 5, 5, 5, 5,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7,
		6, 6, 6, 6, 7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7, 7
	};

	__constant__ short int neighbor_LUTparent[8][27] =
	{
		{0, 1,  1,  3,  4,  4,  3,  4,  4,
		 9, 10, 10, 12, 13, 13, 12, 13, 13,
		 9, 10, 10, 12, 13, 13, 12, 13, 13},

		{1,  1,  2,  4,  4,  5,  4,  4,  5,
		 10, 10, 11, 13, 13, 14, 13, 13, 14,
		 10, 10, 11, 13, 13, 14, 13, 13, 14},

		{3,  4,  4,  3,  4,  4,  6,  7,  7,
		 12, 13, 13, 12, 13, 13, 15, 16, 16,
		 12, 13, 13, 12, 13, 13, 15, 16, 16},

		{4,  4,  5,  4,  4,  5,  7,  7,  8,
		 13, 13, 14, 13, 13, 14, 16, 16, 17,
		 13, 13, 14, 13, 13, 14, 16, 16, 17},

		{9,  10, 10, 12, 13, 13, 12, 13, 13,
		 9,  10, 10, 12, 13, 13, 12, 13, 13,
		 18, 19, 19, 21, 22, 22, 21, 22, 22},

		{10, 10, 11, 13, 13, 14, 13, 13, 14,
		 10, 10, 11, 13, 13, 14, 13, 13, 14,
		 19, 19, 20, 22, 22, 23, 22, 22, 23},

		{12, 13, 13, 12, 13, 13, 15, 16, 16,
		 12, 13, 13, 12, 13, 13, 15, 16, 16,
		 21, 22, 22, 21, 22, 22, 24, 25, 25},

		{13, 13, 14, 13, 13, 14, 16, 16, 17,
		 13, 13, 14, 13, 13, 14, 16, 16, 17,
		 22, 22, 23, 22, 22, 23, 25, 25, 26}
	};

	__constant__ short int neighbor_LUTchild[8][27] =
	{
		{7, 6, 7, 5, 4, 5, 7, 6, 7,
		 3, 2, 3, 1, 0, 1, 3, 2, 3,
		 7, 6, 7, 5, 4, 5, 7, 6, 7},

		{6, 7, 6, 4, 5, 4, 6, 7, 6,
		 2, 3, 2, 0, 1, 0, 2, 3, 2,
		 6, 7, 6, 4, 5, 4, 6, 7, 6},

		{5, 4, 5, 7, 6, 7, 5, 4, 5,
		 1, 0, 1, 3, 2, 3, 1, 0, 1,
		 5, 4, 5, 7, 6, 7, 5, 4, 5,},

		{4, 5, 4, 6, 7, 6, 4, 5, 4,
		 0, 1, 0, 2, 3, 2, 0, 1, 0,
		 4, 5, 4, 6, 7, 6, 4, 5, 4},

		{3, 2, 3, 1, 0, 1, 3, 2, 3,
		 7, 6, 7, 5, 4, 5, 7, 6, 7,
		 3, 2, 3, 1, 0, 1, 3, 2, 3},

		{2, 3, 2, 0, 1, 0, 2, 3, 2,
		 6, 7, 6, 4, 5, 4, 6, 7, 6,
		 2, 3, 2, 0, 1, 0, 2, 3, 2},

		{1, 0, 1, 3, 2, 3, 1, 0, 1,
		 5, 4, 5, 7, 6, 7, 5, 4, 5,
		 1, 0, 1, 3, 2, 3, 1, 0, 1},

		{0, 1, 0, 2, 3, 2, 0, 1, 0,
		 4, 5, 4, 6, 7, 6, 4, 5, 4,
		 0, 1, 0, 2, 3, 2, 0, 1, 0}
	};

	// edge: 02 23 31 10   46 67 75 54   04 26 37 15 
	__constant__ short int d_edgeSharedLUT[48] =
	{
		 3, 4, 12, 13,
		 4, 7, 13, 16,
		 4, 5, 13, 14,
		 1, 4, 10, 13,

		 12, 13, 21, 22,
		 13, 16, 22, 25,
		 13, 14, 22, 23,
		 10, 13, 19, 22,

		 9, 10, 12, 13,
		 12, 13, 15, 16,
		 13, 14, 16, 17,
		 10, 11, 13, 14
	};

	__constant__ short int d_vertSharedLUT[64] =
	{
		0, 1, 3, 4, 9, 10, 12, 13,

		1, 2, 4, 5, 10, 11, 13 ,14,

		3, 4, 6, 7, 12, 13, 15, 16,

		4, 5, 7, 8, 13, 14, 16, 17,

		9, 10, 12, 13, 18, 19, 21, 22,

		10, 11, 13, 14, 19, 20, 22, 23,

		12, 13, 15, 16, 21, 22, 24, 25,

		13, 14, 16, 17, 22, 23, 25, 26
	};
}

namespace {
	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
}

template <typename T>
struct scanMortonFlag : public thrust::unary_function<T, T> {
	__host__ __device__ T operator()(const T& x) {
		// printf("%lu %d\n", b, (b >> 31) & 1);
		return (x >> 31) & 1;
	}
};

__global__ void surfaceVoxelize(const int nTris,
	const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3d* d_gridOrigin,
	const Eigen::Vector3d* d_unitVoxelSize,
	double* d_triangle_data,
	uint32_t* d_voxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	const Eigen::Vector3i surfaceVoxelGridSize = *d_surfaceVoxelGridSize;
	const Eigen::Vector3d unitVoxelSize = *d_unitVoxelSize;
	const Eigen::Vector3d gridOrigin = *d_gridOrigin;
	Eigen::Vector3d delta_p{ unitVoxelSize.x(), unitVoxelSize.y(), unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)
	while (tid < nTris) { // every thread works on specific triangles in its stride
		size_t t = tid * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using modelBBox
		Eigen::Vector3d v0 = Eigen::Vector3d(d_triangle_data[t], d_triangle_data[t + 1], d_triangle_data[t + 2]) - gridOrigin;
		Eigen::Vector3d v1 = Eigen::Vector3d(d_triangle_data[t + 3], d_triangle_data[t + 4], d_triangle_data[t + 5]) - gridOrigin;
		Eigen::Vector3d v2 = Eigen::Vector3d(d_triangle_data[t + 6], d_triangle_data[t + 7], d_triangle_data[t + 8]) - gridOrigin;
		// Edge vectors
		Eigen::Vector3d e0 = v1 - v0;
		Eigen::Vector3d e1 = v2 - v1;
		Eigen::Vector3d e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		Eigen::Vector3d n = e0.cross(e1).normalized();

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<Eigen::Vector3d> t_bbox_world(fminf(v0, fminf(v1, v2)), fmaxf(v0, fmaxf(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<Eigen::Vector3i> t_bbox_grid;
		t_bbox_grid.boxOrigin = clamp(
			Eigen::Vector3i((t_bbox_world.boxOrigin.x() / unitVoxelSize.x()), (t_bbox_world.boxOrigin.y() / unitVoxelSize.y()), (t_bbox_world.boxOrigin.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);
		t_bbox_grid.boxEnd = clamp(
			Eigen::Vector3i((t_bbox_world.boxEnd.x() / unitVoxelSize.x()), (t_bbox_world.boxEnd.y() / unitVoxelSize.y()), (t_bbox_world.boxEnd.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);

		// PREPARE PLANE TEST PROPERTIES
		Eigen::Vector3d c(0.0, 0.0, 0.0);
		if (n.x() > 0.0) { c.x() = unitVoxelSize.x(); }
		if (n.y() > 0.0) { c.y() = unitVoxelSize.y(); }
		if (n.z() > 0.0) { c.z() = unitVoxelSize.z(); }
		double d1 = n.dot((c - v0));
		double d2 = n.dot(((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		Eigen::Vector2d n_xy_e0(-1.0 * e0.y(), e0.x());
		Eigen::Vector2d n_xy_e1(-1.0 * e1.y(), e1.x());
		Eigen::Vector2d n_xy_e2(-1.0 * e2.y(), e2.x());
		if (n.z() < 0.0)
		{
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		double d_xy_e0 = (-1.0 * n_xy_e0.dot(Eigen::Vector2d(v0.x(), v0.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e0[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e0[1]);
		double d_xy_e1 = (-1.0 * n_xy_e1.dot(Eigen::Vector2d(v1.x(), v1.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e1[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e1[1]);
		double d_xy_e2 = (-1.0 * n_xy_e2.dot(Eigen::Vector2d(v2.x(), v2.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e2[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e2[1]);
		// YZ plane
		Eigen::Vector2d n_yz_e0(-1.0 * e0.z(), e0.y());
		Eigen::Vector2d n_yz_e1(-1.0 * e1.z(), e1.y());
		Eigen::Vector2d n_yz_e2(-1.0 * e2.z(), e2.y());
		if (n.x() < 0.0) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		double d_yz_e0 = (-1.0 * n_yz_e0.dot(Eigen::Vector2d(v0.y(), v0.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e0[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e0[1]);
		double d_yz_e1 = (-1.0 * n_yz_e1.dot(Eigen::Vector2d(v1.y(), v1.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e1[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e1[1]);
		double d_yz_e2 = (-1.0 * n_yz_e2.dot(Eigen::Vector2d(v2.y(), v2.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e2[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e2[1]);
		// ZX plane																							 													  
		Eigen::Vector2d n_zx_e0(-1.0 * e0.x(), e0.z());
		Eigen::Vector2d n_zx_e1(-1.0 * e1.x(), e1.z());
		Eigen::Vector2d n_zx_e2(-1.0 * e2.x(), e2.z());
		if (n.y() < 0.0) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		double d_xz_e0 = (-1.0 * n_zx_e0.dot(Eigen::Vector2d(v0.z(), v0.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e0[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e0[1]);
		double d_xz_e1 = (-1.0 * n_zx_e1.dot(Eigen::Vector2d(v1.z(), v1.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e1[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e1[1]);
		double d_xz_e2 = (-1.0 * n_zx_e2.dot(Eigen::Vector2d(v2.z(), v2.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e2[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (uint16_t z = t_bbox_grid.boxOrigin.z(); z <= t_bbox_grid.boxEnd.z(); z++) {
			for (uint16_t y = t_bbox_grid.boxOrigin.y(); y <= t_bbox_grid.boxEnd.y(); y++) {
				for (uint16_t x = t_bbox_grid.boxOrigin.x(); x <= t_bbox_grid.boxEnd.x(); x++) {
					// if (checkBit(voxel_table, location)){ continue; }
					// TRIANGLE PLANE THROUGH BOX TEST
					Eigen::Vector3d p(x * unitVoxelSize.x(), y * unitVoxelSize.y(), z * unitVoxelSize.z());
					double nDOTp = n.dot(p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0) { continue; }

					// PROJECTION TESTS
					// XY
					Eigen::Vector2d p_xy(p.x(), p.y());
					if ((n_xy_e0.dot(p_xy) + d_xy_e0) < 0.0) { continue; }
					if ((n_xy_e1.dot(p_xy) + d_xy_e1) < 0.0) { continue; }
					if ((n_xy_e2.dot(p_xy) + d_xy_e2) < 0.0) { continue; }

					// YZ
					Eigen::Vector2d p_yz(p.y(), p.z());
					if ((n_yz_e0.dot(p_yz) + d_yz_e0) < 0.0) { continue; }
					if ((n_yz_e1.dot(p_yz) + d_yz_e1) < 0.0) { continue; }
					if ((n_yz_e2.dot(p_yz) + d_yz_e2) < 0.0) { continue; }

					// XZ	
					Eigen::Vector2d p_zx(p.z(), p.x());
					if ((n_zx_e0.dot(p_zx) + d_xz_e0) < 0.0) { continue; }
					if ((n_zx_e1.dot(p_zx) + d_xz_e1) < 0.0) { continue; }
					if ((n_zx_e2.dot(p_zx) + d_xz_e2) < 0.0) { continue; }

					uint32_t mortonCode = morton::mortonEncode_LUT(x, y, z);
					atomicExch(d_voxelArray + mortonCode, mortonCode | E_MORTON_32_FLAG); // 最高位设置为1，代表这是个表面的voxel
				}
			}
		}
		tid += stride;
	}
}

void SparseVoxelOctree::meshVoxelize(const size_t& nModelTris,
	const vector<Triangle<V3d>>& modelTris,
	const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3d* d_unitVoxelSize,
	const Eigen::Vector3d* d_gridOrigin,
	thrust::device_vector<uint32_t>& d_CNodeMortonArray)
{
	thrust::device_vector<Eigen::Vector3d> d_triangleThrustVec;
	for (int i = 0; i < nModelTris; ++i)
	{
		d_triangleThrustVec.push_back(modelTris[i].p1);
		d_triangleThrustVec.push_back(modelTris[i].p2);
		d_triangleThrustVec.push_back(modelTris[i].p3);
	}
	double* d_triangleData = (double*)thrust::raw_pointer_cast(&(d_triangleThrustVec[0]));
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, d_surfaceVoxelGridSize,
		d_gridOrigin, d_unitVoxelSize, d_triangleData, d_CNodeMortonArray.data().get());
	getLastCudaError("Kernel 'surfaceVoxelize' launch failed!\n");
	//cudaDeviceSynchronize();
}


_CUDA_GENERAL_CALL_ uint32_t getParentMorton(const uint32_t morton)
{
	return (((morton >> 3) & 0xfffffff));
}

_CUDA_GENERAL_CALL_ bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{
	return getParentMorton(morton_1) == getParentMorton(morton_2);
}

__global__ void compactArray(const int n,
	const bool* d_isValidArray,
	const uint32_t* d_dataArray,
	const size_t* d_esumDataArray,
	uint32_t* d_pactDataArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n && d_isValidArray[tid])
		d_pactDataArray[d_esumDataArray[tid]] = d_dataArray[tid];
}

// 计算表面voxel共对应多少个八叉树节点同时设置父节点的莫顿码数组
__global__ void cpNumNodes(const size_t n,
	const uint32_t* d_pactDataArray,
	size_t* d_nNodesArray,
	uint32_t* d_parentMortonArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= 1 && tid < n)
	{
		if (isSameParent(d_pactDataArray[tid], d_pactDataArray[tid - 1])) d_nNodesArray[tid] = 0;
		else
		{
			const uint32_t parentMorton = getParentMorton(d_pactDataArray[tid]);
			d_parentMortonArray[parentMorton] = parentMorton | E_MORTON_32_FLAG;
			d_nNodesArray[tid] = 8;
		}
	}
}

__global__ void createNode_1(const size_t pactSize,
	const size_t* d_sumNodesArray,
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3d gridOrigin = *d_gridOrigin;
		const double width = *d_width;

		const size_t sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = (pactData & LOWER_3BIT_MASK);
		const uint32_t morton = (pactData & D_MORTON_32_FLAG); // 去除符号位的实际莫顿码
		// 得到mortonCode对应的实际存储节点的位置
		const uint32_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton::morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		tNode.width = width;

		d_begMortonArray[tid] = (morton / 8) * 8;

		//if (blockIdx.x == 15 && threadIdx.x == 126) printf("d_begMortonArray[%d] = %u\n", tid, (unsigned int)d_begMortonArray[tid]);
	}
}

__global__ void createNode_2(const size_t pactSize,
	const size_t d_preChildDepthTreeNodes, // 子节点层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t d_preDepthTreeNodes, // 当前层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t* d_sumNodesArray, // 这一层的节点数量inclusive scan数组
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray,
	SVONode* d_childArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3d gridOrigin = *d_gridOrigin;
		const double width = *d_width;

		const int sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = pactData & LOWER_3BIT_MASK;
		const uint32_t morton = pactData & D_MORTON_32_FLAG;
		const size_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton::morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		tNode.width = width;
		tNode.isLeaf = false;

		d_begMortonArray[tid] = (morton / 8) * 8;

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			tNode.childs[i] = d_preChildDepthTreeNodes + tid * 8 + i;
			d_childArray[tid * 8 + i].parent = d_preDepthTreeNodes + sumNodes + key;
		}
	}
}

__global__ void createRemainNode(const size_t nNodes,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	const uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	extern __shared__ uint32_t sh_begMortonArray[];
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < nNodes)
	{
		sh_begMortonArray[threadIdx.x / 8] = d_begMortonArray[tid / 8];

		__syncthreads();

		if (d_nodeArray[tid].mortonCode == 0)
		{
			const Eigen::Vector3d gridOrigin = *d_gridOrigin;
			const double width = *d_width;

			const uint32_t key = tid & LOWER_3BIT_MASK;
			const uint32_t morton = sh_begMortonArray[threadIdx.x / 8] + key;

			SVONode& tNode = d_nodeArray[tid];
			tNode.mortonCode = morton;

			morton::morton3D_32_decode(morton, x, y, z);
			tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
			tNode.width = width;
		}
	}
}

void SparseVoxelOctree::createOctree(const size_t& nModelTris, const vector<Triangle<V3d>>& modelTris, const AABox<Eigen::Vector3d>& modelBBox, const std::string& base_filename)
{
	assert(surfaceVoxelGridSize.x() >= 1 && surfaceVoxelGridSize.y() >= 1 && surfaceVoxelGridSize.z() >= 1);
	size_t gridCNodeSize = (size_t)morton::mortonEncode_LUT((uint16_t)(surfaceVoxelGridSize.x() - 1), (uint16_t)(surfaceVoxelGridSize.y() - 1), (uint16_t)(surfaceVoxelGridSize.z() - 1)) + 1;
	size_t gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
	Eigen::Vector3d unitVoxelSize = Eigen::Vector3d(modelBBox.boxWidth.x() / surfaceVoxelGridSize.x(),
		modelBBox.boxWidth.y() / surfaceVoxelGridSize.y(),
		modelBBox.boxWidth.z() / surfaceVoxelGridSize.z());
	double unitNodeWidth = unitVoxelSize.x();

	Eigen::Vector3i* d_surfaceVoxelGridSize;
	CUDA_CHECK(cudaMalloc((void**)&d_surfaceVoxelGridSize, sizeof(Eigen::Vector3i)));
	CUDA_CHECK(cudaMemcpy(d_surfaceVoxelGridSize, &surfaceVoxelGridSize, sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice));
	Eigen::Vector3d* d_gridOrigin;
	CUDA_CHECK(cudaMalloc((void**)&d_gridOrigin, sizeof(Eigen::Vector3d)));
	CUDA_CHECK(cudaMemcpy(d_gridOrigin, &modelBBox.boxOrigin, sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice));
	Eigen::Vector3d* d_unitVoxelSize;
	CUDA_CHECK(cudaMalloc((void**)&d_unitVoxelSize, sizeof(Eigen::Vector3d)));
	CUDA_CHECK(cudaMemcpy(d_unitVoxelSize, &unitVoxelSize, sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice));
	double* d_unitNodeWidth;
	CUDA_CHECK(cudaMalloc((void**)&d_unitNodeWidth, sizeof(double)));
	CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(double), cudaMemcpyHostToDevice));

	thrust::device_vector<uint32_t> d_CNodeMortonArray(gridCNodeSize, 0);
	thrust::device_vector<bool> d_isValidCNodeArray;
	thrust::device_vector<size_t> d_esumCNodesArray; // exclusive scan
	thrust::device_vector<uint32_t> d_pactCNodeArray;
	thrust::device_vector<size_t> d_numTreeNodesArray; // 节点数量记录数组
	thrust::device_vector<size_t> d_sumTreeNodesArray; // inlusive scan
	thrust::device_vector<size_t> d_esumTreeNodesArray; // 存储每一层节点数量的exclusive scan数组
	thrust::device_vector<uint32_t> d_begMortonArray;
	thrust::device_vector<SVONode> d_nodeArray; // 存储某一层的节点数组
	thrust::device_vector<SVONode> d_SVONodeArray; // save all sparse octree nodes

	// mesh voxelize
	resizeThrust(d_CNodeMortonArray, gridCNodeSize, (uint32_t)0);
	meshVoxelize(nModelTris, modelTris, d_surfaceVoxelGridSize, d_unitVoxelSize, d_gridOrigin, d_CNodeMortonArray);

	// create octree
	while (true)
	{
		// compute the number of 'coarse nodes'(eg: voxels)
		resizeThrust(d_isValidCNodeArray, gridCNodeSize);
		resizeThrust(d_esumCNodesArray, gridCNodeSize);
		thrust::transform(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end(), d_isValidCNodeArray.begin(), scanMortonFlag<uint32_t>());
		thrust::exclusive_scan(d_isValidCNodeArray.begin(), d_isValidCNodeArray.end(), d_esumCNodesArray.begin(), 0); // 必须加init
		size_t numCNodes = *(d_esumCNodesArray.rbegin()) + *(d_isValidCNodeArray.rbegin());
		if (!numCNodes) { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }

		treeDepth++;

		// compact coarse node array
		d_pactCNodeArray.clear(); resizeThrust(d_pactCNodeArray, numCNodes);
		getOccupancyMaxPotentialBlockSize(gridCNodeSize, minGridSize, blockSize, gridSize, compactArray, 0, 0);
		compactArray << <gridSize, blockSize >> > (gridCNodeSize, d_isValidCNodeArray.data().get(),
			d_CNodeMortonArray.data().get(), d_esumCNodesArray.data().get(), d_pactCNodeArray.data().get());
		getLastCudaError("Kernel 'compactArray' launch failed!\n");
		/*vector<uint32_t> h_pactCNodeArray(numCNodes, 0);
		CUDA_CHECK(cudaMemcpy(h_pactCNodeArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));*/

		if (treeDepth == 1)
		{
			numVoxels = numCNodes;
#ifndef NDEBUG
			// 验证体素
			vector<uint32_t> voxelArray;
			voxelArray.resize(numCNodes);
			CUDA_CHECK(cudaMemcpy(voxelArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));
			saveVoxel(modelBBox, voxelArray, base_filename, unitNodeWidth);
#endif // !NDEBUG
		}

		// compute the number of (real)octree nodes by coarse node array and set parent's morton code to 'd_CNodeMortonArray'
		size_t numNodes = 1;
		if (numCNodes > 1)
		{
			resizeThrust(d_numTreeNodesArray, numCNodes, (size_t)0);
			d_CNodeMortonArray.clear(); resizeThrust(d_CNodeMortonArray, gridTreeNodeSize, (uint32_t)0); // 此时用于记录父节点层的coarse node
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
			const uint32_t firstMortonCode = getParentMorton(d_pactCNodeArray[0]);
			d_CNodeMortonArray[firstMortonCode] = firstMortonCode | E_MORTON_32_FLAG;
			cpNumNodes << <gridSize, blockSize >> > (numCNodes, d_pactCNodeArray.data().get(), d_numTreeNodesArray.data().get(), d_CNodeMortonArray.data().get());
			getLastCudaError("Kernel 'cpNumNodes' launch failed!\n");
			resizeThrust(d_sumTreeNodesArray, numCNodes, (size_t)0); // inlusive scan
			thrust::inclusive_scan(d_numTreeNodesArray.begin(), d_numTreeNodesArray.end(), d_sumTreeNodesArray.begin());

			numNodes = *(d_sumTreeNodesArray.rbegin()) + 8;
		}
		depthNumNodes.emplace_back(numNodes);

		// set octree node array
		d_nodeArray.clear(); resizeThrust(d_nodeArray, numNodes, SVONode());
		d_begMortonArray.clear(); resizeThrust(d_begMortonArray, numCNodes);
		if (treeDepth == 1)
		{
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_1);
			createNode_1 << <gridSize, blockSize >> > (numCNodes, d_sumTreeNodesArray.data().get(),
				d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(), d_nodeArray.data().get());
			getLastCudaError("Kernel 'createNode_1' launch failed!\n");
			cudaDeviceSynchronize();

			d_esumTreeNodesArray.push_back(0);

			numFineNodes = numNodes;
		}
		else
		{
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_2);
			createNode_2 << <gridSize, blockSize >> > (numCNodes, *(d_esumTreeNodesArray.rbegin() + 1), *(d_esumTreeNodesArray.rbegin()),
				d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(),
				d_nodeArray.data().get(), (d_SVONodeArray.data() + (*(d_esumTreeNodesArray.rbegin() + 1))).get());
			getLastCudaError("Kernel 'createNode_2' launch failed!\n");
		}
		auto newEndOfBegMorton = thrust::unique(d_begMortonArray.begin(), d_begMortonArray.end());
		resizeThrust(d_begMortonArray, newEndOfBegMorton - d_begMortonArray.begin());

		blockSize = 256; gridSize = (numNodes + blockSize - 1) / blockSize;
		createRemainNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize / 8 >> > (numNodes, d_gridOrigin, d_unitNodeWidth,
			d_begMortonArray.data().get(), d_nodeArray.data().get());
		getLastCudaError("Kernel 'createRemainNode' launch failed!\n");

		d_SVONodeArray.insert(d_SVONodeArray.end(), d_nodeArray.begin(), d_nodeArray.end());

		d_esumTreeNodesArray.push_back(numNodes + (*d_esumTreeNodesArray.rbegin()));

		uint32_t numParentCNodes = *thrust::max_element(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end());
		bool isValidMorton = (numParentCNodes >> 31) & 1;
		// '+ isValidMorton' to prevent '(numParentNodes & D_MORTON_32_FLAG) = 0'同时正好可以让最后的大小能存储到最大的莫顿码
		numParentCNodes = (numParentCNodes & D_MORTON_32_FLAG) + isValidMorton;
		if (numParentCNodes != 0)
		{
			resizeThrust(d_CNodeMortonArray, numParentCNodes);
			unitNodeWidth *= 2.0; CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(double), cudaMemcpyHostToDevice));
			gridCNodeSize = numParentCNodes; gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
			if (numNodes / 8 == 0) { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }
		}
		else { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	}

	// copy to host
	numTreeNodes = d_esumTreeNodesArray[treeDepth];
	svoNodeArray.resize(numTreeNodes);
	auto freeResOfCreateTree = [&]()
	{
		cleanupThrust(d_CNodeMortonArray);
		cleanupThrust(d_isValidCNodeArray);
		cleanupThrust(d_esumCNodesArray);
		cleanupThrust(d_pactCNodeArray);
		cleanupThrust(d_numTreeNodesArray);
		cleanupThrust(d_sumTreeNodesArray);
		cleanupThrust(d_nodeArray);

		CUDA_CHECK(cudaFree(d_surfaceVoxelGridSize));
		CUDA_CHECK(cudaFree(d_gridOrigin));
		CUDA_CHECK(cudaFree(d_unitNodeWidth));
		CUDA_CHECK(cudaFree(d_unitVoxelSize));
	};
	freeResOfCreateTree();

	constructNodeAtrributes(d_esumTreeNodesArray, d_SVONodeArray);
	CUDA_CHECK(cudaMemcpy(svoNodeArray.data(), d_SVONodeArray.data().get(), sizeof(SVONode) * numTreeNodes, cudaMemcpyDeviceToHost));
	cleanupThrust(d_numTreeNodesArray);
	cleanupThrust(d_SVONodeArray);
}

namespace {
	__device__ size_t d_topNodeIdx;
}
template<bool topFlag>
__global__ void findNeighbors(const size_t nNodes,
	const size_t preESumTreeNodes,
	SVONode* d_nodeArray)
{
	if (topFlag)
	{
		d_nodeArray[0].neighbors[13] = d_topNodeIdx;
	}
	else
	{
		size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
		size_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;

		if (tid_x < nNodes && tid_y < 27)
		{
			SVONode& t = d_nodeArray[preESumTreeNodes + tid_x];
			const SVONode& p = d_nodeArray[t.parent];
			const uint8_t key = (t.mortonCode) & LOWER_3BIT_MASK;
			const unsigned int p_neighborIdx = p.neighbors[morton::neighbor_LUTparent[key][tid_y]];
			if (p_neighborIdx != UINT32_MAX)
			{
				const SVONode& h = d_nodeArray[p_neighborIdx];
				t.neighbors[tid_y] = h.childs[morton::neighbor_LUTchild[key][tid_y]];
			}
			else t.neighbors[tid_y] = UINT32_MAX;
		}
	}

}

void SparseVoxelOctree::constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	dim3 gridSize, blockSize;
	blockSize.x = 32, blockSize.y = 32;
	gridSize.y = 1;
	// find neighbors(up to bottom)
	if (treeDepth >= 1)
	{
		const size_t idx = d_SVONodeArray.size() - 1;
		CUDA_CHECK(cudaMemcpyToSymbol(d_topNodeIdx, &idx, sizeof(size_t)));
		findNeighbors<true> << <1, 1 >> > (1, 0, (d_SVONodeArray.data() + idx).get());
		for (int i = treeDepth - 2; i >= 0; --i)
		{
			const size_t nNodes = depthNumNodes[i];
			gridSize.x = (nNodes + blockSize.x - 1) / blockSize.x;
			findNeighbors<false> << <gridSize, blockSize >> > (nNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get());
		}
	}
}

__global__ void determineNodeVertex(const size_t nNodes,
	const size_t nodeOffset,
	const SVONode* d_nodeArray,
	node_vertex_type* d_nodeVertArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		size_t nodeIdx = nodeOffset + tid_x;
		uint16_t x, y, z;
		const Eigen::Vector3d& origin = d_nodeArray[nodeIdx].origin;
		const double& width = d_nodeArray[nodeIdx].width;

		/*Eigen::Vector3d verts[8] =
		{
			origin,
			Eigen::Vector3d(origin.x() + width, origin.y(), origin.z()),
			Eigen::Vector3d(origin.x(), origin.y() + width, origin.z()),
			Eigen::Vector3d(origin.x() + width, origin.y() + width, origin.z()),

			Eigen::Vector3d(origin.x(), origin.y(), origin.z() + width),
			Eigen::Vector3d(origin.x() + width, origin.y(), origin.z() + width),
			Eigen::Vector3d(origin.x(), origin.y() + width, origin.z() + width),
			Eigen::Vector3d(origin.x() + width, origin.y() + width, origin.z() + width),
		};

		for (int i = 0; i < 8; ++i)
		{
			size_t idx = tid_x;
			for (int j = 0; j < 8; ++j)
			{
				if (d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]] < idx) idx = d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]];
			}
			d_nodeVertArray[tid_x * 8 + i] = thrust::make_pair<Eigen::Vector3d, uint32_t>(verts[i], idx);
		}*/

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			morton::morton3D_32_decode(i, x, y, z);
			Eigen::Vector3d corner = origin + width * Eigen::Vector3d(x, y, z);
			size_t idx = nodeIdx;
#pragma unroll
			for (int j = 0; j < 8; ++j)
				if (d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]] < idx) idx = d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]];

			d_nodeVertArray[tid_x * 8 + i] = thrust::make_pair(corner, idx);
		}
	}
}

__global__ void determineNodeEdge(const size_t nNodes,
	const SVONode* d_nodeArray,
	node_edge_type* d_nodeEdgeArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		Eigen::Vector3d origin = d_nodeArray[tid_x].origin;
		double width = d_nodeArray[tid_x].width;

		// 0-2 2-3 1-3 0-1; 4-6 6-7 5-7 4-5; 0-4 2-6 3-7 1-5;
		thrust_edge_type edges[12] =
		{
			thrust::make_pair(origin, origin + Eigen::Vector3d(0, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, 0), origin + Eigen::Vector3d(width, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, 0), origin + Eigen::Vector3d(width, width, 0)),
			thrust::make_pair(origin,origin + Eigen::Vector3d(width, 0, 0)),

			thrust::make_pair(origin + Eigen::Vector3d(0, 0, width), origin + Eigen::Vector3d(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, width), origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, width),origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, 0, width), origin + Eigen::Vector3d(width, 0, width)),

			thrust::make_pair(origin, origin + Eigen::Vector3d(0, 0, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, 0), origin + Eigen::Vector3d(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, width, 0), origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, 0), origin + Eigen::Vector3d(width, 0, width)),
		};

#pragma unroll
		for (int i = 0; i < 12; ++i)
		{
			thrust_edge_type edge = edges[i];
			size_t idx = tid_x;

#pragma unroll
			for (int j = 0; j < 4; ++j)
				if (d_nodeArray[tid_x].neighbors[morton::d_edgeSharedLUT[i * 4 + j]] < idx) idx = d_nodeArray[tid_x].neighbors[morton::d_edgeSharedLUT[i * 4 + j]];

			d_nodeEdgeArray[tid_x * 12 + i] = thrust::make_pair(edge, idx);
		}
	}
}

namespace {
	template <typename T>
	struct lessPoint {
		__host__ __device__ int operator()(const T& a, const T& b) const {
			for (size_t i = 0; i < a.size(); ++i) {
				if (fabs(a[i] - b[i]) < 1e-9) continue;

				if (a[i] < b[i]) return 1;
				else if (a[i] > b[i]) return -1;
			}
			return 0;
		}
	};

	struct sortVert {
		__host__ __device__ bool operator()(const node_vertex_type& a, const node_vertex_type& b) {
			int _t = lessPoint<V3d>{}(a.first, b.first);
			if (_t == 0) return a.second < b.second;
			else if (_t == 1) return true;
			else return false;
		}
	};

	struct sortEdge {
		__host__ __device__ bool operator()(node_edge_type& a, node_edge_type& b) {
			int _t_0 = lessPoint<V3d>{}(a.first.first, b.first.first);
			if (_t_0 == 0)
			{
				int _t_1 = lessPoint<V3d>{}(a.first.second, b.first.second);
				if (_t_1 == 0) return a.second < b.second;
				else if (_t_1 == 1) return true;
				else return false;
			}
			else if (_t_0 == 1) return true;
			else return false;
		}
	};

	struct uniqueVert {
		__host__ __device__ bool operator()(const node_vertex_type& a, const node_vertex_type& b) {
			return (a.first).isApprox(b.first, 1e-9);
		}
	};

	struct uniqueEdge {
		__host__ __device__
			bool operator()(const node_edge_type& a, const node_edge_type& b) {
			return ((a.first.first.isApprox(b.first.first)) && (a.first.second.isApprox(b.first.second))) ||
				((a.first.first.isApprox(b.first.second)) && (a.first.second.isApprox(b.first.first)));
		}
	};
}

#define MAX_STREAM 16
void SparseVoxelOctree::constructNodeVertexAndEdge(const thrust::device_vector<size_t>& d_esumTreeNodesArray, thrust::device_vector<SVONode>& d_SVONodeArray)
{
	assert(treeDepth + 1 <= MAX_STREAM, "the number of stream is too small!\n");
	cudaStream_t streams[MAX_STREAM];
	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

	depthNodeVertexArray.resize(treeDepth);
	esumDepthNodeVerts.resize(treeDepth + 1, 0);
	for (int i = 0; i < treeDepth; ++i)
	{
		const size_t numNodes = depthNumNodes[i];
		thrust::device_vector<node_vertex_type> d_nodeVertArray(numNodes * 8);

		getOccupancyMaxPotentialBlockSize(numNodes, minGridSize, blockSize, gridSize, determineNodeVertex, 0, 0);
		determineNodeVertex << <gridSize, blockSize, 0, streams[i] >> > (numNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get(), d_nodeVertArray.data().get());
		getLastCudaError("Kernel 'determineNodeVertex' launch failed!\n");

		thrust::sort(thrust::cuda::par.on(streams[i]), d_nodeVertArray.begin(), d_nodeVertArray.end(), sortVert());
		auto vertNewEnd = thrust::unique(thrust::cuda::par.on(streams[i]), d_nodeVertArray.begin(), d_nodeVertArray.end(), uniqueVert());
		cudaStreamSynchronize(streams[i]);

		size_t cur_numNodeVerts = vertNewEnd - d_nodeVertArray.begin();
		resizeThrust(d_nodeVertArray, cur_numNodeVerts);
		numNodeVerts += cur_numNodeVerts;

		std::vector<node_vertex_type> h_nodeVertArray;
		h_nodeVertArray.resize(cur_numNodeVerts);
		CUDA_CHECK(cudaMemcpy(h_nodeVertArray.data(), d_nodeVertArray.data().get(), sizeof(node_vertex_type) * cur_numNodeVerts, cudaMemcpyDeviceToHost));
		depthNodeVertexArray[i] = h_nodeVertArray;
		esumDepthNodeVerts[i + 1] = esumDepthNodeVerts[i] + cur_numNodeVerts;

		nodeVertexArray.insert(nodeVertexArray.end(), h_nodeVertArray.begin(), h_nodeVertArray.end());
	}

	thrust::device_vector <node_edge_type> d_fineNodeEdgeArray(numFineNodes * 12);
	getOccupancyMaxPotentialBlockSize(numFineNodes, minGridSize, blockSize, gridSize, determineNodeEdge, 0, 0);
	determineNodeEdge << <gridSize, blockSize, 0, streams[treeDepth] >> > (numFineNodes, d_SVONodeArray.data().get(), d_fineNodeEdgeArray.data().get());
	getLastCudaError("Kernel 'determineNodeEdge' launch failed!\n");

	thrust::sort(thrust::cuda::par.on(streams[treeDepth]), d_fineNodeEdgeArray.begin(), d_fineNodeEdgeArray.end(), sortEdge());
	auto edgeNewEnd = thrust::unique(thrust::cuda::par.on(streams[treeDepth]), d_fineNodeEdgeArray.begin(), d_fineNodeEdgeArray.end(), uniqueEdge());
	cudaStreamSynchronize(streams[treeDepth]);

	numFineNodeEdges = edgeNewEnd - d_fineNodeEdgeArray.begin();
	resizeThrust(d_fineNodeEdgeArray, numFineNodeEdges);
	fineNodeEdgeArray.resize(numFineNodeEdges);
	CUDA_CHECK(cudaMemcpy(fineNodeEdgeArray.data(), d_fineNodeEdgeArray.data().get(), sizeof(node_edge_type) * numFineNodeEdges, cudaMemcpyDeviceToHost));

	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

void SparseVoxelOctree::constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	constructNodeNeighbors(d_esumTreeNodesArray, d_SVONodeArray);

	constructNodeVertexAndEdge(d_esumTreeNodesArray, d_SVONodeArray);
}

std::tuple<vector<std::pair<V3d, double>>, vector<size_t>> SparseVoxelOctree::setInDomainPoints(const uint32_t& nodeIdx, const int& nodeDepth,
	const vector<size_t>& esumDepthNodeVertexSize, vector<std::map<V3d, size_t>>& nodeVertex2Idx)
{
	int parentDepth = nodeDepth;
	auto parentIdx = svoNodeArray[nodeIdx].parent;
	vector<std::pair<V3d, double>> dm_points;
	vector<size_t> dm_pointsIdx;

	auto getCorners = [&](const SVONode& node, const int& depth)
	{
		const V3d& nodeOrigin = node.origin;
		const double& nodeWidth = node.width;
		const size_t& esumNodeVerts = esumDepthNodeVertexSize[depth];

		for (int k = 0; k < 8; ++k)
		{
			const int xOffset = k & 1;
			const int yOffset = (k >> 1) & 1;
			const int zOffset = (k >> 2) & 1;

			V3d corner = nodeOrigin + nodeWidth * V3d(xOffset, yOffset, zOffset);

			dm_points.emplace_back(std::make_pair(corner, nodeWidth));
			dm_pointsIdx.emplace_back(esumNodeVerts + nodeVertex2Idx[depth][corner]);
		}
	};

	while (parentIdx != UINT_MAX)
	{
		const auto& svoNode = svoNodeArray[parentIdx];
		getCorners(svoNode, parentDepth);
		parentIdx = svoNode.parent;
		parentDepth++;
	}

	return std::make_tuple(dm_points, dm_pointsIdx);
}

std::vector<std::tuple<V3d, double, size_t>> SparseVoxelOctree::mq_setInDomainPoints(const uint32_t& nodeIdx,
	const vector<size_t>& esumDepthNodeVertexArray, vector<std::map<V3d, size_t>>& nodeVertex2Idx)
{
	int curDepth = 0;
	uint32_t curNodeIdx = nodeIdx;
	vector<std::tuple<V3d, double, size_t>> dm_points; // 坐标、宽度和在所有顶点数组中的下标

	auto getCorners = [&](const SVONode& node, const int& depth)
	{
		const V3d& nodeOrigin = node.origin;
		const double& nodeWidth = node.width;
		const size_t& esumNodeVerts = esumDepthNodeVertexArray[depth];

		for (int k = 0; k < 8; ++k)
		{
			const int xOffset = k & 1;
			const int yOffset = (k >> 1) & 1;
			const int zOffset = (k >> 2) & 1;

			V3d corner = nodeOrigin + nodeWidth * V3d(xOffset, yOffset, zOffset);

			dm_points.emplace_back(std::make_tuple(corner, nodeWidth, esumNodeVerts + nodeVertex2Idx[depth].at(corner)));
		}
	};

	while (curNodeIdx != UINT_MAX)
	{
		const auto& svoNode = svoNodeArray[curNodeIdx];
		getCorners(svoNode, curDepth);
		curNodeIdx = svoNode.parent;
		curDepth++;
	}

	return dm_points;
}

//////////////////////
//  I/O: Save Data  //
//////////////////////
void SparseVoxelOctree::saveSVO(const std::string& filename) const
{
	checkDir(filename);
	std::ofstream output(filename.c_str(), std::ios::out);
	if (!output) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename.c_str()); return; }
	//assert(output);

#ifndef SILENT
	std::cout << "[I/O] Writing Sparse Voxel Octree data in obj format to file " << std::quoted(filename.c_str()) << std::endl;
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = numTreeNodes / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (const auto& node : svoNodeArray)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif
		gvis::writeCube(node.origin, Eigen::Vector3d(node.width, node.width, node.width), output, faceBegIdx);
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}

void SparseVoxelOctree::saveVoxel(const AABox<Eigen::Vector3d>& modelBBox, const vector<uint32_t>& voxelArray,
	const std::string& base_filename, const double& width) const
{
	std::string filename_output = base_filename + std::string("_voxel.obj");
	checkDir(filename_output);
	std::ofstream output(filename_output.c_str(), std::ios::out);
	if (!output) { fprintf(stderr, "[I/O] Error: File %s could not be opened!", filename_output.c_str()); return; }
	//assert(output);

#ifndef SILENT
	std::cout << "[I/O] Writing data in obj voxels format to file " << std::quoted(filename_output.c_str()) << std::endl;
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = voxelArray.size() / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (size_t i = 0; i < voxelArray.size(); ++i)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif

		const auto& morton = voxelArray[i];

		uint16_t x, y, z;
		morton::morton3D_32_decode((morton & D_MORTON_32_FLAG), x, y, z);
		const Eigen::Vector3d nodeOrigin = modelBBox.boxOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		gvis::writeCube(nodeOrigin, Eigen::Vector3d(width, width, width), output, faceBegIdx);
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}
