from vodesfunc import source, lehmer_merge
from vstools import core, depth, finalize_clip

core.set_affinity(range(0, 32, 2), 18000) # Use every other thread

# Create Lehmer Merged OP for splicing
path = [
    "E:\Encodes\Moonlit\src\[BDMV][211027][Tsuki ga Michibiku Isekai Douchuu][Vol.01]\BDMV\STREAM",
    "E:\Encodes\Moonlit\src\[BDMV][211124][Tsuki ga Michibiku Isekai Douchuu][Vol.02]\BDMV\STREAM",
    "E:\Encodes\Moonlit\src\[BDMV][211222][Tsuki ga Michibiku Isekai Douchuu][Vol.03]\BDMV\STREAM",
    "E:\Encodes\Moonlit\src\[BDMV][220126][Tsuki ga Michibiku Isekai Douchuu][Vol.04]\BDMV\STREAM",
]

op_01 = source(f'{path[0]}/00004.m2ts')[1104:3261]   #2157
op_02 = source(f'{path[0]}/00005.m2ts')[1582:3741]   #2159
op_03 = source(f'{path[0]}/00006.m2ts')[1008:3165]   #2157
op_04 = source(f'{path[1]}/00004.m2ts')[2446:4603]   #2157
op_05 = source(f'{path[1]}/00005.m2ts')[2062:4221]   #2159
op_06 = source(f'{path[1]}/00006.m2ts')[1822:3981]   #2159
op_07 = source(f'{path[2]}/00004.m2ts')[576:2734]    #2158
op_08 = source(f'{path[2]}/00005.m2ts')[2110:4269]   #2159
op_09 = source(f'{path[2]}/00006.m2ts')[696:2854]    #2158
op_10 = source(f'{path[3]}/00005.m2ts')[1582:3741]   #2159

op_sources = [op_01, op_02, op_03, op_04, op_05, op_06, op_07, op_08, op_09, op_10]
op_long = [op_02, op_05, op_06, op_08, op_10]

op_sources = [depth(clip, 16) for clip in op_sources]
op_long = [depth(clip, 16) for clip in op_long]

merge_sources = lehmer_merge(*op_sources)
merge_long = lehmer_merge(*op_long)
merged_op = merge_sources[:2157] + merge_long[2157:] # Trim to shortest length OP + merge of longest length OPs

# Output for lossless encode
finalize_clip(merged_op).set_output()