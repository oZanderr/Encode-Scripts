import vskernels as vsk
import vodesfunc as vof
from awsmfunc import bbmod
from lvsfunc import custom_mask_clip
from jvsfunc import retinex_edgemask, rfs
from vsaa import Nnedi3, based_aa
from vsdeband import Placebo
from vsmasktools import KirschTCanny
from vsscale import Waifu2x, descale_detail_mask
from vsdenoise import BM3DCudaRTC, Prefilter, Profile, nl_means
from vstools import core, vs, depth, join, get_y, iterate, initialize_clip, finalize_clip

# Set threads, affinities, and cache size
core.set_affinity(range(16, 32), 18000)

# Degrain wrapper
def mvtools(clip: vs.VideoNode, thSAD: int = 150, prefilter: vs.VideoNode | Prefilter = Prefilter.NONE) -> vs.VideoNode:
    from vsdenoise import MVTools, SADMode, SearchMode, MotionMode, PelType
    d_args = dict(
        prefilter=prefilter, thSAD=thSAD, block_size=32, overlap=16,
        sad_mode=SADMode.ADAPTIVE_SATD_MIXED, search=SearchMode.DIAMOND, 
        motion=MotionMode.HIGH_SAD, pel_type=PelType.BICUBIC,
    )

    return MVTools.denoise(clip, **d_args)

# Splice OP
def lehmer_merge_op(clip: vs.VideoNode, op_start: int, op_end: int, op_offset: int = 0, merged_op: str = r'./moonlit_merged_op.mkv') -> vs.VideoNode:
    merged_op = depth(vof.source(merged_op, force_lsmas=True), 16)

    if op_offset != 0:
        merged_op = merged_op[:-op_offset]

    return clip[:op_start] + merged_op + clip[op_end+1:]

# Sneedfiltering stuff
op_start = 1296
op_end = 3452
op = [(op_start, op_end)]
ed = [(31744, 33901)]

masks = [
    'masks/mask_12_3477_3595.png', 
    'masks/mask_logo_left.png', 
    'masks/mask_logo_right.png',
]

mask_ranges = [
    (3477, 3595),
    (16464, 16567),
    (16572, 16639), 
]

deint_ranges = [
    (op_start+836, op_start+893), 
    (op_start+955, op_start+1003), 
    (op_start+1540, op_start+1608), 
    (op_start+1636, op_start+1642), 
    (op_start+1651, op_start+1656), 
    (op_start+1660, op_start+1662),
]

zones = [(op_start, op_end, 1.2)]

# Setup automation
setup = vof.Setup('12')
JPBD = vof.SRC_FILE(f'{setup.vol4_dir}/00007.m2ts')

src = initialize_clip(JPBD.src_cut, 16)
merge = lehmer_merge_op(src, op_start, op_end, 2)

# Fix dirty lines
edgefix = depth(bbmod(merge, 1, 1, 1, 1, thresh=40, blur=600), 32)
luma = get_y(edgefix)

# Shift chroma
cshift = edgefix.std.SetFrameProp('_ChromaLocation', 1).resize.Bicubic(chromaloc=0)

# NCOP align
ncop = vof.source('ncop.mkv', force_lsmas=True)
blank = ncop.std.BlankClip(length=src.num_frames)
ncop = blank[:op_start] + ncop + blank[op_end+1:]
ncop_y = get_y(depth(ncop, 32))

# Remove YADIF artifacts
deint = Nnedi3(field=1, pscrn=1).interpolate(luma, False)
credit_mask = core.std.Expr([luma, ncop_y], 'x y - abs 0.18 < 0 1 ?')
credit_mask = iterate(credit_mask, core.std.Maximum, 4)
credit_mask = iterate(credit_mask, core.std.Inflate, 2)
masked = core.std.MaskedMerge(deint, luma, credit_mask)
luma = rfs(luma, masked, deint_ranges)

# Rescale
descale = vsk.Lanczos.descale(luma, 1664, 936)
upscale = Waifu2x(tiles=2).scale(descale, 1920, 1080)
reupscale = vsk.Lanczos.scale(descale, 1920, 1080)

# Custom Masking
custom_mask = custom_mask_clip(upscale, luma, masks, mask_ranges)
descale_mask = descale_detail_mask(luma, reupscale, 0.052)
descale_mask = rfs(descale_mask, custom_mask, mask_ranges)

masked = core.std.MaskedMerge(upscale, luma, descale_mask)
masked = rfs(masked, luma, op) # OP isn't rescaleable

rescaled = depth(join(masked, cshift), 16)

# Denoise
ref = mvtools(rescaled, thSAD=120)
bm3d = BM3DCudaRTC.denoise(rescaled, ref=ref, sigma=0.7, tr=3, profile=Profile.LOW_COMPLEXITY, planes=0)
denoise = nl_means(bm3d, ref=ref, strength=0.65, tr=3, wmode=3, planes=[1, 2])

# AA
mask_aa = KirschTCanny.edgemask(get_y(denoise), 60<<8, 180<<8).std.Convolution([1]*9)
aa = based_aa(denoise, 1.85, lmask=mask_aa, downscaler=vsk.Bicubic(-1/3, 1/6))
aa = core.std.MaskedMerge(aa, denoise, depth(descale_mask, 16))
aa = rfs(aa, denoise, op + ed)

# Deband
deband = Placebo.deband(aa, radius=15, thr=2.15, iterations=2, grains=0)
mask_deband = retinex_edgemask(aa, brz=9500).rgvs.RemoveGrain(3)
deband = core.std.MaskedMerge(deband, aa, mask_deband)

# Regrain
grain = vof.grain(deband, [2.8, 0.55], 3.6, static=True, luma_scaling=8, seed=7777777)
grain_op = vof.grain(deband, [2.5, 0.35], 2.8, static=True, luma_scaling=8, seed=7777777)

grain = rfs(grain, grain_op, op)
final = finalize_clip(grain)

if __name__ == '__main__':
    settings = vof.sb(preset='veryslow', crf=13.5, psy_rd=2.05, psy_rdoq=2.15, qcomp=0.73, aq_strength=0.85, rect=False)
    audio = setup.encode_audio(JPBD, track=0, codec='flac', dither_flac=False)
    video = setup.encode_video(final, settings, zones, generate_qpfile=True, src=JPBD)
    vof.Mux(setup,
        vof.VT(video, "Encode by Zander"),
        vof.AT(audio, "Japanese 2.0 FLAC"),
        vof.Chapters(JPBD).set_names(["Prologue", "Opening", "Part A", "Part B", "Ending", "Preview"]).shift_chapter(4, 2843)
    ).run()
else:
    vof.out(src, "JPBD")
    vof.out(final, "Filtered")