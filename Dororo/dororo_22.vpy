import vskernels as vsk
import vodesfunc as vof
from vsaa import based_aa
from vsdeband import Placebo
from vsdehalo import fine_dehalo
from lvsfunc import custom_mask_clip
from vsmasktools import KirschTCanny
from vsrgtools import contrasharpening
from jvsfunc import retinex_edgemask, rfs
from vsscale import Waifu2x, descale_detail_mask
from vsdenoise import BM3DCudaRTC, Prefilter, Profile, nl_means
from vstools import core, vs, get_y, join, depth, finalize_clip

# Set threads, affinities, and cache size
core.set_affinity(range(16, 32), 18000)

# Degrain wrapper
def mvtools(clip: vs.VideoNode, thSAD: int = 150, prefilter: vs.VideoNode | Prefilter = Prefilter.NONE, **kwargs) -> vs.VideoNode:
    from vsdenoise import MVTools, SADMode, SearchMode, MotionMode, PelType
    d_args = dict(
        prefilter=prefilter, thSAD=thSAD, block_size=32, overlap=16,
        sad_mode=SADMode.ADAPTIVE_SPATIAL_MIXED, search=SearchMode.DIAMOND, 
        motion=MotionMode.HIGH_SAD, pel_type=PelType.WIENER,
    )
    d_args.update(**kwargs)

    return MVTools.denoise(clip, **d_args)

# Sneedfiltering stuff
op_start = 456
op_end = 2612
ed_start = 31911
ed_end = 34068

op = [(op_start, op_end)]

aa_ranges = [
    (op_end+1, op_end+96),
    (ed_end+1, ed_end+120),
]

masks = [
    'masks/dororo_op2_01.png', 
    'masks/dororo_op2_02.png', 
    'masks/dororo_op2_03.png', 
    'masks/dororo_op2_04.png', 
    'masks/dororo_op2_05.png', 
    'masks/dororo_op2_06.png',
    'masks/dororo_op2_07.png', 
    'masks/dororo_op2_08.png', 
    'masks/dororo_op2_09.png', 
    'masks/dororo_op2_10.png', 
    'masks/dororo_op2_11.png', 
    'masks/dororo_op2_12.png',
    'masks/dororo_op2_13.png',
]

mask_ranges = [
    (op_start+270, op_start+367), 
    (op_start+384, op_start+491), 
    (op_start+510, op_start+617), 
    (op_start+638, op_start+745), 
    (op_start+751, op_start+844), 
    (op_start+938, op_start+1023),
    (op_start+1191, op_start+1286), 
    (op_start+1304, op_start+1374), 
    (op_start+1502, op_start+1605), 
    (op_start+1629, op_start+1712), 
    (op_start+1721, op_start+1808), 
    (op_start+2007, op_start+2109),
    (op_start+2110, op_start+2156),
]

# Setup automation
setup = vof.Setup('22')
JPBD = vof.SRC_FILE(f'{setup.vol4_dir}/00004.m2ts')
USBD = vof.SRC_FILE(f'{setup.vol3_us_dir}/00053.m2ts', 0, -2156)

src = depth(JPBD.src_cut, 16)
luma = get_y(src)

# Rescale
descale = vsk.Catrom.descale(luma, 1432, 806)
rescale = Waifu2x.scale(descale, 1920, 1080, tiles=2)
upscale = vsk.Catrom.scale(descale, 1920, 1080)

aa = based_aa(rescale, supersampler=False)
aa = rfs(rescale, aa, op + aa_ranges)
dh = fine_dehalo(aa, brightstr=0.85, thmi=75, thma=95)

line_mask = KirschTCanny.edgemask(luma, 60<<8, 160<<8)
custom_mask = custom_mask_clip(upscale, luma, masks, mask_ranges)
credit_mask = descale_detail_mask(depth(luma, 32), depth(upscale, 32), 0.045)
credit_mask = depth(credit_mask, 16)
credit_mask = rfs(credit_mask, custom_mask, op + aa_ranges)

rescale = core.std.MaskedMerge(luma, dh, line_mask)
rescale = core.std.MaskedMerge(rescale, luma, credit_mask)
rescaled = join(rescale, src)

# Denoise
ref = mvtools(rescale, thSAD=65, prefilter=Prefilter.DFTTEST)
ref = ref.ttmpsm.TTempSmooth(maxr=1, thresh=1, mdiff=0, strength=1)
bm3d = BM3DCudaRTC.denoise(rescale, ref=ref, profile=Profile.LOW_COMPLEXITY, sigma=0.5, radius=3)
nlm = nl_means(rescaled, strength=0.28, tr=3, sr=2, planes=[1, 2])
denoise = join(bm3d, nlm)

# Deband
mask_deband = retinex_edgemask(denoise, 1, [50, 200, 250], 8000).rgvs.RemoveGrain(6)
deband = Placebo.deband(denoise, radius=16, thr=[2.2, 2.0], iterations=2, grain=0)
deband = core.std.MaskedMerge(deband, denoise, mask_deband)
csharp_deband = contrasharpening(deband, rescaled, mode=3)

# Regrain
grain = vof.grain(csharp_deband, [3.2, 0.6], 3.8, temporal_average=55, seed=7777777)
final = finalize_clip(grain)

if __name__ == '__main__':
    settings = vof.sb(preset='veryslow', crf=14, psy_rd=2.0, psy_rdoq=2.25, qcomp=0.72, aq_strength=0.85, deblock=-1, rect=False)
    en_audio = setup.encode_audio(USBD, track=0, codec='flac', dither_flac=True)
    audio = setup.encode_audio(JPBD, track=0, codec='flac', dither_flac=True)
    video = setup.encode_video(final, settings, src=JPBD, print_command=True)
    vof.Mux(setup,
        vof.VT(video, "Encode by Zander"),
        vof.AT(audio, "Japanese 2.0 FLAC"),
        vof.AT(en_audio, "English 2.0 FLAC", lang='en', default=False),
        vof.Chapters(JPBD).set_names(["Prologue", "Opening", "Part A", "Part B", "Ending", "Preview"])
    ).run()
else:
    vof.out(src, "JPBD")
    vof.out(final, "Filtered")