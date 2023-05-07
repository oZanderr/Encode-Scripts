import vodesfunc as vof
import vskernels as vsk
from awsmfunc import bbmod
from jvsfunc import retinex_edgemask
from vsdeband import Placebo
from vsmasktools import SobelStd
from vsdenoise import BM3DCudaRTC, Prefilter, Profile, nl_means
from vstools import core, vs, initialize_clip, finalize_clip

core.set_affinity(range(16, 32), 22000)

# Degrain wrapper
def mvtools(clip: vs.VideoNode, thSAD: int = 150, prefilter: vs.VideoNode | Prefilter = Prefilter.NONE, **kwargs) -> vs.VideoNode:
    from vsdenoise import MVTools, SADMode, SearchMode, MotionMode, PelType
    d_args = dict(
        prefilter=prefilter, thSAD=thSAD, block_size=32, overlap=16,
        sad_mode=(SADMode.ADAPTIVE_SPATIAL_MIXED, SADMode.ADAPTIVE_SATD_MIXED), search=SearchMode.DIAMOND, 
        motion=MotionMode.HIGH_SAD, pel_type=PelType.AUTO,
    )
    d_args.update(**kwargs)

    return MVTools.denoise(clip, **d_args)

# Automation setup
setup = vof.Setup('movie')
ITBD = vof.SRC_FILE(f'{setup.movie_dir}/00007.m2ts')
USBD = vof.SRC_FILE(f'{setup.movie_us_dir}/00012.m2ts', 12, -12)

src = initialize_clip(ITBD.src_cut, 16)

# Fix edges
edgefix = bbmod(src, 1, 1, 1, 1, 30)

# Rescale
lmask = SobelStd.edgemask(edgefix, multi=2, planes=(0, True)).std.Maximum()
rescale = vof.DescaleTarget(
    height=720, kernel=vsk.Bilinear,
    upscaler=vof.Clamped_Doubler(False, vof.Waifu2x_Doubler(tiles=2), 85),
    downscaler=vsk.Catrom,
    line_mask=lmask, credit_mask_thr=0.035
    ).generate_clips(edgefix)

rescaled = rescale.get_upscaled(edgefix)

# Denoise
ref = mvtools(rescaled, 70, Prefilter.MINBLUR3)
bm3d = BM3DCudaRTC.denoise(rescaled, ref=ref, sigma=0.6, tr=3, profile=Profile.NORMAL, planes=0)
den = nl_means(bm3d, ref=ref, strength=0.25, tr=2, planes=[1, 2])

# Deband
mask_deband = retinex_edgemask(den, brz=8500).rgvs.RemoveGrain(3)
deband = Placebo.deband(den, 40, [1.6, 1.8], 2, 0)
deband = Placebo.deband(deband, 15, [1.4, 1.6], 2, 0)
deband = core.std.MaskedMerge(deband, den, mask_deband)

# Regrain
grain = vof.grain(deband, [2.5, 0.5], 2.8, temporal_average=45, seed=5555555)
final = finalize_clip(grain)

if __name__ == '__main__':
    settings = vof.sb('veryslow', crf=14.5, psy_rd=2.0, psy_rdoq=2.15, qcomp=0.73, aq_strength=0.85, rect=False)
    audio = setup.encode_audio(ITBD, track=1, codec='opus', q=200)
    en_audio = setup.encode_audio(USBD, track=0, codec='opus', q=384)
    video = setup.encode_video(final, settings, src=ITBD, print_command=True)
    vof.Mux(setup, 
        vof.VT(video, 'Encode by Zander', lang='it'), 
        vof.AT(audio, 'Japanese 2.0 Opus'), 
        vof.AT(en_audio, 'English 5.1 Opus', lang='en', default=False),
        vof.Chapters(ITBD).set_names(["Recap Part A", "Recap Part B", "Part A", "Part B", "Part C", "Part D", "Part E", "Credits"])
    ).run()
else:
    vof.out(src, "ITBD")
    vof.out(final, "Filtered")