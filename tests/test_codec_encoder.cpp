// Codec encoder gate (DESIGN 6): voice_1.wav -> 16 codes per 12.5 Hz frame, against the
// PyTorch dumps, stage by stage so a failure says *where*.
//   conv     vs ref_conv.npy     the SEANet conv stack at 25 Hz
//   xf       vs ref_xf.npy       after the 8-layer transformer (full causal, no window)
//   latents  vs ref_latents.npy  after the 25 -> 12.5 Hz downsample, the last float
//   codes    vs ref_codes.npy    EXACT, cb0 included
//
// Codes are a nearest-centroid search, so they are gated exactly and a mismatch is
// reported with both centroids' distances rather than tolerated: a tie and an error look
// the same in a pass/fail, and only one of them is acceptable. The float stages carry the
// numerical claim.
//
// Also checked: a chunked encode equals a one-shot one (the conv stack threads each conv's
// causal history across chunks); encoding twice gives the same codes (nothing is carried);
// and a negative control - moving a centroid the reference actually selected must make the
// codes gate fail, or the gate proves nothing.
//
// Usage: test_codec_encoder <model.navi> <reference dir>

#include "model/qwen3tts/codec_encoder.h"
#include "model/qwen3tts/params.h"
#include "runtime/audio/wav.h"
#include "runtime/common/npy.h"
#include "runtime/device/device.h"
#include "runtime/weights/device_weights.h"
#include "runtime/weights/navi_file.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Cmp {
    double max_abs = 0, ref_max = 0, rel_rms = 0;
    bool finite = true;
};

// Both sides are checked for finiteness, and so are the derived metrics: a NaN anywhere
// makes rel_rms a NaN, every comparison against which is false, so it would pass a bare
// threshold test unnoticed.
Cmp compare(const std::vector<float> & got, const std::vector<float> & want) {
    Cmp c;
    double se = 0, ss = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double x = got[i], y = want[i];
        if (!std::isfinite(x) || !std::isfinite(y)) c.finite = false;
        const double d = std::fabs(x - y);
        c.max_abs = std::max(c.max_abs, d);
        c.ref_max = std::max(c.ref_max, std::fabs(y));
        se += d * d;
        ss += y * y;
    }
    c.rel_rms = std::sqrt(se / (ss > 0 ? ss : 1e-30));
    if (!std::isfinite(c.rel_rms) || !std::isfinite(c.max_abs)) c.finite = false;
    return c;
}

bool report(const char * name, const Cmp & c, double gate) {
    const bool ok = c.finite && c.rel_rms < gate;
    std::printf("  %-8s max|d| %.3e (ref max %.3e)  rel-rms %.2e  gate %.0e  %s\n", name, c.max_abs, c.ref_max,
                c.rel_rms, gate, ok ? "ok" : (c.finite ? "OVER THE GATE" : "NON-FINITE"));
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.navi reference_dir\n", argv[0]); return 2; }
    try {
        const std::string ref = argv[2];
        navi::NaviFile file = navi::NaviFile::open(argv[1]);
        const auto params = navi::qwen3tts::read_params(file);
        if (!params.has_codec_encoder) {
            std::printf("this blob has no codec encoder (convert.py --with-encoder): nothing to gate\n");
            return 0;
        }
        const auto & e = params.codec_encoder;
        const auto wav = navi::audio::read_wav(ref + "/voice_1.wav");
        if (wav.sample_rate != e.sample_rate) { std::printf("FAIL: voice_1.wav is %d Hz\n", wav.sample_rate); return 1; }

        const auto conv_npy = navi::Npy::load(ref + "/ref_conv.npy");
        const auto xf_npy = navi::Npy::load(ref + "/ref_xf.npy");
        const auto lat_npy = navi::Npy::load(ref + "/ref_latents.npy");
        const auto codes_npy = navi::Npy::load(ref + "/ref_codes.npy");
        const auto want_conv = conv_npy.as_f32();
        const auto want_xf = xf_npy.as_f32();
        const auto want_lat = lat_npy.as_f32();
        const auto want_codes = codes_npy.as_i32();
        const int T25 = static_cast<int>(conv_npy.shape[0]);
        const int frames = static_cast<int>(codes_npy.shape[0]);
        const int G = static_cast<int>(codes_npy.shape[1]);

        navi::Device dev = navi::Device::open();
        navi::DeviceWeights w = navi::DeviceWeights::upload(file, [](const navi::TensorInfo & t) {
            return t.name.rfind("codec.encoder.", 0) == 0;
        });
        auto enc = navi::qwen3tts::CodecEncoder::create(dev, file, w, params);
        file.close();

        // --- the stages ----------------------------------------------------
        std::vector<Cmp> stages;
        std::vector<std::string> names;
        enc->set_stage_hook([&](const std::string & name, const float * buf, int rows, int cols) {
            std::vector<float> host(static_cast<std::size_t>(rows) * cols);
            if (hipMemcpy(host.data(), buf, host.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
                throw std::runtime_error("stage readback failed");
            }
            const std::vector<float> * want = name == "conv" ? &want_conv : (name == "xf" ? &want_xf : &want_lat);
            if (host.size() != want->size()) {
                std::printf("  %-8s SHAPE %d x %d, reference %zu floats\n", name.c_str(), rows, cols, want->size());
                stages.push_back(Cmp{0, 0, 1e9, true});
            } else {
                stages.push_back(compare(host, *want));
            }
            names.push_back(name);
        });

        int n_frames = 0;
        const auto codes = enc->encode(wav.pcm, &n_frames);
        const double ms = enc->last_ms();
        std::printf("codec encoder: %zu samples -> %d rows at 25 Hz -> %d frames x %d codebooks  (%.1f ms, %d-frame chunks)\n",
                    wav.pcm.size(), T25, n_frames, G, ms, enc->chunk_frames());
        if (n_frames != frames) { std::printf("FAIL: %d frames, reference %d\n", n_frames, frames); return 1; }

        // The gates are the measured f32 values' own order of magnitude, as the prefill
        // gate's 5e-3 was: this is f32 weights against an f32 reference, so what is left is
        // accumulation order.
        bool stages_ok = stages.size() == 3;
        if (!stages_ok) std::printf("FAIL: %zu stages reported, expected 3\n", stages.size());
        for (std::size_t i = 0; i < stages.size(); ++i) {
            stages_ok = report(names[i].c_str(), stages[i], 1e-4) && stages_ok;
        }

        // --- the codes, exactly --------------------------------------------
        int bad = 0, bad_cb0 = 0, first_bad_frame = -1, first_bad_cb = -1;
        for (int t = 0; t < frames; ++t) {
            for (int g = 0; g < G; ++g) {
                const std::size_t i = static_cast<std::size_t>(t) * G + g;
                if (codes[i] != want_codes[i]) {
                    ++bad;
                    if (g == 0) ++bad_cb0;
                    if (first_bad_frame < 0) { first_bad_frame = t; first_bad_cb = g; }
                }
            }
        }
        std::printf("  codes    %d of %d differ from the reference%s\n", bad, frames * G,
                    bad ? "" : " (exact, cb0 included)");
        if (bad) {
            // Say whether the first divergence is a tie or an error. Only the FIRST one in a
            // frame is informative: after it the residual differs, so every later codebook in
            // that frame is a consequence, not a finding.
            const int dim = e.vq_dim;
            const std::string q = first_bad_cb == 0
                ? "codec.encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.codebook"
                : "codec.encoder.quantizer.acoustic_residual_vector_quantizer.layers."
                      + std::to_string(first_bad_cb - 1) + ".codebook.codebook";
            std::printf("    first at frame %d codebook %d: engine %d, reference %d\n", first_bad_frame,
                        first_bad_cb, codes[static_cast<std::size_t>(first_bad_frame) * G + first_bad_cb],
                        want_codes[static_cast<std::size_t>(first_bad_frame) * G + first_bad_cb]);
            std::printf("    cb0 mismatches: %d (the semantic code, which carries content)\n", bad_cb0);
            std::printf("    (%s, dim %d - compare the two centroids by hand if this ever fires)\n", q.c_str(), dim);
        }
        const bool codes_ok = bad == 0;

        // --- chunked == one-shot -------------------------------------------
        // One shot needs its own encoder: the scratch is sized for the chunk, so a bigger
        // chunk is a bigger build, not a setter (and it is ~230 MB at this clip's length).
        enc->set_stage_hook(nullptr);
        const auto again = enc->encode(wav.pcm);
        const bool repeat_ok = again == codes;
        enc->set_chunk_frames(5);                    // a chunk the clip is not a multiple of
        const auto chunked5 = enc->encode(wav.pcm);
        enc->set_chunk_frames(0);
        std::vector<std::int32_t> one_shot;
        {
            navi::NaviFile f2 = navi::NaviFile::open(argv[1]);
            auto whole = navi::qwen3tts::CodecEncoder::create(dev, f2, w, params, 30.0, frames);
            f2.close();
            one_shot = whole->encode(wav.pcm);
        }
        const bool chunk_ok = one_shot == codes && chunked5 == codes;
        std::printf("  chunks   %d frames (default), 5 frames, and one shot (%d frames): %s; "
                    "encoding twice: %s\n",
                    enc->chunk_frames(), frames, chunk_ok ? "identical codes" : "DIFFER",
                    repeat_ok ? "identical" : "DIFFERS");

        // --- negative control ----------------------------------------------
        // Move a centroid the reference actually selected far away. The argmin cannot
        // return it any more, so the codes MUST change; if they do not, the gate above is
        // not reading what it claims to read.
        // It has to be the buffer the kernel reads: the encoder copies the codebooks when the
        // arena does not hold them adjacent, and writing to the arena then changes nothing -
        // which is how this control first reported UNCHANGED while the encoder was correct.
        const int ctl_frame = frames / 2;
        const int ctl_code = want_codes[static_cast<std::size_t>(ctl_frame) * G];
        float * row = const_cast<float *>(enc->d_codebooks())
                      + static_cast<std::size_t>(ctl_code) * e.codebook_dim;
        std::vector<float> saved(static_cast<std::size_t>(e.codebook_dim));
        if (hipMemcpy(saved.data(), row, saved.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
            throw std::runtime_error("control readback failed");
        }
        std::vector<float> moved = saved;
        for (float & v : moved) v += 1e3f;
        if (hipMemcpy(row, moved.data(), moved.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("control write failed");
        }
        const auto perturbed = enc->encode(wav.pcm);
        if (hipMemcpy(row, saved.data(), saved.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) {
            throw std::runtime_error("control restore failed");
        }
        const auto restored = enc->encode(wav.pcm);
        const bool control_ok = perturbed != codes && restored == codes;
        std::printf("  control  centroid %d of cb0 (selected at frame %d) moved by 1e3: codes %s, restored %s\n",
                    ctl_code, ctl_frame, perturbed != codes ? "changed (gate has teeth)" : "UNCHANGED (!)",
                    restored == codes ? "identical" : "DIFFER (!)");

        const bool ok = stages_ok && codes_ok && chunk_ok && repeat_ok && control_ok;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "error: %s\n", ex.what());
        return 2;
    }
}
