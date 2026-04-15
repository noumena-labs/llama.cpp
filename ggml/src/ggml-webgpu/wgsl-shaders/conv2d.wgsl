#include "common_decls.tmpl"
enable f16;

@group(0) @binding(0)
#if defined(WEIGHT_F32)
var<storage, read_write> weights: array<f32>;
#elif defined(WEIGHT_F16)
var<storage, read_write> weights: array<f16>;
#endif

@group(0) @binding(1)
#if defined(INPUT_F32)
var<storage, read_write> input: array<f32>;
#elif defined(INPUT_F16)
var<storage, read_write> input: array<f16>;
#endif

@group(0) @binding(2)
#if defined(OUTPUT_F32)
var<storage, read_write> output: array<f32>;
#elif defined(OUTPUT_F16)
var<storage, read_write> output: array<f16>;
#endif

struct Params {
    offset_w: u32,
    offset_i: u32,
    offset_o: u32,

    // Element strides
    sw0: u32, sw1: u32, sw2: u32, sw3: u32,
    si0: u32, si1: u32, si2: u32, si3: u32,
    so0: u32, so1: u32, so2: u32, so3: u32,

    KW: u32, KH: u32, IC: u32, OC: u32,
    IW: u32, IH: u32, IC_in: u32, N: u32,
    OW: u32, OH: u32, OC_out: u32, N_out: u32,

    s0: u32, s1: u32,
    p0: u32, p1: u32,
    d0: u32, d1: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

fn load_weight(idx: u32) -> f32 {
    #if defined(WEIGHT_F32)
        return weights[idx];
    #elif defined(WEIGHT_F16)
        return f32(weights[idx]);
    #endif
}

fn load_input(idx: u32) -> f32 {
    #if defined(INPUT_F32)
        return input[idx];
    #elif defined(INPUT_F16)
        return f32(input[idx]);
    #endif
}

fn store_output(idx: u32, val: f32) {
    #if defined(OUTPUT_F32)
        output[idx] = val;
    #elif defined(OUTPUT_F16)
        output[idx] = f16(val);
    #endif
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {

    let threads_per_group = u32(WG_SIZE);
    let i_out = gid.x + (num_wg.x * threads_per_group) * gid.y;
    let n_out = params.OW * params.OH * params.OC_out * params.N_out;

    var sum: f32 = 0.0;
    if (i_out >= n_out) {
        return;
    }

    var i = i_out;
    let n = i / (params.OC_out * params.OH * params.OW);
    i = i % (params.OC_out * params.OH * params.OW);
    let oc = i / (params.OH * params.OW);
    i = i % (params.OH * params.OW);
    let oh = i / params.OW;
    let ow = i % params.OW;

    // Kernel layout: [KW, KH, IC, OC]
    // Input layout:  [IW, IH, IC, N]
    // Output layout: [OW, OH, OC, N]

    for (var ic: u32 = 0; ic < params.IC; ic += 1) {
        let w_base_ic = ic * params.sw2 + params.sw3;
        let in_base = n * params.si3 + ic * params.si2;

       for (var kh: u32 = 0; kh < params.KH; kh += 1)  {
        for (var kw: u32 = 0; kw < params.KW; kw += 1) {
            let ih = i32(oh * params.s1) + i32(kh * params.d1) - i32(params.p1);
            let iw = i32(ow * params.s0) + i32(kw * params.d0) - i32(params.p0);

            if (ih >= 0 && ih < i32(params.IH) && iw >= 0 && iw < i32(params.IW)) {
                let w_idx = w_base_ic + kh * params.sw1 + kw * params.sw0 + params.offset_w;
                let in_idx = in_base + u32(ih) * params.si1 + u32(iw) * params.si0 + params.offset_i;

                sum += load_weight(w_idx) * load_input(in_idx);
            }
        }
       }
    }

    let out_idx = params.offset_o + ow * params.so0 + oh * params.so1 + oc * params.so2 + n * params.so3;
    store_output(out_idx, sum);
}