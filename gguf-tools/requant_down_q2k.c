/*
 * requant_down_q2k.c
 *
 * Convert the routed-expert down tensors of a DeepSeek GGUF in place from
 * IQ3_XXS (type 18) or MXFP4 (type 39) to Q2_K (type 10), the combination
 * the ds4fa ROCm routed-MoE kernel set supports (IQ2_XXS gate/up + Q2_K
 * down).
 *
 * The GGUF metadata section is byte-identical in size (tensor info records
 * keep the same field widths), so the conversion compacts the data section
 * in place: each down tensor shrinks from 98 B/block (or 136 B/32-elem for
 * MXFP4) to 84 B/256-elem, and the write cursor never overtakes the read
 * cursor while tensors are processed in order.  The file is truncated at
 * the end.
 *
 * usage: requant_down_q2k MODEL.gguf [--dry-run]
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "quants.h"

#define QK_K 256

#define IQ3_XXS_TYPE 18
#define MXFP4_TYPE 39
#define Q2_K_TYPE 10
#define IQ3_XXS_BLOCK 98
#define MXFP4_BLOCK 136
#define Q2_K_BLOCK 84

static const int8_t kvalues_fp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static int down_block_size(uint32_t type) {
    return type == IQ3_XXS_TYPE ? IQ3_XXS_BLOCK : MXFP4_BLOCK;
}

static const uint8_t kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

static const uint8_t ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint32_t iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

/* ---- fp16 helpers (ggml-compatible) ---- */

static uint32_t f32_to_f16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t m = x & 0x7fffffffu;
    if (m >= 0x7f800000u) return sign | 0x7c00u;
    if (m > 0x47ffefffu) return sign | 0x7c00u;
    if (m < 0x38800000u) {
        if (m < 0x33000000u) return sign;
        m >>= 23;
        uint32_t mm = (x & 0x7fffffu) | 0x800000u;
        mm >>= (m - 0x70);
        if (mm & 0x1000u) mm += 0x1000u;
        return sign | (mm >> 13);
    }
    m -= 0x38000000u;
    if (m & 0x1000u) m += 0x1000u;
    return sign | (m >> 13);
}

static float f16_bits_to_f32(uint32_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t x;
    if (e == 0u) {
        x = sign | (m << 13);
    } else if (e == 0x1fu) {
        x = sign | 0x7f800000u | (m << 13);
    } else {
        x = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

/* ---- IQ3_XXS dequant (port of ggml dequantize_row_iq3_xxs) ---- */

void requant_dequant_iq3_xxs_row(const uint8_t *block, float *out, int ncols) {
    uint16_t dh;
    memcpy(&dh, block, 2);
    const float d = f16_bits_to_f32(dh);
    const uint8_t *qs = block + 2;
    const uint8_t *sas = qs + QK_K / 4;

    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        uint32_t aux32;
        memcpy(&aux32, sas + 4 * ib32, 4);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
            const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 0]);
            const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
            for (int j = 0; j < 4; ++j) {
                out[j + 0] = db * (float)grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.0f : 1.0f);
                out[j + 4] = db * (float)grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
            }
            out += 8;
        }
        qs += 8;
    }
    (void)ncols;
}

/* ---- MXFP4 dequant (port of ggml dequantize_row_mxfp4) ---- */

void requant_dequant_mxfp4_row(const uint8_t *block, float *out, int ncols) {
    const uint8_t e = block[0];
    float d = 0.0f;
    if (e != 0) {
        d = ldexpf(1.0f, (int)e - 127);
    }
    const uint8_t *qs = block + 1;
    for (int j = 0; j < 16; ++j) {
        out[2 * j + 0] = kvalues_fp4[qs[j] & 0x0F] * d;
        out[2 * j + 1] = kvalues_fp4[qs[j] >> 4] * d;
    }
    (void)ncols;
}

/* ---- GGUF parsing ---- */

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[4];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t bytes;
    uint64_t new_offset;
    uint64_t type_pos;
    uint64_t offset_pos;
} tensor_info;

typedef struct {
    int fd;
    uint64_t n_tensors;
    tensor_info *tensors;
    uint64_t data_start;
    uint64_t alignment;
    uint64_t old_total;
    uint64_t new_total;
    int dry_run;
    int verbose;
} ctx;

static void die(const char *msg) {
    fprintf(stderr, "requant_down_q2k: %s\n", msg);
    exit(1);
}

static uint64_t align_up(uint64_t v, uint64_t a) {
    uint64_t r = v % a;
    return r == 0 ? v : v + a - r;
}

static uint64_t tensor_old_bytes(uint32_t type, uint64_t ne0, uint64_t elements) {
    switch (type) {
    case 0:  return 4 * elements;           /* f32 */
    case 1:  return 2 * elements;           /* f16 */
    case 30: return 2 * elements;           /* bf16 */
    case 8:  return 34 * (ne0 / 32) * (elements / ne0);
    case 10: return 84 * (ne0 / 256) * (elements / ne0);
    case 12: return 144 * (ne0 / 256) * (elements / ne0);
    case 13: return 176 * (ne0 / 256) * (elements / ne0);
    case 14: return 210 * (ne0 / 256) * (elements / ne0);
    case 16: return 66 * (ne0 / 256) * (elements / ne0);
    case 18: return 98 * (ne0 / 256) * (elements / ne0);
    case 22: return 82 * (ne0 / 256) * (elements / ne0);
    case 26: return 4 * elements;           /* i32 */
    case 39: return 17 * (ne0 / 32) * (elements / ne0); /* mxfp4 */
    default:
        fprintf(stderr, "requant_down_q2k: unsupported tensor type %u in layout\n", type);
        return 0;
    }
}

static int is_down_expert(const char *name, uint32_t type) {
    if (type != IQ3_XXS_TYPE && type != MXFP4_TYPE) return 0;
    return strstr(name, "ffn_down_exps.weight") != NULL;
}

static uint64_t read_u64(int fd, uint64_t pos) {
    uint64_t v;
    if (pread(fd, &v, 8, (off_t)pos) != 8) die("short read");
    return v;
}

static void parse_gguf(ctx *c, const char *path) {
    c->fd = open(path, O_RDWR);
    if (c->fd < 0) die("cannot open model for read/write");
    uint8_t hdr[24];
    if (pread(c->fd, hdr, 24, 0) != 24) die("short header");
    if (memcmp(hdr, "GGUF", 4) != 0) die("not a GGUF file");
    uint64_t n_tensors, n_kv;
    memcpy(&n_tensors, hdr + 8, 8);
    memcpy(&n_kv, hdr + 16, 8);
    c->n_tensors = n_tensors;
    printf("header: version %u, %llu tensors, %llu KV pairs\n",
           *(uint32_t *)(hdr + 4), (unsigned long long)n_tensors,
           (unsigned long long)n_kv);

    uint64_t pos = 24;
    c->alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = read_u64(c->fd, pos); pos += 8;
        char *kname = malloc((size_t)klen + 1);
        if (pread(c->fd, kname, (size_t)klen, (off_t)pos) != (ssize_t)klen) die("kv name read");
        kname[klen] = 0;
        pos += klen;
        uint32_t ktype;
        if (pread(c->fd, &ktype, 4, (off_t)pos) != 4) die("kv type read");
        pos += 4;
        switch (ktype) {
        case 0: case 1: pos += 1; break;
        case 2: case 3: pos += 2; break;
        case 4: case 5: case 6: pos += 4; break;
        case 7: pos += 1; break;
        case 8: {
            uint64_t vlen = read_u64(c->fd, pos); pos += 8 + vlen;
            break;
        }
        case 9: {
            uint32_t etype; uint64_t n;
            if (pread(c->fd, &etype, 4, (off_t)pos) != 4) die("array type read");
            pos += 4;
            n = read_u64(c->fd, pos); pos += 8;
            uint64_t esize = (etype == 8) ? 0
                           : (etype == 0 || etype == 1 || etype == 7) ? 1
                           : (etype == 2 || etype == 3) ? 2
                           : (etype == 4 || etype == 5 || etype == 6) ? 4 : 8;
            if (etype == 8) {
                for (uint64_t j = 0; j < n; j++) {
                    uint64_t vlen = read_u64(c->fd, pos); pos += 8 + vlen;
                }
            } else {
                pos += esize * n;
            }
            break;
        }
        case 10: case 11: case 12: pos += 8; break;
        default: die("unknown GGUF metadata type");
        }
        if (strcmp(kname, "general.alignment") == 0 && ktype == 4) {
            uint32_t align;
            if (pread(c->fd, &align, 4, (off_t)(pos - 4)) == 4 && align != 0) {
                c->alignment = align;
            }
        }
        free(kname);
    }
    printf("metadata ends at %llu, alignment %llu\n",
           (unsigned long long)pos, (unsigned long long)c->alignment);

    c->tensors = calloc((size_t)n_tensors, sizeof(tensor_info));
    if (!c->tensors) die("out of memory");
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &c->tensors[i];
        uint64_t nlen = read_u64(c->fd, pos); pos += 8;
        t->name = malloc((size_t)nlen + 1);
        if (pread(c->fd, t->name, (size_t)nlen, (off_t)pos) != (ssize_t)nlen) die("tensor name read");
        t->name[nlen] = 0;
        pos += nlen;
        uint32_t ndim;
        if (pread(c->fd, &ndim, 4, (off_t)pos) != 4) die("tensor ndim read");
        pos += 4;
        t->ndim = ndim;
        if (ndim == 0 || ndim > 4) die("tensor ndim out of range");
        uint64_t elements = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            t->dim[d] = read_u64(c->fd, pos); pos += 8;
            elements *= t->dim[d];
        }
        t->type_pos = pos;
        uint32_t type;
        if (pread(c->fd, &type, 4, (off_t)pos) != 4) die("tensor type read");
        pos += 4;
        t->type = type;
        t->offset_pos = pos;
        t->rel_offset = read_u64(c->fd, pos); pos += 8;
        t->bytes = tensor_old_bytes(type, t->dim[0], elements);
        if (t->bytes == 0) die("tensor byte size unsupported");
    }
    c->data_start = align_up(pos, c->alignment);
    printf("tensor directory ends at %llu, data starts at %llu\n",
           (unsigned long long)pos, (unsigned long long)c->data_start);
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info *t = &c->tensors[i];
        t->abs_offset = c->data_start + t->rel_offset;
        if (t->abs_offset < c->data_start) die("tensor offset overflow");
        if (i + 1 < n_tensors) {
            tensor_info *n = &c->tensors[i + 1];
            if (n->rel_offset < t->rel_offset + t->bytes) {
                fprintf(stderr, "tensor overlap: %s and %s\n", t->name, n->name);
                die("tensor directory is not monotonically ordered");
            }
        }
    }
}

static uint64_t compute_layout(ctx *c) {
    uint64_t cum = c->data_start;
    uint64_t n_down = 0;
    for (uint64_t i = 0; i < c->n_tensors; i++) {
        tensor_info *t = &c->tensors[i];
        t->new_offset = align_up(cum, c->alignment);
        uint64_t new_bytes = t->bytes;
        if (is_down_expert(t->name, t->type)) {
            new_bytes = (t->bytes / down_block_size(t->type)) * Q2_K_BLOCK;
            n_down++;
        }
        cum = t->new_offset + new_bytes;
        if (cum > t->abs_offset + t->bytes) {
            fprintf(stderr, "in-place safety violated at tensor %s\n", t->name);
            die("new layout would clobber unread data");
        }
        if (c->verbose) {
            printf("  %-56s type %u -> %u size %llu -> %llu off %llu -> %llu\n",
                   t->name, t->type,
                   is_down_expert(t->name, t->type) ? Q2_K_TYPE : t->type,
                   (unsigned long long)t->bytes,
                   (unsigned long long)new_bytes,
                   (unsigned long long)t->abs_offset,
                   (unsigned long long)t->new_offset);
        }
    }
    printf("down expert tensors converted: %llu, data section %llu -> %llu bytes\n",
           (unsigned long long)n_down, (unsigned long long)(c->data_start + (c->tensors[c->n_tensors - 1].rel_offset + c->tensors[c->n_tensors - 1].bytes)),
           (unsigned long long)cum);
    c->new_total = cum;
    return n_down;
}

typedef struct {
    const tensor_info *t;
    int fd;
    int tid;
    int nthreads;
    uint8_t *out;      /* output buffer, new_bytes */
    const uint8_t *in; /* input buffer, bytes */
} qjob;

static void *quant_worker(void *arg) {
    qjob *j = (qjob *)arg;
    const tensor_info *t = j->t;
    const uint64_t blocks = t->bytes / down_block_size(t->type);
    const uint64_t per = (blocks + (uint64_t)j->nthreads - 1) / (uint64_t)j->nthreads;
    const uint64_t b0 = (uint64_t)j->tid * per;
    const uint64_t b1 = b0 + per < blocks ? b0 + per : blocks;
    float *f32 = malloc((size_t)(b1 - b0) * QK_K * sizeof(float));
    if (!f32) return (void *)1;
    const int bsize = down_block_size(t->type);
    for (uint64_t b = b0; b < b1; b++) {
        if (t->type == IQ3_XXS_TYPE) {
            requant_dequant_iq3_xxs_row(j->in + b * bsize, f32 + (b - b0) * QK_K, QK_K);
        } else {
            for (int s = 0; s < QK_K / 32; s++) {
                requant_dequant_mxfp4_row(j->in + (b * (QK_K / 32) + s) * 17,
                                          f32 + (b - b0) * QK_K + 32 * s, 32);
            }
        }
    }
    ds4q_quantize_chunk(DS4Q_TYPE_Q2_K, f32, j->out + b0 * Q2_K_BLOCK, 0,
                        (int64_t)(b1 - b0), QK_K, NULL);
    free(f32);
    return NULL;
}

static void convert_tensor(ctx *c, tensor_info *t) {
    uint8_t *in = NULL, *out = NULL;
    const uint64_t copy_off = t->abs_offset;
    const uint64_t write_off = t->new_offset;
    if (is_down_expert(t->name, t->type)) {
        const uint64_t blocks = t->bytes / down_block_size(t->type);
        const uint64_t new_bytes = blocks * Q2_K_BLOCK;
        in = malloc((size_t)t->bytes);
        if (!in) die("out of memory for input tensor");
        if (pread(c->fd, in, (size_t)t->bytes, (off_t)copy_off) != (ssize_t)t->bytes) {
            die("failed to read down expert tensor");
        }
        out = malloc((size_t)new_bytes);
        if (!out) die("out of memory for output tensor");
        const int nthreads = 16;
        pthread_t th[16];
        qjob jobs[16];
        int rc = 0;
        for (int i = 0; i < nthreads; i++) {
            jobs[i].t = t; jobs[i].fd = c->fd; jobs[i].tid = i;
            jobs[i].nthreads = nthreads; jobs[i].out = out; jobs[i].in = in;
            if (pthread_create(&th[i], NULL, quant_worker, &jobs[i]) != 0) {
                rc = 1; break;
            }
        }
        for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
        if (rc) die("thread create failed");
        if (pwrite(c->fd, out, (size_t)new_bytes, (off_t)write_off) != (ssize_t)new_bytes) {
            die("failed to write down expert tensor");
        }
        free(in);
        free(out);
        printf("  converted %s (%llu -> %llu bytes)\n",
               t->name, (unsigned long long)t->bytes,
               (unsigned long long)new_bytes);
    } else {
        uint8_t buf[1u << 20];
        uint64_t left = t->bytes;
        uint64_t rpos = copy_off, wpos = write_off;
        while (left > 0) {
            size_t chunk = left < sizeof(buf) ? (size_t)left : sizeof(buf);
            ssize_t got = pread(c->fd, buf, chunk, (off_t)rpos);
            if (got != (ssize_t)chunk) die("copy read failed");
            ssize_t put = pwrite(c->fd, buf, chunk, (off_t)wpos);
            if (put != (ssize_t)chunk) die("copy write failed");
            rpos += chunk;
            wpos += chunk;
            left -= chunk;
        }
    }
    if (c->verbose) {
        printf("  moved %s -> %llu\n", t->name, (unsigned long long)write_off);
    }
}

static void rewrite_tensor_info(ctx *c) {
    for (uint64_t i = 0; i < c->n_tensors; i++) {
        tensor_info *t = &c->tensors[i];
        uint32_t new_type = is_down_expert(t->name, t->type) ? Q2_K_TYPE : t->type;
        uint64_t new_rel = t->new_offset - c->data_start;
        if (pwrite(c->fd, &new_type, 4, (off_t)t->type_pos) != 4) die("type patch failed");
        if (pwrite(c->fd, &new_rel, 8, (off_t)t->offset_pos) != 8) die("offset patch failed");
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && strcmp(argv[2], "--dry-run") != 0)) {
        fprintf(stderr, "usage: %s MODEL.gguf [--dry-run]\n", argv[0]);
        return 1;
    }
    ctx c;
    memset(&c, 0, sizeof(c));
    c.dry_run = argc == 3;
    c.verbose = getenv("REQUANT_VERBOSE") != NULL;
    parse_gguf(&c, argv[1]);
    if (compute_layout(&c) == 0) {
        printf("no IQ3_XXS down expert tensors found; nothing to do\n");
        return 0;
    }
    if (c.dry_run) {
        printf("dry-run: layout computed, no changes written\n");
        return 0;
    }
    printf("converting data section...\n");
    for (uint64_t i = 0; i < c.n_tensors; i++) {
        convert_tensor(&c, &c.tensors[i]);
    }
    printf("rewriting tensor directory...\n");
    rewrite_tensor_info(&c);
    if (ftruncate(c.fd, (off_t)c.new_total) != 0) die("truncate failed");
    fsync(c.fd);
    printf("done: file truncated to %llu bytes\n", (unsigned long long)c.new_total);
    return 0;
}
