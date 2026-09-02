#pragma once

// The Hadamard basis the exl3 trellis format is quantized and served in, and
// the normalisation that belongs to it.
//
// One number, one derivation. The width was written out as ``1/sqrt(128)`` at
// every use, and the sites that would have to move if the basis widened are
// indistinguishable in source from the sites that must not: an exl3 linear is
// wrapped in TWO independent rotations, one over the contraction axis and one
// over the output axis, and widening either says nothing about the other. A
// find-and-replace over the literal is therefore wrong at some of them.
//
// ``had_r_scale`` is exact for a power-of-two width. ``1/sqrt(4w) ==
// 0.5 * 1/sqrt(w)`` is an exponent decrement, so the recursion introduces no
// rounding of its own and the only inexact value in it is the correctly
// rounded float of ``1/sqrt(2)``. C++ has no ``constexpr sqrt``, which is why
// it is spelled this way rather than divided out. The ``static_assert`` pins
// the result to the literal every call site carried before this header, so the
// derivation cannot drift from the arithmetic that ships.
//
// WIDENING THE BASIS starts here and does not end here: this constant then has
// to become one per AXIS, and every kernel's lane layout -- four elements per
// lane across a warp is what makes the width 128 -- is a rewrite, not a
// constant change.

namespace exl3_had {

constexpr int HAD_BASIS_WIDTH = 128;

constexpr float HAD_RSQRT2 = 0.70710678118654752440f;

constexpr float had_r_scale(int width)
{
    return width <= 1 ? 1.0f
         : width == 2 ? HAD_RSQRT2
         : had_r_scale(width / 4) * 0.5f;
}

static_assert(had_r_scale(HAD_BASIS_WIDTH) == 0.088388347648f,
              "had_r_scale(128) must be the exl3 r_scale bit for bit");

// exl3's ``r_scale``: what ``had_r_128`` computes for ``scale == 1.0f``.
constexpr float HAD_R_SCALE = had_r_scale(HAD_BASIS_WIDTH);

}  // namespace exl3_had
