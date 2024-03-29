/**
 * @internal
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 XLib. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include "Host/Classes/BitRef.hpp"
#include "Host/Numeric.hpp"
#include <cstddef>

namespace xlib {

class Bitmask {
public:
    explicit Bitmask()            noexcept = default;
    explicit Bitmask(size_t size) noexcept;
    ~Bitmask() noexcept;

    void init(size_t size) noexcept;
    void free()            noexcept;
    void copy(const Bitmask& bitmask) noexcept;

    bool   operator[](size_t index) const noexcept;
    BitRef operator[](size_t index) noexcept;

    void   randomize()              noexcept;
    void   randomize(uint64_t seed) noexcept;
    void   clear()                  noexcept;
    void   fill()                   noexcept;
    size_t size()                   const noexcept;

    void print() const noexcept;

    Bitmask(const Bitmask&)        = delete;
    bool operator=(const Bitmask&) = delete;
private:
    size_t    _num_word { 0 };
    size_t    _size     { 0 };
    unsigned* _array    { nullptr };
};

//==============================================================================

template<unsigned SIZE>
class BitmaskStack {
public:
    explicit BitmaskStack() noexcept = default;

    bool   operator[](unsigned index) const noexcept;
    BitRef operator[](unsigned index) noexcept;

    void     clear()     noexcept;
    unsigned get_count() const noexcept;

    BitmaskStack(const BitmaskStack&)   = delete;
    bool operator=(const BitmaskStack&) = delete;
private:
    unsigned _array[xlib::CeilDiv<SIZE, 32>::value] = {};
};

} // namespace xlib

#include "impl/Bitmask.i.hpp"
