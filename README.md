# nseqops - High-Performance Vectorized Math Library

This library provides high-performance vectorized operations on sequences of `float64` values using AVX2 and SSE4.2 instructions. Designed for computationally intensive tasks, it delivers efficient arithmetic and linear algebra operations.

## Features

- **AVX2 and SSE4.2 Support**: Optimized for both AVX2 and SSE4.2 instruction sets to ensure compatibility and performance.
- **Vectorized Computation**: Accelerates numerical computations with vectorization.
- **Comprehensive API**: Includes sequence-to-sequence and sequence-to-scalar operations.
- **Precision**: Designed for `float64` values.

## Installation

### Requirements

- **Nim >= 2.2.0**: [Install Nim](https://nim-lang.org/)
- **nimsimd >= 1.3.1**: Ensure `nimsimd` is installed.
- **AVX2 or SSE4.2-Compatible CPU**: Ensure your CPU supports AVX2 or SSE4.2 instructions.

### Setup

Install the library using `nimble`:

```bash
nimble install https://github.com/nappworks/nseqops
```

## Usage

### Import the Library

For AVX2:
```nim
import nseqops/avx2
```

For SSE4.2:
```nim
import nseqops/sse4
```

### Examples

#### Summation of a Sequence
```nim
let data = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
echo sumSeq(data)  # Output: 28.0
```

#### Element-wise Operations
```nim
let a = @[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
let b = @[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

# Element-wise addition
echo addSeqs(a, b)  # Output: @[8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]

# Dot product
echo dotSeqs(a, b)  # Output: 84.0
```

#### Scalar Operations
```nim
# Add a scalar value to each element
echo addSeqScalar(a, 3.0)  # Output: @[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Multiply each element by a scalar
echo mulSeqScalar(a, 2.0)  # Output: @[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
```

## API Reference

### Sequence Operations

1. `sumSeq(data: seq[float64]): float64`
2. `prodSeq(data: seq[float64]): float64`
3. `addSeqs(a, b: seq[float64]): seq[float64]`
4. `subSeqs(a, b: seq[float64]): seq[float64]`
5. `mulSeqs(a, b: seq[float64]): seq[float64]`
6. `divSeqs(a, b: seq[float64]): seq[float64]`
7. `dotSeqs(a, b: seq[float64]): float64`

### Scalar Operations

1. `addSeqScalar(a: seq[float64], b: float64): seq[float64]`
2. `subSeqScalar(a: seq[float64], b: float64): seq[float64]`
3. `mulSeqScalar(a: seq[float64], b: float64): seq[float64]`
4. `divSeqScalar(a: seq[float64], b: float64): seq[float64]`

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Additional Notes

- This library currently supports only `float64` values.
- For SSE4.2, calculations are performed with 128-bit registers (`float64x2`).
- For AVX2, calculations are performed with 256-bit registers (`float64x4`).
- Support for AVX-512 is planned for future updates.

### Compilation Instructions

- **Using AVX**
  ```
  nim c -r --passC:"-mavx" hogehoge.nim
  ```

- **Using SSE4**
  ```
  nim c -r --passC:"-msse4" hogehoge.nim
  ```

> **Note**: Failing to include the appropriate flags will result in compilation errors.

---

Let us know if you encounter any issues or need additional features!

---
