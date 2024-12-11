{.passC: "-msse4".}
{.passL: "-msse4".}
## This module provides functions for performing arithmetic operations on sequences
## of float64 values using SSE42 instructions for vectorization.
##
## Usage Examples
## --------------
##
## ```nim
## import vector_math
##
## let dataA = @[1.0, 2.0, 3.0, 4.0]
## let dataB = @[5.0, 6.0, 7.0, 8.0]
##
## echo sumSeq(dataA)           # Output: 10.0
## echo prodSeq(dataA)          # Output: 24.0
## echo addSeqs(dataA, dataB)   # Output: @[6.0, 8.0, 10.0, 12.0]
## ```
##
## Each function is designed to take advantage of AVX instructions for faster
## computation when operating on sequences of float64 values.

import nimsimd/sse42

proc sumSeq*(data: seq[float64]): float64 =
  ## Sums the elements of the given sequence using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `data`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A float64 value representing the sum of all elements in the sequence.
  runnableExamples:
    let data = @[1.0, 2.0, 3.0, 4.0]
    doAssert sumSeq(data) == 10.0

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    sumVec: M128d = mm_setzero_pd()
    i = 0
  let
    dataLen = data.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Sum the vectorized part using SSE2
  while i < alignedLen:
    let vec = mm_loadu_pd(addr data[i])
    sumVec = mm_add_pd(sumVec, vec)
    i += VecSize

  # Sum the elements of sumVec
  var sumArray: array[VecSize, float64]
  mm_storeu_pd(addr sumArray[0], sumVec)
  var totalSum: float64 = 0.0
  for j in 0 ..< VecSize:
    totalSum += sumArray[j]

  # Sum the remaining elements
  for k in i ..< dataLen:
    totalSum += data[k]

  return totalSum

proc prodSeq*(data: seq[float64]): float64 =
  ## Calculates the product of all elements in the given sequence using SSE2 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `data`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A float64 value representing the product of all elements in the sequence.
  runnableExamples:
    let data = @[1.0, 2.0, 3.0, 4.0]
    doAssert prodSeq(data) == 24.0
    let dataWithOne = @[1.0, 5.0, 1.0, 10.0]
    doAssert prodSeq(dataWithOne) == 50.0
    let emptyData = @[]
    doAssert prodSeq(emptyData) == 1.0  # Assuming the product of an empty sequence is 1.0

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    prodVec: M128d = mm_set1_pd(1.0)
    i = 0
  let
    dataLen = data.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Multiply the vectorized part using SSE2
  while i < alignedLen:
    let vec = mm_loadu_pd(addr data[i])
    prodVec = mm_mul_pd(prodVec, vec)
    i += VecSize

  # Multiply the elements of prodVec
  var prodArray: array[VecSize, float64]
  mm_storeu_pd(addr prodArray[0], prodVec)
  var totalProd: float64 = 1.0
  for j in 0 ..< VecSize:
    totalProd *= prodArray[j]

  # Multiply the remaining elements
  for k in i ..< dataLen:
    totalProd *= data[k]

  return totalProd

proc addSeqs*(a, b: seq[float64]): seq[float64] =
  ## Adds two sequences element-wise using SSE2 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `a`: The first sequence of float64 values.
  ## - `b`: The second sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence where each element is the sum of the corresponding elements from `a` and `b`.
  ##
  ## Notes
  ## -----
  ## The lengths of both sequences must be equal.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let seqB = @[5.0, 6.0, 7.0, 8.0]
    doAssert addSeqs(seqA, seqB) == @[6.0, 8.0, 10.0, 12.0]
    let seqC = @[0.0, -1.0, 2.5]
    let seqD = @[1.0, 1.0, -2.5]
    doAssert addSeqs(seqC, seqD) == @[1.0, 0.0, 0.0]
    let seqE = @[]
    let seqF = @[]
    doAssert addSeqs(seqE, seqF) == @[]

  assert a.len == b.len, "Sequences must have the same length"

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Add the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecB = mm_loadu_pd(addr b[i])
    let vecR = mm_add_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Add the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] + b[j]

  return result

proc subSeqs*(a, b: seq[float64]): seq[float64] =
  ## Subtracts the second sequence from the first sequence element-wise using SSE2 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `a`: The first sequence of float64 values.
  ## - `b`: The second sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence where each element is the difference of the corresponding elements from `a` and `b`.
  ##
  ## Notes
  ## -----
  ## The lengths of both sequences must be equal.
  runnableExamples:
    let seqA = @[5.0, 10.0, 15.0, 20.0]
    let seqB = @[1.0, 2.0, 3.0, 4.0]
    doAssert subSeqs(seqA, seqB) == @[4.0, 8.0, 12.0, 16.0]
    let seqC = @[0.0, -1.0, 2.5]
    let seqD = @[1.0, 1.0, 0.5]
    doAssert subSeqs(seqC, seqD) == @[-1.0, -2.0, 2.0]
    let seqE = @[]
    let seqF = @[]
    doAssert subSeqs(seqE, seqF) == @[]

  assert a.len == b.len, "Sequences must have the same length"

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Subtract the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecB = mm_loadu_pd(addr b[i])
    let vecR = mm_sub_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] - b[j]

  return result


proc mulSeqs*(a, b: seq[float64]): seq[float64] =
  ## Multiplies two sequences element-wise using SSE2 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `a`: The first sequence of float64 values.
  ## - `b`: The second sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence where each element is the product of the corresponding elements from `a` and `b`.
  ##
  ## Notes
  ## -----
  ## The lengths of both sequences must be equal.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let seqB = @[5.0, 6.0, 7.0, 8.0]
    doAssert mulSeqs(seqA, seqB) == @[5.0, 12.0, 21.0, 32.0]
    let seqC = @[0.5, -1.0, 2.5]
    let seqD = @[2.0, 1.0, -2.0]
    doAssert mulSeqs(seqC, seqD) == @[1.0, -1.0, -5.0]
    let seqE = @[]
    let seqF = @[]
    doAssert mulSeqs(seqE, seqF) == @[]

  assert a.len == b.len, "Sequences must have the same length"

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Multiply the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecB = mm_loadu_pd(addr b[i])
    let vecR = mm_mul_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Multiply the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] * b[j]

  return result


proc divSeqs*(a, b: seq[float64]): seq[float64] =
  ## Divides the first sequence by the second sequence element-wise using SSE2 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `a`: The first sequence of float64 values (numerator).
  ## - `b`: The second sequence of float64 values (denominator).
  ##
  ## Returns
  ## -------
  ## - A new sequence where each element is the quotient of the corresponding elements from `a` and `b`.
  ##
  ## Notes
  ## -----
  ## The lengths of both sequences must be equal.
  runnableExamples:
    let seqA = @[10.0, 20.0, 30.0, 40.0]
    let seqB = @[2.0, 4.0, 5.0, 8.0]
    doAssert divSeqs(seqA, seqB) == @[5.0, 5.0, 6.0, 5.0]
    let seqC = @[0.0, -10.0, 25.0]
    let seqD = @[1.0, 5.0, -5.0]
    doAssert divSeqs(seqC, seqD) == @[0.0, -2.0, -5.0]
    let seqE = @[]
    let seqF = @[]
    doAssert divSeqs(seqE, seqF) == @[]

  assert a.len == b.len, "Sequences must have the same length"

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Divide the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecB = mm_loadu_pd(addr b[i])
    let vecR = mm_div_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] / b[j]

  return result

proc dotSeqs*(a, b: seq[float64]): float64 =
  ## Computes the dot product of two sequences using SSE3 vectorization.
  ##
  ## Parameters
  ## ----------
  ## - `a`: The first sequence of float64 values.
  ## - `b`: The second sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A float64 value representing the dot product of the two sequences.
  ##
  ## Notes
  ## -----
  ## The lengths of both sequences must be equal.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let seqB = @[5.0, 6.0, 7.0, 8.0]
    doAssert dotSeqs(seqA, seqB) == 70.0
    let seqC = @[0.5, -1.0, 2.5]
    let seqD = @[2.0, 1.0, -2.0]
    doAssert dotSeqs(seqC, seqD) == -5.5
    let seqE = @[]
    let seqF = @[]
    doAssert dotSeqs(seqE, seqF) == 0.0

  assert a.len == b.len, "Sequences must have the same length"

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    sumVec: M128d = mm_setzero_pd()
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Calculate the dot product of the vectorized part using SSE3
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecB = mm_loadu_pd(addr b[i])
    let vecR = mm_mul_pd(vecA, vecB)
    sumVec = mm_add_pd(sumVec, vecR)
    i += VecSize

  # Horizontal add to sum the elements of sumVec
  sumVec = mm_hadd_pd(sumVec, sumVec)
  var totalSum: float64
  mm_store_sd(addr totalSum, sumVec)

  # Calculate the dot product of the remaining elements
  for k in i ..< dataLen:
    totalSum += a[k] * b[k]

  return totalSum


proc addSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Adds a scalar value to each element in the sequence using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `a`: A sequence of float64 values.
  ## - `b`: A float64 value to add to each element of the sequence.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element incremented by the scalar value `b`.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let scalarB = 5.0
    doAssert addSeqScalar(seqA, scalarB) == @[6.0, 7.0, 8.0, 9.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = -2.0
    doAssert addSeqScalar(seqC, scalarD) == @[-2.0, -3.0, 0.5]
    let seqE = @[]
    let scalarF = 10.0
    doAssert addSeqScalar(seqE, scalarF) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Add the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_add_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Add the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] + b

  return result

proc addScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Adds a scalar value to each element in the sequence (commutative version).
  ##
  ## Parameters
  ## ----------
  ## - `b`: A float64 value to add to each element of the sequence.
  ## - `a`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element incremented by the scalar value `b`.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let scalarB = 5.0
    doAssert addSeqScalar(scalarB, seqA) == @[6.0, 7.0, 8.0, 9.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = -2.0
    doAssert addSeqScalar(scalarD, seqC) == @[-2.0, -3.0, 0.5]
    let seqE = @[]
    let scalarF = 10.0
    doAssert addSeqScalar(scalarF, seqE) == @[]

  return addSeqScalar(a, b)

proc subSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Subtracts a scalar value from each element in the sequence using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `a`: A sequence of float64 values.
  ## - `b`: A float64 value to subtract from each element of the sequence.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element decremented by the scalar value `b`.
  runnableExamples:
    let seqA = @[10.0, 20.0, 30.0, 40.0]
    let scalarB = 5.0
    doAssert subSeqScalar(seqA, scalarB) == @[5.0, 15.0, 25.0, 35.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = -2.0
    doAssert subSeqScalar(seqC, scalarD) == @[2.0, 1.0, 4.5]
    let seqE = @[]
    let scalarF = 10.0
    doAssert subSeqScalar(seqE, scalarF) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Subtract the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_sub_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] - b

  return result

proc subScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Subtracts each element in the sequence from a scalar value using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `b`: A float64 value from which each element of the sequence will be subtracted.
  ## - `a`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence with the result of `b - a[i]` for each element `a[i]` in the sequence.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let scalarB = 10.0
    doAssert subScalarSeq(scalarB, seqA) == @[9.0, 8.0, 7.0, 6.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = 5.0
    doAssert subScalarSeq(scalarD, seqC) == @[5.0, 6.0, 2.5]
    let seqE = @[]
    let scalarF = 10.0
    doAssert subScalarSeq(scalarF, seqE) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Subtract the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_sub_pd(vecB, vecA)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = b - a[j]

  return result

proc mulSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Multiplies each element in the sequence by a scalar value using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `a`: A sequence of float64 values.
  ## - `b`: A float64 value to multiply each element of the sequence by.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element multiplied by the scalar value `b`.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let scalarB = 2.0
    doAssert mulSeqScalar(seqA, scalarB) == @[2.0, 4.0, 6.0, 8.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = -2.0
    doAssert mulSeqScalar(seqC, scalarD) == @[0.0, 2.0, -5.0]
    let seqE = @[]
    let scalarF = 10.0
    doAssert mulSeqScalar(seqE, scalarF) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Multiply the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_mul_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Multiply the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] * b

  return result

proc mulScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Multiplies each element in the sequence by a scalar value (commutative version).
  ##
  ## Parameters
  ## ----------
  ## - `b`: A float64 value to multiply each element of the sequence by.
  ## - `a`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element multiplied by the scalar value `b`.
  runnableExamples:
    let seqA = @[1.0, 2.0, 3.0, 4.0]
    let scalarB = 2.0
    doAssert mulSeqScalar(scalarB, seqA) == @[2.0, 4.0, 6.0, 8.0]
    let seqC = @[0.0, -1.0, 2.5]
    let scalarD = -2.0
    doAssert mulSeqScalar(scalarD, seqC) == @[0.0, 2.0, -5.0]
    let seqE = @[]
    let scalarF = 10.0
    doAssert mulSeqScalar(scalarF, seqE) == @[]

  return mulSeqScalar(a, b)

proc divSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Divides each element in the sequence by a scalar value using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `a`: A sequence of float64 values.
  ## - `b`: A float64 value to divide each element of the sequence by.
  ##
  ## Returns
  ## -------
  ## - A new sequence with each element divided by the scalar value `b`.
  runnableExamples:
    let seqA = @[10.0, 20.0, 30.0, 40.0]
    let scalarB = 10.0
    doAssert divSeqScalar(seqA, scalarB) == @[1.0, 2.0, 3.0, 4.0]
    let seqC = @[0.0, -10.0, 25.0]
    let scalarD = 5.0
    doAssert divSeqScalar(seqC, scalarD) == @[0.0, -2.0, 5.0]
    let seqE = @[]
    let scalarF = 1.0
    doAssert divSeqScalar(seqE, scalarF) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Divide the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_div_pd(vecA, vecB)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] / b

  return result

proc divScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Divides a scalar value by each element in the sequence using SSE2 vectorization for performance.
  ##
  ## Parameters
  ## ----------
  ## - `b`: A float64 value to be divided by each element of the sequence.
  ## - `a`: A sequence of float64 values.
  ##
  ## Returns
  ## -------
  ## - A new sequence with the result of `b / a[i]` for each element `a[i]` in the sequence.
  runnableExamples:
    let seqA = @[1.0, 2.0, 4.0, 8.0]
    let scalarB = 16.0
    doAssert divScalarSeq(scalarB, seqA) == @[16.0, 8.0, 4.0, 2.0]
    let seqC = @[-1.0, 0.5, 2.0]
    let scalarD = 2.0
    doAssert divScalarSeq(scalarD, seqC) == @[-2.0, 4.0, 1.0]
    let seqE = @[]
    let scalarF = 1.0
    doAssert divScalarSeq(scalarF, seqE) == @[]

  const
    VecSize = 2  # __m128d holds 2 float64 values

  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Divide the vectorized part using SSE2
  while i < alignedLen:
    let vecA = mm_loadu_pd(addr a[i])
    let vecR = mm_div_pd(vecB, vecA)
    mm_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen:
    result[j] = b / a[j]

  return result