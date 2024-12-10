## This module provides functions for performing arithmetic operations on sequences
## of float64 values using AVX2 instructions for vectorization.
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

import nimsimd/avx2

func sumSeq*(data: seq[float64]): float64 =
  ## Sums the elements of the given sequence using AVX vectorization for performance.
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
    VecSize = 4  # __m256d holds 4 float64 values
  var
    sumVec: M256d = mm256_setzero_pd()
    i = 0
  let
    dataLen = data.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Sum the vectorized part using AVX
  while i < alignedLen:
    let vec = mm256_loadu_pd(addr data[i])
    sumVec = mm256_add_pd(sumVec, vec)
    i += VecSize

  # Sum the elements of sumVec
  var sumArray: array[VecSize, float64]
  mm256_storeu_pd(addr sumArray[0], sumVec)
  var totalSum: float64 = 0.0
  for j in 0 ..< VecSize:
    totalSum += sumArray[j]

  # Sum the remaining elements
  for k in i ..< dataLen:
    totalSum += data[k]

  return totalSum


func prodSeq*(data: seq[float64]): float64 =
  ## Calculates the product of all elements in the given sequence using AVX vectorization.
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
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    prodVec: M256d = mm256_set1_pd(1.0)
    i = 0
  let
    dataLen = data.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Multiply the vectorized part using AVX
  while i < alignedLen:
    let vec = mm256_loadu_pd(addr data[i])
    prodVec = mm256_mul_pd(prodVec, vec)
    i += VecSize

  # Multiply the elements of prodVec
  var prodArray: array[VecSize, float64]
  mm256_storeu_pd(addr prodArray[0], prodVec)
  var totalProd: float64 = 1.0
  for j in 0 ..< VecSize:
    totalProd *= prodArray[j]

  # Multiply the remaining elements
  for k in i ..< dataLen:
    totalProd *= data[k]

  return totalProd

func addSeqs*(a, b: seq[float64]): seq[float64] =
  ## Adds two sequences element-wise using AVX vectorization.
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
    let a = @[1.0, 2.0, 3.0, 4.0]
    let b = @[5.0, 6.0, 7.0, 8.0]
    doAssert addSeqs(a, b) == @[6.0, 8.0, 10.0, 12.0]
  assert a.len == b.len, "Sequences must have the same length"
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Add the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecB = mm256_loadu_pd(addr b[i])
    let vecR = mm256_add_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Add the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] + b[j]

  return result

func subSeqs*(a, b: seq[float64]): seq[float64] =
  ## Subtracts the second sequence from the first sequence element-wise using AVX vectorization.
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
    let a = @[5.0, 6.0, 7.0, 8.0]
    let b = @[1.0, 2.0, 3.0, 4.0]
    doAssert subSeqs(a, b) == @[4.0, 4.0, 4.0, 4.0]
  assert a.len == b.len, "Sequences must have the same length"
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Subtract the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecB = mm256_loadu_pd(addr b[i])
    let vecR = mm256_sub_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] - b[j]

  return result

func mulSeqs*(a, b: seq[float64]): seq[float64] =
  ## Multiplies two sequences element-wise using AVX vectorization.
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
    let a = @[1.0, 2.0, 3.0, 4.0]
    let b = @[5.0, 6.0, 7.0, 8.0]
    doAssert mulSeqs(a, b) == @[5.0, 12.0, 21.0, 32.0]
  assert a.len == b.len, "Sequences must have the same length"
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Multiply the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecB = mm256_loadu_pd(addr b[i])
    let vecR = mm256_mul_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Multiply the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] * b[j]

  return result

func divSeqs*(a, b: seq[float64]): seq[float64] =
  ## Divides the first sequence by the second sequence element-wise using AVX vectorization.
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
    let a = @[10.0, 20.0, 30.0, 40.0]
    let b = @[2.0, 4.0, 5.0, 10.0]
    doAssert divSeqs(a, b) == @[5.0, 5.0, 6.0, 4.0]
  assert a.len == b.len, "Sequences must have the same length"
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  result = newSeq[float64](a.len)

  # Divide the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecB = mm256_loadu_pd(addr b[i])
    let vecR = mm256_div_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] / b[j]

  return result

func dotSeqs*(a, b: seq[float64]): float64 =
  ## Computes the dot product of two sequences using AVX vectorization.
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
    let a = @[1.0, 2.0, 3.0, 4.0]
    let b = @[5.0, 6.0, 7.0, 8.0]
    doAssert dotSeqs(a, b) == 70.0
  assert a.len == b.len, "Sequences must have the same length"
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    sumVec: M256d = mm256_setzero_pd()
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)

  # Calculate the dot product of the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecB = mm256_loadu_pd(addr b[i])
    let vecR = mm256_mul_pd(vecA, vecB)
    sumVec = mm256_add_pd(sumVec, vecR)
    i += VecSize

  # Sum the elements of sumVec
  var sumArray: array[VecSize, float64]
  mm256_storeu_pd(addr sumArray[0], sumVec)
  var totalSum: float64 = 0.0
  for j in 0 ..< VecSize:
    totalSum += sumArray[j]

  # Calculate the dot product of the remaining elements
  for k in i ..< dataLen:
    totalSum += a[k] * b[k]

  return totalSum



proc addSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Adds a scalar value to each element in the sequence using AVX vectorization for performance.
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
    let data = @[1.0, 2.0, 3.0, 4.0]
    let result = addSeqScalar(data, 2.0)
    doAssert result == @[3.0, 4.0, 5.0, 6.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Add the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_add_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
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
  return addSeqScalar(a, b)

proc subSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Subtracts a scalar value from each element in the sequence using AVX vectorization for performance.
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
    let data = @[5.0, 6.0, 7.0, 8.0]
    let result = subSeqScalar(data, 2.0)
    doAssert result == @[3.0, 4.0, 5.0, 6.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Subtract the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_sub_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] - b

  return result

proc subScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Subtracts each element in the sequence from a scalar value using AVX vectorization for performance.
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
    let data = @[1.0, 2.0, 3.0, 4.0]
    let result = subScalarSeq(5.0, data)
    doAssert result == @[4.0, 3.0, 2.0, 1.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Subtract the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_sub_pd(vecB, vecA)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Subtract the remaining elements
  for j in i ..< dataLen:
    result[j] = b - a[j]

  return result

proc mulSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Multiplies each element in the sequence by a scalar value using AVX vectorization for performance.
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
    let data = @[1.0, 2.0, 3.0, 4.0]
    let result = mulSeqScalar(data, 2.0)
    doAssert result == @[2.0, 4.0, 6.0, 8.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Multiply the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_mul_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
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
  return mulSeqScalar(a, b)

proc divSeqScalar*(a: seq[float64], b: float64): seq[float64] =
  ## Divides each element in the sequence by a scalar value using AVX vectorization for performance.
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
    let data = @[4.0, 8.0, 12.0, 16.0]
    let result = divSeqScalar(data, 4.0)
    doAssert result == @[1.0, 2.0, 3.0, 4.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Divide the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_div_pd(vecA, vecB)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen:
    result[j] = a[j] / b

  return result

proc divScalarSeq*(b: float64, a: seq[float64]): seq[float64] =
  ## Divides a scalar value by each element in the sequence using AVX vectorization for performance.
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
    let data = @[1.0, 2.0, 4.0, 8.0]
    let result = divScalarSeq(16.0, data)
    doAssert result == @[16.0, 8.0, 4.0, 2.0]
  const
    VecSize = 4  # __m256d holds 4 float64 values
  var
    i = 0
  let
    dataLen = a.len
    alignedLen = dataLen - (dataLen mod VecSize)
    vecB = mm256_set1_pd(b)  # Broadcast `b` to all elements

  result = newSeq[float64](dataLen)

  # Divide the vectorized part using AVX
  while i < alignedLen:
    let vecA = mm256_loadu_pd(addr a[i])
    let vecR = mm256_div_pd(vecB, vecA)
    mm256_storeu_pd(addr result[i], vecR)
    i += VecSize

  # Divide the remaining elements
  for j in i ..< dataLen - 1:
    result[j] = b / a[j]

  return result
